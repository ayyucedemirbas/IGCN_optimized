import torch
import torch.nn.functional as F
import numpy as np

def to_sparse(x: torch.Tensor) -> torch.sparse_coo_tensor:
    idx = x.nonzero().t()
    if idx.numel() == 0:
        return torch.sparse_coo_tensor(size=x.size())
    vals = x[idx[0], idx[1]]
    return torch.sparse_coo_tensor(idx, vals, x.size())


def cosine_adj(data: torch.Tensor, topk: int) -> torch.sparse_coo_tensor:
    normed = F.normalize(data, p=2, dim=1)
    sim = torch.mm(normed, normed.t())
    vals, idx = torch.topk(sim, k=topk+1, dim=-1)
    idx = idx[:,1:]
    vals = vals[:,1:]
    row = torch.arange(data.size(0)).unsqueeze(1).repeat(1, topk).flatten()
    col = idx.flatten()
    edge_val = vals.flatten()
    adj = torch.sparse_coo_tensor(torch.stack([row, col]), edge_val, (data.size(0), data.size(0)))
    adj_t = adj.transpose(0,1)
    adj = adj + adj_t
    I = torch.eye(data.size(0), device=data.device)
    A = adj.to_dense() + I
    A = F.normalize(A, p=1, dim=1)
    return to_sparse(A)