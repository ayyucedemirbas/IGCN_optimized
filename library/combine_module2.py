import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch import nn

class GCN(nn.Module):
    def __init__(self, in_sizes=(16,16,16), hid_size=8, out_size=2):
        super().__init__()
        self.convs = nn.ModuleList([GCNConv(in_dim, hid_size) for in_dim in in_sizes])
        self.att_fc = nn.Linear(hid_size, 1, bias=False)
        self.out_convs = nn.ModuleList([GCNConv(hid_size, out_size) for _ in in_sizes])

    def forward(self, datas):  # datas: list of Data objects
        embs = []
        coefs = []
        for conv, data in zip(self.convs, datas):
            x = F.relu(conv(data.x, data.edge_index, data.edge_attr))
            x = F.dropout(x, p=0.5, training=self.training)
            embs.append(x)
            score = torch.exp(F.leaky_relu(self.att_fc(x)))
            coefs.append(score)

        coef_sum = sum(coefs)
        weights = [c/coef_sum for c in coefs]

        h = sum(w * e for w, e in zip(weights, embs))
        outs = [out_conv(h, datas[i].edge_index, datas[i].edge_attr)
                for i, out_conv in enumerate(self.out_convs)]
        out = sum(outs)
        return out, weights