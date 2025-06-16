import os
import pickle
import pandas as pd
import torch
from utils import cosine_adj

BASE_PATH = ''
DATASET = 'sample_data'
NODE_EMD = os.path.join('data', DATASET)
CSV_DIR = os.path.join('dataset')

for i, omic in enumerate(['mRNA','DNA','miRNA']):
    csv_file = os.path.join(CSV_DIR, f"{i+1}_.csv")
    df = pd.read_csv(csv_file, header=None).values
    x = torch.tensor(df, dtype=torch.float)
    topk = int(x.size(0) * 0.05)
    adj = cosine_adj(x, topk)
    edge_idx = adj._indices()
    out_file = os.path.join('data', DATASET, f'edges_n_{omic}.pkl')
    with open(out_file, 'wb') as f:
        pickle.dump(edge_idx, f)
