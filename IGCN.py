import os
import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch_geometric.data import Data
from library.combine_module2 import GCN
import main

BASE_PATH = ''
DATA_DIR = os.path.join(BASE_PATH, 'dataset')
EDGE_DIR = os.path.join(BASE_PATH, 'data', 'sample_data')
SAVE_FIG = 'attention_weights.png'
HID_SIZE = 64
LR = 0.005
EPOCHS = 400
FOLDS = 5


def load_data():
    labels = pd.read_csv(os.path.join(DATA_DIR, 'labels_.csv'), header=None).iloc[:,0].values
    features, edges = [], []
    for i, omic in enumerate(['mRNA','DNA','miRNA']):
        df = pd.read_csv(os.path.join(DATA_DIR, f"{i+1}_.csv"), header=None).values
        features.append(torch.tensor(df, dtype=torch.float))
        with open(os.path.join(EDGE_DIR, f'edges_n_{omic}.pkl'), 'rb') as f:
            edges.append(torch.tensor(pickle.load(f), dtype=torch.long))
    return features, edges, labels


def train_and_evaluate(features, edges, labels):
    torch.manual_seed(42)
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
    metrics = {'acc': [], 'wf1': [], 'mf1': [], 'mcc': []}
    all_weights = []
    run_times = []

    device = torch.device('cpu')
    feats = [f.to(device) for f in features]
    edgs = [e.to(device) for e in edges]
    y = torch.tensor(labels, dtype=torch.long, device=device)

    for fold, (train_idx, test_idx) in enumerate(skf.split(feats[0], labels)):
        DATA = []
        start_time = time.time()
        for f, e in zip(feats, edgs):
            data = Data(x=f, edge_index=e, edge_attr=torch.ones(e.size(1)), y=y)
            train_mask = torch.zeros_like(y, dtype=torch.bool)
            test_mask = torch.zeros_like(y, dtype=torch.bool)
            train_mask[train_idx] = True
            test_mask[test_idx] = True
            data.train_mask = train_mask
            data.test_mask = test_mask
            DATA.append(data)

        model = GCN(in_sizes=[f.shape[1] for f in feats], hid_size=HID_SIZE,
                    out_size=len(np.unique(labels))).to(device)
        optimizer = Adam(model.parameters(), lr=LR)
        criterion = torch.nn.CrossEntropyLoss()

        model.train()
        for epoch in range(EPOCHS):
            optimizer.zero_grad()
            out, coefs = model(DATA)
            loss = criterion(out[DATA[0].train_mask], y[DATA[0].train_mask])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            out, coefs = model(DATA)
        preds = out.argmax(dim=1).cpu().numpy()
        gt = labels[test_idx]
        pr = preds[test_idx]

        metrics['acc'].append(accuracy_score(gt, pr))
        metrics['wf1'].append(f1_score(gt, pr, average='weighted'))
        metrics['mf1'].append(f1_score(gt, pr, average='macro'))
        metrics['mcc'].append(matthews_corrcoef(gt, pr))

        if fold == FOLDS - 1:
            all_weights = [c.cpu().numpy() for c in coefs]
        end_time = time.time()
        run_times.append(end_time - start_time)

    for k, v in metrics.items():
        print(f"{k}: {np.mean(v):.3f} ± {np.std(v):.3f}")

    return all_weights, labels, preds, test_idx, run_times


if __name__ == '__main__':
    feats, edgs, labels = load_data()
    weights, labels, preds, test_idx, run_times = train_and_evaluate(feats, edgs, labels)

    print(f"time: {np.mean(run_times):.3f}")

    coef1, coef2, coef3 = weights
    y_test = labels[test_idx]
    pred_test = preds[test_idx]
    ids_per_class = [np.where((y_test == cls) & (pred_test == cls))[0][:10]
                     for cls in np.unique(labels)]
    ids_concat = np.concatenate(ids_per_class)

    Coef1 = coef1[test_idx, 0][ids_concat]
    Coef2 = coef2[test_idx, 0][ids_concat]
    Coef3 = coef3[test_idx, 0][ids_concat]

    plt.figure(figsize=(50, 6))
    plt.plot(Coef1, '--k^', label='mRNA', markersize=8)
    plt.plot(Coef2, '-ro', label='DNA meth.', markersize=8)
    plt.plot(Coef3, '-go', label='miRNA', markersize=8)
    plt.xlim(-0.5, len(Coef1) - 0.5)
    plt.legend(loc='center right', fontsize=19)
    plt.ylabel('Attention coefficients', fontsize=19, fontweight='bold')
    plt.xlabel('Samples', fontsize=19, fontweight='bold')
    plt.yticks(fontsize=16, fontweight='bold')
    plt.xticks(fontsize=16, fontweight='bold')
    plt.title('TCGA-BRCA', fontsize=24, fontweight='bold')
    plt.tight_layout()
    plt.savefig(SAVE_FIG, dpi=300)
    plt.show()
