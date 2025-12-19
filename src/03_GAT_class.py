"""
GNN for class
"""

from config import DIR_CONFIG

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from tqdm import tqdm

def make_pos_edges(A):
    pos_edges = set()

    for (parent, child), val in np.ndenumerate(A):
        if val == 1:
            # 1-hop
            pos_edges.add((parent, child))

            # 2-hop
            grandparents = np.where(A[:, parent] == 1)[0]
            for grandparent in grandparents:
                pos_edges.add((child, grandparent))

    return list(pos_edges)

def sample_neg_edges(n_nodes, pos_edge_set, n_samples):
    neg_edges = set()
    while len(neg_edges) < n_samples:
        u = torch.randint(0, n_nodes, (1,)).item()
        v = torch.randint(0, n_nodes, (1,)).item()
        
        if u == v:
            continue
        if (u, v) not in pos_edge_set:
            neg_edges.add((u, v))
            
    return torch.tensor(list(neg_edges))

class ClassGAT(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=256, out_dim=768, heads=4, dropout=0.2):
        super().__init__()

        self.gat1 = GATConv(
            in_channels=in_dim,
            out_channels=hidden_dim,
            heads=heads,
            concat=True,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(hidden_dim * heads)

        self.gat2 = GATConv(
            in_channels=hidden_dim * heads,
            out_channels=out_dim,
            heads=1,
            concat=False,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(out_dim)

        self.res1 = nn.Linear(in_dim, hidden_dim * heads)
        self.res2 = nn.Linear(hidden_dim * heads, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h0 = x
        h = F.dropout(x, p=self.dropout, training=self.training)
        h = self.gat1(h, edge_index)
        h = self.norm1(h)
        h = F.relu(h)
        h = h + self.res1(h0)

        h1 = h
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.gat2(h, edge_index)
        h = self.norm2(h)
        h = h + self.res2(h1)

        return h

def main():
    print("=" * 60)
    print("03. GAT FOR CLASS")
    print("=" * 60)
    
    # for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    class_hierarchy_path = DIR_CONFIG['raw_dir'] + "/class_hierarchy.txt"
    class_emb_path = DIR_CONFIG['processed_dir'] + "/bert_class_emb.pt"
    class_emb_dict = torch.load(class_emb_path)
    class_emb = class_emb_dict['embeddings']

    edges = np.genfromtxt(class_hierarchy_path, dtype=np.int32)
    edges = np.array(list(map(lambda x: [x[0], x[1]], edges)))
    src, dst = edges[:, 0], edges[:, 1]
    edges_bi = np.vstack([
        np.stack([src, dst], axis=1),
        np.stack([dst, src], axis=1),
    ])
    edge_index = torch.tensor(edges_bi, dtype=torch.long).t().contiguous()

    # make adjacency matrix (directed)
    n_class = class_emb.shape[0]
    A = np.eye(n_class, dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1

    # GAT Learning for Class
    print("GAT Learning for Class")
    
    epochs = 100
    lr = 1e-3

    model = ClassGAT(in_dim=768, hidden_dim=256, out_dim=768, heads=4, dropout=0.2)
    model.to(device)
    x = class_emb.to(device)
    edge_index = edge_index.to(device)  # (n_edges, 2)
    pos_edge_index = torch.tensor(make_pos_edges(A), dtype=torch.long).to(device)  # (n_pos_edges, 2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    pos_edge_set = set(
        (int(u), int(v)) for u, v in pos_edge_index
    )

    num_nodes = x.size(0)
    num_pos = len(pos_edge_index)

    for epoch in tqdm(range(1, epochs + 1)):
        model.train()
        optimizer.zero_grad()

        z = model(x, edge_index)  # (N, d)

        # positive scores
        u_pos, v_pos = pos_edge_index[:, 0], pos_edge_index[:, 1]
        pos_score = (z[u_pos] * z[v_pos]).sum(dim=1)

        # negative edges
        neg_edge_index = sample_neg_edges(
            num_nodes, pos_edge_set, num_pos
        ).to(device)  # (n_neg_edges, 2)
        u_neg, v_neg = neg_edge_index[:, 0], neg_edge_index[:, 1]
        neg_score = (z[u_neg] * z[v_neg]).sum(dim=1)

        # labels
        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat([
            torch.ones_like(pos_score),
            torch.zeros_like(neg_score),
        ])

        loss = loss_fn(scores, labels)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"[Epoch {epoch:03d}] loss = {loss.item():.4f}")

            cos = F.cosine_similarity(z[u_pos], z[v_pos])
            cos_neg = F.cosine_similarity(z[u_neg], z[v_neg])
            print(cos.mean(), cos_neg.mean())

    print("pos score mean:", pos_score.mean().item())
    print("neg score mean:", neg_score.mean().item())
    torch.save(model.state_dict(), "class_GAT.pt")

    # GAT Inference for Class
    print("GAT Inference for Class")
    model.eval()
    with torch.no_grad():
        class_emb = model(x, edge_index)

    GAT_emb_path = DIR_CONFIG['processed_dir'] + "/GAT_class_embeddings.pt"
    torch.save(class_emb.cpu(), GAT_emb_path)

if __name__== '__main__':
    main()