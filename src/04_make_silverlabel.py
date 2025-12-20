"""
Make Silverlabel
"""

from config import DIR_CONFIG

import torch
import pickle
import numpy as np
import pandas as pd

from collections import Counter
from collections import defaultdict
from sklearn.preprocessing import normalize

def row_minmax(x):
    return (x - x.min(axis=1, keepdims=True)) / (
        x.max(axis=1, keepdims=True) - x.min(axis=1, keepdims=True) + 1e-8
    )

def build_child2parents(hierarchy_path):
    child2parents = defaultdict(list)

    with open(hierarchy_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parent, child = map(int, line.split())
            child2parents[child].append(parent)

    return dict(child2parents)

def remove_all_ancestors(labels, child2parents):
    labels = set(labels)

    changed = True
    while changed:
        changed = False
        for l in list(labels):
            for p in child2parents.get(l, []):
                if p in labels:
                    labels.remove(p)
                    changed = True

    return sorted(labels)

def main():
    print("=" * 60)
    print("04. MAKE SILVERLABEL")
    print("=" * 60)

    # hierarchy
    hier_path = DIR_CONFIG['raw_dir'] + "/class_hierarchy.txt"
    child2parents = build_child2parents(hier_path)
    
    # TFIDF
    tfidf_review_emb_path = DIR_CONFIG['processed_dir'] + "/tfidf_review_emb.pkl"
    with open(tfidf_review_emb_path, "rb") as f:
        tfidf_review_emb = pickle.load(f)
    tfidf_review_vecs = tfidf_review_emb["embeddings"]
    tfidf_class_emb_path  = DIR_CONFIG['processed_dir'] + "/tfidf_class_emb.pt"
    tfidf_class_emb = torch.load(tfidf_class_emb_path)
    tfidf_class_vecs = tfidf_class_emb["embeddings"].numpy()

    tfidf_sim_matrix = tfidf_review_vecs @ tfidf_class_vecs.T

    # BERT
    bert_review_emb_path = DIR_CONFIG['processed_dir'] + "/bert_review_emb.pt"
    bert_review_emb = torch.load(bert_review_emb_path)
    review_ids = bert_review_emb['ids']
    bert_class_emb_path = DIR_CONFIG['processed_dir'] + "/GAT_class_emb.pt"
    bert_class_emb = torch.load(bert_class_emb_path)

    # Cosine Similarity
    bert_review_emb = normalize(bert_review_emb['embeddings'], axis=1)
    bert_class_emb = normalize(bert_class_emb, axis=1)
    
    bert_sim_matrix = bert_review_emb @ bert_class_emb.T

    tfidf_norm = row_minmax(tfidf_sim_matrix)
    bert_norm = row_minmax(bert_sim_matrix)
    sim_matrix = 0.3 * tfidf_norm + 0.7 * bert_norm

    # make silver label (top-K)
    freq = Counter()
    labels, scores = [], []
    min_k = 1
    max_k = 3
    candidate_k = 10
    alpha = 0.7

    for row in sim_matrix:
        cand = np.argsort(-row)[:candidate_k]

        penalized = [(c, row[c] - alpha * np.log(1 + freq[c])) for c in cand]
        penalized.sort(key=lambda x: x[1], reverse=True)

        selected = []
        for c, _ in penalized:
            selected.append(c)
            selected = remove_all_ancestors(selected, child2parents)

            if len(selected) >= max_k:
                break

        if len(selected) < min_k:
            selected = [cand[0]]

        labels.append(selected)
        scores.append(row[selected].tolist())

        for c in selected:
            freq[c] += 1

    df_silver = pd.DataFrame({
        "review_id": review_ids,
        "silver_labels": labels,
        "silver_scores": scores
    })

    # for choose threshold 
    lens = [len(l) for l in labels]
    print(pd.Series(lens).describe())

    silver_label_pt_path = DIR_CONFIG['processed_dir'] + "/silver_label.pt"
    torch.save(df_silver, silver_label_pt_path)
    # For human
    silver_label_csv_path = DIR_CONFIG['processed_dir'] + "/silver_label.csv"
    df_silver.to_csv(silver_label_csv_path, index=False)


if __name__== '__main__':
    main()