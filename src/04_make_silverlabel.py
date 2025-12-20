"""
Make Silverlabel
"""

from config import DIR_CONFIG

import torch
import pickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize

def row_minmax(x):
    return (x - x.min(axis=1, keepdims=True)) / (
        x.max(axis=1, keepdims=True) - x.min(axis=1, keepdims=True) + 1e-8
    )

def main():
    print("=" * 60)
    print("04. MAKE SILVERLABEL")
    print("=" * 60)
    
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

    # tfidf_norm = row_minmax(tfidf_sim_matrix)
    # bert_norm = row_minmax(bert_sim_matrix)
    sim_matrix = 0.3 * tfidf_sim_matrix + 0.7 * bert_sim_matrix

    # make silver label (top-K)
    labels, scores = [], []
    threshold = 0.3
    min_k = 1

    for row in sim_matrix:
        idx = np.where(row >= threshold)[0]

        if len(idx) < min_k:
            idx = np.argsort(-row)[:min_k]

        labels.append(idx.tolist())
        scores.append(row[idx].tolist())
    
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