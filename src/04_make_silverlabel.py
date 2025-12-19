"""
Make Silverlabel
"""

from config import DIR_CONFIG

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

def main():
    print("=" * 60)
    print("04. MAKE SILVERLABEL")
    print("=" * 60)
    
    review_emb_path = DIR_CONFIG['processed_dir'] + "/bert_review_emb.pt"
    review_emb = torch.load(review_emb_path)
    review_ids = review_emb['ids']
    print(review_emb['embeddings'].shape)
    class_emb_path = DIR_CONFIG['processed_dir'] + "/GAT_class_emb.pt"
    class_emb = torch.load(class_emb_path)

    # Cosine Similarity
    review_emb = normalize(review_emb['embeddings'], axis=1)
    class_emb = normalize(class_emb, axis=1)
    sim_matrix = review_emb @ class_emb.T

    # make silver label (top-K)
    labels, scores = [], []
    top_k = 3

    for row in sim_matrix:
        idx = np.argsort(-row)[:top_k]

        labels.append(idx.tolist())
        scores.append(row[idx].tolist())

    df_silver = pd.DataFrame({
        "review_id": review_ids,
        "silver_labels": labels,
        "silver_scores": scores
    })

    silver_label_pt_path = DIR_CONFIG['processed_dir'] + "/silver_label.pt"
    torch.save(df_silver, silver_label_pt_path)
    # For human
    silver_label_csv_path = DIR_CONFIG['processed_dir'] + "/silver_label.csv"
    df_silver.to_csv(silver_label_csv_path, index=False)


if __name__== '__main__':
    main()