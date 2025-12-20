"""
BERT Embedding
"""

from config import DIR_CONFIG

import torch
import pickle
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer


def main():
    print("=" * 60)
    print("02. TF-IDF EMBEDDING")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    class_path = DIR_CONFIG['processed_dir'] + "/proc_class.csv"
    df_class = pd.read_csv(class_path)

    class_ids = df_class['class_index'].tolist()
    df_class["class_text"] = (
        df_class["class_name_raw"].str.replace("_", " ", regex=False)
        + " "
        + df_class["keywords_raw"].str.replace(",", " ", regex=False)
    )
    class_texts = df_class['class_text'].tolist()

    
    review_path = DIR_CONFIG['processed_dir'] + "/proc_train_corpus.csv"
    df_review = pd.read_csv(review_path)
    
    review_ids = df_review['review_id'].tolist()
    review_texts = df_review['review_text'].tolist()
    
    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), stop_words="english", norm="l2")

    all_texts = review_texts + class_texts
    tfidf_all = vectorizer.fit_transform(all_texts)

    len_review = len(review_texts)
    review_vecs = tfidf_all[:len_review]
    class_vecs = tfidf_all[len_review:]

    print(review_vecs.shape)
    print(class_vecs.shape)

    class_emb_path = DIR_CONFIG['processed_dir'] + "/tfidf_class_emb.pt"
    torch.save({"ids": class_ids, "embeddings": torch.from_numpy(class_vecs.toarray()).float()}, class_emb_path)

    review_emb_path = DIR_CONFIG['processed_dir'] + "/tfidf_review_emb.pkl"
    with open(review_emb_path, "wb") as f:
        pickle.dump(
            {
                "ids": review_ids,
                "embeddings": review_vecs
            },
            f
        )

if __name__== '__main__':
    main()