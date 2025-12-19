"""
Data PreProcessing
"""

from config import DIR_CONFIG

import pandas as pd

def main():
    print("=" * 60)
    print("01. DATA PREPROCESSING")
    print("=" * 60)

    # preprocess train_corpus
    train_corpus_path = DIR_CONFIG['raw_dir'] + "/train/train_corpus.txt"
    
    df_train_corpus = pd.read_csv(train_corpus_path, sep="\t", header=None, names=["review_id", "review_text"])
    
    proc_train_corpus_path = DIR_CONFIG['processed_dir'] + "/proc_train_corpus.csv"
    df_train_corpus.to_csv(proc_train_corpus_path, index=False)

    # preprocess class_related_keywords
    class_path = DIR_CONFIG['raw_dir'] + "/classes.txt"
    class_keywords_path = DIR_CONFIG['raw_dir'] + "/class_related_keywords.txt"
    
    df_class = pd.read_csv(class_path, sep="\t", header=None, names=["class_index", "class_name_raw"])
    df_class_keywords = pd.read_csv(class_keywords_path, sep=":", header=None, names=["class_name_raw", "keywords_raw"])
    df_proc_class = pd.merge(df_class, df_class_keywords, on='class_name_raw', how='left')
    # df_proc_class["class_name"] = df_proc_class["class_name_raw"].str.replace("_", " ", regex=False)
    # df_proc_class["keywords"] = df_proc_class["keywords_raw"].str.replace(",", " ", regex=False)
    
    proc_class_path = DIR_CONFIG['processed_dir'] + "/proc_class.csv"
    df_proc_class.to_csv(proc_class_path, index=False)

if __name__== '__main__':
    main()