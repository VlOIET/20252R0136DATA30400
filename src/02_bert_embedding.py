"""
BERT Embedding
"""

from config import DIR_CONFIG

import torch
import pandas as pd
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

# function for bert input
def make_bert_input(row):
    
    class_name = row['class_name_raw']
    class_keywords = row['keywords_raw'].replace(",", ", ")

    bert_input = f"""
    This text describes the product category "{class_name}". It is related to {class_keywords}.
    """
    
    return bert_input


def main():
    print("=" * 60)
    print("02. BERT EMBEDDING")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    MODEL_NAME = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertModel.from_pretrained(MODEL_NAME).eval().to(device)

    # BERT Embedding for class
    print("BERT Embedding for Class")
    class_path = DIR_CONFIG['processed_dir'] + "/proc_class.csv"
    df_class = pd.read_csv(class_path)
    df_class['bert_input'] = df_class.apply(make_bert_input, axis=1)
    
    class_ids = df_class['class_index'].tolist()
    class_emb_list = []

    batch_size = 64
    for i in tqdm(range(0, len(df_class), batch_size)):
        batch = df_class.iloc[i: i + batch_size]
        
        encoded = tokenizer(
            batch['bert_input'].tolist(),
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(model.device)
            
        with torch.no_grad():
            output = model(**encoded)

        # mean pooling
        token_emb = output.last_hidden_state
        input_mask_expanded = encoded["attention_mask"].unsqueeze(-1).expand(token_emb.size()).float()
        sum_emb = torch.sum(token_emb * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        mean_emb = sum_emb / sum_mask
        class_emb_list.append(mean_emb.cpu())

    class_emb_path = DIR_CONFIG['processed_dir'] + "/bert_class_emb.pt"
    torch.save({"ids": class_ids, "embeddings": torch.cat(class_emb_list, dim=0)}, class_emb_path)

    # BERT Embedding for review
    print("BERT Embedding for Review")
    review_path = DIR_CONFIG['processed_dir'] + "/proc_train_corpus.csv"
    df_review = pd.read_csv(review_path)
    
    review_ids = df_review['review_id'].tolist()
    review_emb_list = []

    batch_size = 64
    for i in tqdm(range(0, len(df_class), batch_size)):
        batch = df_review.iloc[i: i + batch_size]
        
        encoded = tokenizer(
            batch['review_text'].tolist(),
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(model.device)
            
        with torch.no_grad():
            output = model(**encoded)

        # mean pooling
        token_emb = output.last_hidden_state
        input_mask_expanded = encoded["attention_mask"].unsqueeze(-1).expand(token_emb.size()).float()
        sum_emb = torch.sum(token_emb * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        mean_emb = sum_emb / sum_mask
        review_emb_list.append(mean_emb.cpu())

    review_emb_path = DIR_CONFIG['processed_dir'] + "/bert_review_emb.pt"
    torch.save({"ids": class_ids, "embeddings": torch.cat(review_emb_list, dim=0)}, review_emb_path)

if __name__== '__main__':
    main()