"""
Train classifier
"""

from config import DIR_CONFIG

import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn

from tqdm import tqdm
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class ReviewDataset(Dataset):
    def __init__(self, review_df, silverlabel_df, tokenizer, num_classes=531):
        self.review_df = review_df
        self.silverlabel_df = silverlabel_df
        self.tokenizer = tokenizer
        self.num_classes = num_classes
        self.max_len = 256

    def __len__(self):
        return len(self.review_df)

    def __getitem__(self, idx):
        review_row = self.review_df.iloc[idx]
        silverlabel_row = self.silverlabel_df.iloc[idx]

        text = review_row['review_text']
        labels = silverlabel_row['silver_labels']

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )

        y = torch.zeros(self.num_classes)
        y[labels] = 1.0

        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels': y
        }

class BertClassifier(nn.Module):
    def __init__(self, num_classes=531):
        super().__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(
            self.bert.config.hidden_size,
            num_classes
        )

    def forward(self, input_ids, attention_mask):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        token_emb = output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        sum_emb = torch.sum(token_emb * mask, dim=1)
        sum_mask = mask.sum(dim=1)
        mean_emb = sum_emb / sum_mask
        logits = self.classifier(mean_emb)
        return logits


def main():
    print("=" * 60)
    print("05. TRAIN CLASSIFIER")
    print("=" * 60)
    
    # for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    review_path = DIR_CONFIG['processed_dir'] + "/proc_train_corpus.csv"
    df_review = pd.read_csv(review_path)
    
    silverlabel_path = DIR_CONFIG['processed_dir'] + "/silver_label.pt"
    silverlabel_pt = torch.load(silverlabel_path)
    df_silver = pd.DataFrame({
        "review_id": silverlabel_pt['review_id'],
        "silver_labels": silverlabel_pt['silver_labels'],
        "silver_scores": silverlabel_pt['silver_scores']
    })
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    dataset = ReviewDataset(df_review, df_silver, tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=data_collator)
    
    model = BertClassifier().to(device)
    for param in model.bert.parameters():
        param.requires_grad = False

    for layer in model.bert.encoder.layer[-3:]:
        for param in layer.parameters():
            param.requires_grad = True

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-5)
    loss_fn = nn.BCEWithLogitsLoss()

    epochs = 3
    model.train()
    for epoch in tqdm(range(1, epochs + 1)):
        total_loss = 0
        for batch in tqdm(loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Epoch {epoch}] loss = {total_loss/len(loader):.4f}")

    classifier_path = DIR_CONFIG['classifier_dir'] + "/classifier.pt"
    torch.save(model.state_dict(), classifier_path)

if __name__== '__main__':
    main()