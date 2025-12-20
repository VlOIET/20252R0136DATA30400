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
from collections import Counter
from collections import defaultdict
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
        labels = silverlabel_row['labels']

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
    
    def update_labels(self, new_label_df):
        self.silverlabel_df = new_label_df

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

def compute_pos_weight(df_labels, num_classes, device):
    freq = Counter()
    for labels in df_labels["labels"]:
        for l in labels:
            freq[l] += 1

    pos_weight = torch.ones(num_classes)

    for c, f in freq.items():
        pos_weight[c] = 1.0 / np.log(1.0 + f)
    
    print("top label freq:", freq.most_common(5))

    return pos_weight.to(device)

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
    df_labels = pd.DataFrame({
        "review_id": silverlabel_pt['review_id'],
        "labels": silverlabel_pt['silver_labels']
    })

    hier_path = DIR_CONFIG['raw_dir'] + "/class_hierarchy.txt"
    child2parents = build_child2parents(hier_path)

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    dataset = ReviewDataset(df_review, df_labels, tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer)
    loader_train = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=data_collator)
    loader_eval  = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=data_collator)

    model = BertClassifier().to(device)
    for param in model.bert.parameters():
        param.requires_grad = True

    """ for layer in model.bert.encoder.layer[-3:]:
        for param in layer.parameters():
            param.requires_grad = True """

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-5)
    loss_fn = nn.BCEWithLogitsLoss()

    epochs = 1
    conf_schedule = {1: 0.8, 2: 0.75, 3: 0.7}

    for epoch in tqdm(range(1, epochs + 1)):
        # pos_weight = compute_pos_weight(df_labels, 531, device)
        loss_fn = nn.BCEWithLogitsLoss()     

        model.train()
        total_loss = 0

        for batch in tqdm(loader_train):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Epoch {epoch}] loss = {total_loss/len(loader_train):.4f}")

    """         # Generate pseudo label
        conf = conf_schedule[epoch]
        print(f"Generate pseudo labels with conf >= {conf}")

        model.eval()

        new_labels = []
        new_scores = []

        for batch in tqdm(loader_eval):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits)

            for p in probs:
                idx = (p >= conf).nonzero(as_tuple=True)[0]
                new_labels.append(idx.cpu().tolist())
                new_scores.append(p[idx].cpu().tolist())

        # Merge silver + pseudo
        max_k = 3
        merged_labels = []
        for i in range(len(df_labels)):
            base = set(df_labels.iloc[i]["labels"])
            pseudo_labels = new_labels[i]
            pseudo_scores = new_scores[i]

            score_map = {l: 0.6 for l in base}
            for l, s in zip(pseudo_labels, pseudo_scores):
                score_map[l] = max(score_map.get(l, 0), s)

            sorted_labels = sorted(score_map.items(), key=lambda x: x[1], reverse=True)

            labels = [l for l, _ in sorted_labels[:10]]

            # hierarchy pruning 추가
            labels = remove_all_ancestors(labels, child2parents)
            labels = labels[:max_k]

            # fallback
            if len(labels) == 0:
                labels = [sorted_labels[0][0]]

            merged_labels.append(labels)

        df_labels["labels"] = merged_labels
        dataset.update_labels(df_labels)

        lens = [len(l) for l in merged_labels]
        s = pd.Series(lens)

        print("\n[MERGED LABEL STATS]")
        print(s.describe())
        print("\nvalue counts (top 10):")
        print(s.value_counts().sort_index().head(10)) """

    classifier_path = DIR_CONFIG['classifier_dir'] + "/classifier.pt"
    torch.save(model.state_dict(), classifier_path)

if __name__== '__main__':
    main()