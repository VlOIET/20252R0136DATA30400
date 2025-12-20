"""
Test classifier
"""

from config import DIR_CONFIG

import os
import csv
import torch
import random
import torch.nn as nn

from tqdm import tqdm
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# --- Paths ---
TEST_DIR = DIR_CONFIG['raw_dir'] + "/test"
TEST_CORPUS_PATH = os.path.join(TEST_DIR, "test_corpus.txt")  # product_id \t text
SUBMISSION_PATH = "submission.csv"  # output file

# --- Constants ---
NUM_CLASSES = 531  # total number of classes (0–530)
MIN_LABELS = 2     # minimum number of labels per sample
MAX_LABELS = 3     # maximum number of labels per sample

# --- Load test corpus ---
def load_corpus(path):
    """Load test corpus into {pid: text} dictionary."""
    pid2text = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                pid, text = parts
                pid2text[pid] = text
    return pid2text

pid2text_test = load_corpus(TEST_CORPUS_PATH)
pid_list_test = list(pid2text_test.keys())

print("=" * 60)
print("06. TEST CLASSIFIER")
print("=" * 60)

# dataset
class TestDataset(Dataset):
    def __init__(self, pid_list, pid2text, tokenizer):
        self.pid_list = pid_list
        self.pid2text = pid2text
        self.tokenizer = tokenizer
        self.max_len = 256

    def __len__(self):
        return len(self.pid_list)

    def __getitem__(self, idx):
        pid = self.pid_list[idx]
        text = self.pid2text[pid]

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len
        )

        return {
            "pid": pid,
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }

# Classifier
class BertClassifier(nn.Module):
    def __init__(self, num_classes=531):
        super().__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        hidden = self.bert.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, input_ids, attention_mask, return_emb=False):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # legacy mean pooling code
        """ token_emb = output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        sum_emb = torch.sum(token_emb * mask, dim=1)
        sum_mask = mask.sum(dim=1)
        mean_emb = sum_emb / sum_mask
        logits = self.classifier(mean_emb)
        
        if return_emb:
            return logits, mean_emb """
        
        cls_emb = output.last_hidden_state[:, 0]
        logits = self.classifier(cls_emb)
        
        return logits
        
# inference        
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
data_collator = DataCollatorWithPadding(tokenizer)

def collate_fn(batch):
    pids = [item["pid"] for item in batch]

    features = [
        {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
        }
        for item in batch
    ]

    batch_enc = data_collator(features)
    batch_enc["pid"] = pids

    return batch_enc

dataset = TestDataset(pid_list_test, pid2text_test, tokenizer)
loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

model = BertClassifier().to(device)
classifier_path = DIR_CONFIG['classifier_dir'] + "/classifier.pt"
state_dict = torch.load(classifier_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

all_pids, all_labels = [], []
with torch.no_grad():
    for batch in tqdm(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits)

        # top-k (max 3)
        conf = 0.7
        min_labels = 2
        max_labels = 3

        for pid, p in zip(batch["pid"], probs):
            """ idx = (p >= conf).nonzero(as_tuple=True)[0]

            if len(idx) < min_labels:
                idx = torch.topk(p, k=min_labels).indices

            if len(idx) > max_labels:
                idx = idx[torch.argsort(p[idx], descending=True)[:max_labels]]

            labels = idx.cpu().tolist() """

            topk = torch.topk(p, k=3).indices
            labels = topk.cpu().tolist()

            all_pids.append(pid)
            all_labels.append(labels)

# --- Save submission file ---
with open(SUBMISSION_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "label"])
    for pid, labels in zip(all_pids, all_labels):
        writer.writerow([pid, ",".join(map(str, labels))])

print(f"Dummy submission file saved to: {SUBMISSION_PATH}")
print(f"Total samples: {len(all_pids)}, Classes per sample: {MIN_LABELS}-{MAX_LABELS}")