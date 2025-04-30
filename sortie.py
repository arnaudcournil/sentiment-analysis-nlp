# sentiment_word_vector_model.py
import os
import re
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# --------------- UTILS ---------------
def tokenize(text):
    text = re.sub(r"<br />", " ", text)
    return re.findall(r"\b\w+\b", text.lower())

def build_vocab(reviews, vocab_size=5000, min_freq=5):
    word_counts = {}
    for text in reviews:
        for word in tokenize(text):
            word_counts[word] = word_counts.get(word, 0) + 1
    sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, count in sorted_words:
        if count >= min_freq and len(vocab) < vocab_size:
            vocab[word] = len(vocab)
    return vocab

def encode(text, vocab, max_len=400):
    tokens = tokenize(text)
    indices = [vocab.get(token, vocab["<UNK>"]) for token in tokens[:max_len]]
    if len(indices) < max_len:
        indices += [vocab["<PAD>"]] * (max_len - len(indices))
    return indices

# --------------- DATASET ---------------
class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = encode(self.texts[idx], self.vocab)
        label = self.labels[idx]
        return torch.tensor(text), torch.tensor(label, dtype=torch.float)

# --------------- MODEL ---------------
class SentimentAwareEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.biases = nn.Parameter(torch.zeros(vocab_size))
        self.sentiment_regressor = nn.Linear(embedding_dim, 1)

    def forward(self, docs, thetas):
        word_vecs = self.embeddings(docs)              # [B, L, D]
        avg_vec = word_vecs.mean(dim=1)                # [B, D]
        sentiment_logits = self.sentiment_regressor(avg_vec).squeeze(1)
        sentiment_probs = torch.sigmoid(sentiment_logits)
        return sentiment_probs

# --------------- TRAINING LOOP ---------------
def train_model(model, train_loader, optimizer, epochs=5, device="cpu"):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss, total_acc = 0, 0
        for docs, labels in train_loader:
            docs, labels = docs.to(device), labels.to(device)
            batch_size = docs.shape[0]
            thetas = torch.randn(batch_size, model.embeddings.embedding_dim).to(device)

            optimizer.zero_grad()
            preds = model(docs, thetas)
            loss = F.binary_cross_entropy(preds, labels)
            loss.backward()
            optimizer.step()

            acc = ((preds > 0.5) == labels).float().mean()
            total_loss += loss.item()
            total_acc += acc.item()

        avg_loss = total_loss / len(train_loader)
        avg_acc = total_acc / len(train_loader)
        print(f"Epoch {epoch + 1} - Loss: {avg_loss:.4f} - Acc: {avg_acc:.4f}")

# --------------- MAIN SCRIPT ---------------
if __name__ == "__main__":
    # Dummy example to demonstrate
    print("Loading example dataset...")
    pos_texts = ["This movie was fantastic! I loved it." for _ in range(500)]
    neg_texts = ["Awful film. Completely boring and a waste of time." for _ in range(500)]
    texts = pos_texts + neg_texts
    labels = [1.0] * 500 + [0.0] * 500

    vocab = build_vocab(texts, vocab_size=5000)
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

    train_dataset = IMDBDataset(train_texts, train_labels, vocab)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = SentimentAwareEmbeddingModel(vocab_size=len(vocab), embedding_dim=50)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Training model...")
    train_model(model, train_loader, optimizer, epochs=10)
    print("Done.")
