import os
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

base_dir = "aclImdb"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

def load_reviews_from_dir(directory, label):
    data = []
    for fname in os.listdir(directory):
        if fname.endswith(".txt"):
            with open(os.path.join(directory, fname), encoding="utf-8") as f:
                text = f.read()
                data.append((text, label))
    return data

def load_all_data():
    train_pos = load_reviews_from_dir(os.path.join(train_dir, "pos"), 1)
    train_neg = load_reviews_from_dir(os.path.join(train_dir, "neg"), 0)
    test_pos  = load_reviews_from_dir(os.path.join(test_dir,  "pos"), 1)
    test_neg  = load_reviews_from_dir(os.path.join(test_dir,  "neg"), 0)
    return train_pos + train_neg + test_pos + test_neg

raw_data = load_all_data()
df = pd.DataFrame(raw_data, columns=["text", "label"])

def clean_text(text):
    text = re.sub(r"<br\s*/?>", " ", text)  # remplace <br> et <br /> par un espace
    text = re.sub(r"\s+", " ", text)        # normalise les espaces
    return text.strip()

df["text"] = df["text"].apply(clean_text)

vectorizer = CountVectorizer(max_features=5050, token_pattern=r"(?u)\b\w+\b")
vectorizer.fit(df["text"])
full_vocab = vectorizer.get_feature_names_out()

# --- Exclusion des 50 mots les plus fréquents comme dans l'article ---
excluded_words = full_vocab[:50]
vocab = full_vocab[50:]  # Top 5000 mots après exclusion des 50 premiers

def filter_tokens(text, vocab_set):
    tokens = text.split()
    return " ".join([tok for tok in tokens if tok in vocab_set])

vocab_set = set(vocab)
df["filtered_text"] = df["text"].apply(lambda x: filter_tokens(x, vocab_set))

# --- Résultat final : df avec colonne "filtered_text" et "label" ---
print(df[["filtered_text", "label"]].head(15))
