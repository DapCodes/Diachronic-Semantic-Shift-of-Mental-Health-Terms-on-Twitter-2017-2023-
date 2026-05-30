"""
01_preprocessing.py
====================
Preprocessing pipeline for Mental Health Twitter dataset.
Cleans tweets and splits corpus by year for diachronic analysis.

Author  : Your Name
Dataset : MH_Campaign_Tweets_Tokenised_1723 (Kaggle, 2017-2023)
"""

import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

os.makedirs('data/processed', exist_ok=True)

# ── Load Dataset ──────────────────────────────────────────────
print("=" * 55)
print("  FASE 1: Preprocessing")
print("=" * 55)

print("\n[1/4] Loading dataset...")
df = pd.read_csv(
    'data/raw/twitter-2017-2023/MH_Campaign_Tweets_Tokenised_1723.csv'
)

# Parse tanggal & tahun
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['year'] = df['Date'].dt.year
df = df.dropna(subset=['year', 'tweet'])
df['year'] = df['year'].astype(int)

print(f"    Total tweets loaded : {len(df):,}")
print(f"    Rentang tahun       : {df['year'].min()}–{df['year'].max()}")

# ── Distribusi per tahun ──────────────────────────────────────
print("\n[2/4] Distribusi data per tahun:")
year_counts = df['year'].value_counts().sort_index()
for year, count in year_counts.items():
    bar = "█" * (count // 5000)
    print(f"    {year}: {count:>7,}  {bar}")

# ── Cleaning ──────────────────────────────────────────────────
stop_words = set(stopwords.words('english'))

def clean_tweet(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)      # hapus URL
    text = re.sub(r'@\w+', '', text)                 # hapus mention
    text = re.sub(r'#(\w+)', r'\1', text)            # hapus simbol #
    text = re.sub(r'[^\w\s]', '', text)              # hapus tanda baca
    text = re.sub(r'\d+', '', text)                  # hapus angka
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text: str) -> list:
    tokens = word_tokenize(text)
    return [t for t in tokens if t not in stop_words and len(t) > 2]

print("\n[3/4] Cleaning & tokenizing tweets...")
df['clean_text'] = df['tweet'].apply(clean_tweet)
df['tokens']     = df['clean_text'].apply(tokenize)
print(f"    Selesai: {len(df):,} tweets diproses")

# ── Cek kata kunci ────────────────────────────────────────────
print("\n    Distribusi kata kunci target:")
keywords = ['healing', 'toxic', 'burnout']
for kw in keywords:
    n = df['tweet'].str.contains(kw, case=False, na=False).sum()
    print(f"      '{kw}': {n:,} tweets")

# ── Simpan corpus per tahun ───────────────────────────────────
print("\n[4/4] Menyimpan corpus per tahun...")
for year in sorted(df['year'].unique()):
    corpus   = df[df['year'] == year]['tokens'].tolist()
    filename = f'data/processed/corpus_{year}.txt'
    with open(filename, 'w', encoding='utf-8') as f:
        for tokens in corpus:
            f.write(' '.join(tokens) + '\n')
    print(f"    {year}: {len(corpus):>7,} tweets → {filename}")

# Simpan dataframe lengkap
out_path = 'data/processed/tweets_cleaned.csv'
df[['tweet', 'clean_text', 'tokens', 'year']].to_csv(out_path, index=False)

print(f"\n✓ Preprocessing selesai!")
print(f"  Output: data/processed/")
