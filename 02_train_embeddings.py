"""
02_train_embeddings.py
=======================
Train Word2Vec models per year and align them using
Orthogonal Procrustes for diachronic semantic analysis.

Method  : Diachronic Word Embeddings (Hamilton et al., 2016)
Align   : Orthogonal Procrustes (Smith et al., 2017)
Baseline: Year 2017
"""

import os
import pickle
import numpy as np
from gensim.models import Word2Vec
from scipy.linalg import orthogonal_procrustes

os.makedirs('models', exist_ok=True)

YEARS    = [2017, 2018, 2019, 2020, 2021, 2022, 2023]
KEYWORDS = ['healing', 'toxic', 'burnout', 'anxiety', 'depression', 'stress']

# ── FASE 1: Train Word2Vec per tahun ──────────────────────────
print("=" * 55)
print("  FASE 2: Training Diachronic Word Embeddings")
print("=" * 55)

models = {}
for year in YEARS:
    corpus_path = f'data/processed/corpus_{year}.txt'

    sentences = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            if tokens:
                sentences.append(tokens)

    print(f"\n  Training Word2Vec {year}: {len(sentences):,} kalimat...", end=' ', flush=True)

    model = Word2Vec(
        sentences   = sentences,
        vector_size = 100,
        window      = 5,
        min_count   = 5,
        workers     = 4,
        epochs      = 10,
        seed        = 42
    )

    model.save(f'models/word2vec_{year}.model')
    models[year] = model
    vocab_size = len(model.wv.index_to_key)
    print(f"✓  (vocab: {vocab_size:,} kata)")

# ── FASE 2: Alignment (Orthogonal Procrustes) ─────────────────
print("\n" + "=" * 55)
print("  FASE 3: Aligning ke Shared Vector Space")
print("=" * 55)
print(f"\n  Baseline: tahun {YEARS[0]}")

base_model      = models[YEARS[0]]
aligned_vectors = {YEARS[0]: base_model.wv}

for year in YEARS[1:]:
    model        = models[year]
    common_words = list(
        set(base_model.wv.index_to_key) & set(model.wv.index_to_key)
    )[:3000]

    A = np.array([model.wv[w]      for w in common_words])
    B = np.array([base_model.wv[w] for w in common_words])

    R, _ = orthogonal_procrustes(A, B)

    aligned = {word: model.wv[word] @ R for word in model.wv.index_to_key}
    aligned_vectors[year] = aligned

    print(f"  {year} → aligned  (common vocab: {len(common_words):,} kata) ✓")

with open('models/aligned_vectors.pkl', 'wb') as f:
    pickle.dump(aligned_vectors, f)

# ── FASE 3: Hitung Semantic Displacement ─────────────────────
print("\n" + "=" * 55)
print("  FASE 4: Semantic Displacement Score")
print("=" * 55)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def get_vector(vecs, word: str):
    try:
        return vecs[word]
    except Exception:
        return None

results    = {}
base_vecs  = aligned_vectors[YEARS[0]]

print(f"\n  {'Kata':<12}", end='')
for y in YEARS[1:]:
    print(f"  {y}", end='')
print(f"\n  {'(sim vs 2017)':<12}", end='')
for _ in YEARS[1:]:
    print(f"   ───", end='')
print()

for kw in KEYWORDS:
    results[kw] = {}
    base_vec    = get_vector(base_vecs, kw)

    if base_vec is None:
        print(f"  {kw:<12}  (tidak ada di vocab 2017)")
        continue

    print(f"  {kw:<12}", end='')
    for year in YEARS[1:]:
        vec = get_vector(aligned_vectors[year], kw)
        if vec is not None:
            sim = cosine_sim(base_vec, vec)
            results[kw][year] = sim
            print(f"  {sim:.3f}", end='')
        else:
            results[kw][year] = None
            print(f"    N/A", end='')
    print()

with open('models/displacement_results.pkl', 'wb') as f:
    pickle.dump(results, f)

# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  RINGKASAN DISPLACEMENT SCORE (1 - cosine similarity)")
print("=" * 55)

focus = ['healing', 'toxic', 'burnout']
for kw in focus:
    valid = {y: s for y, s in results[kw].items() if s is not None}
    if not valid:
        continue
    max_year = min(valid, key=valid.get)
    max_disp = 1 - valid[max_year]
    last_disp = 1 - list(valid.values())[-1]
    level = "TINGGI" if last_disp > 0.5 else "SEDANG" if last_disp > 0.3 else "RENDAH"
    print(f"\n  '{kw}':")
    print(f"    Pergeseran terbesar : {max_year} (score={max_disp:.3f})")
    print(f"    Displacement akhir  : {last_disp:.3f} → {level}")

print(f"\n✓ Model & hasil tersimpan di models/")
