"""
03_visualize.py
================
Generate all visualizations for semantic shift analysis:
  1. Semantic similarity over time (line charts)
  2. Displacement heatmap
  3. Displacement comparison (3 focus keywords)
  4. t-SNE semantic space plot
  5. Nearest neighbors summary
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from gensim.models import Word2Vec
from sklearn.manifold import TSNE

os.makedirs('results', exist_ok=True)

# ── Load data ─────────────────────────────────────────────────
with open('models/displacement_results.pkl', 'rb') as f:
    results = pickle.load(f)

with open('models/aligned_vectors.pkl', 'rb') as f:
    aligned_vectors = pickle.load(f)

YEARS    = [2017, 2018, 2019, 2020, 2021, 2022, 2023]
KEYWORDS = ['healing', 'toxic', 'burnout', 'anxiety', 'depression', 'stress']
FOCUS_KW = ['healing', 'toxic', 'burnout']
COLORS   = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C']

sns.set_theme(style='whitegrid', font_scale=1.1)

# ── PLOT 1: Semantic Similarity per kata ──────────────────────
print("[1/4] Plotting semantic similarity over time...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle(
    'Semantic Similarity vs Baseline 2017\n'
    '(Semakin rendah = semakin besar pergeseran makna)',
    fontsize=14, fontweight='bold'
)

for idx, (kw, color) in enumerate(zip(KEYWORDS, COLORS)):
    ax        = axes[idx // 3][idx % 3]
    years_plt = YEARS[1:]
    sims      = [results[kw].get(y) for y in years_plt]
    valid     = [(y, s) for y, s in zip(years_plt, sims) if s is not None]

    if not valid:
        ax.set_title(f'"{kw}" (no data)')
        continue

    ys, ss = zip(*valid)
    ax.plot(ys, ss, 'o-', color=color, linewidth=2.5, markersize=8)
    ax.fill_between(ys, ss, alpha=0.12, color=color)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4, label='Baseline 2017')
    ax.axvspan(2020, 2021, alpha=0.08, color='red', label='COVID-19')
    ax.set_ylim(0.3, 1.1)
    ax.set_title(f'"{kw}"', fontsize=13, fontweight='bold')
    ax.set_xlabel('Tahun')
    ax.set_ylabel('Cosine Similarity')
    ax.set_xticks(years_plt)
    ax.legend(fontsize=8)

    # Annotate minimum
    min_s    = min(ss)
    min_year = ys[list(ss).index(min_s)]
    ax.annotate(
        f'min={min_s:.3f}',
        xy=(min_year, min_s),
        xytext=(8, 8), textcoords='offset points',
        fontsize=8, color='red',
        arrowprops=dict(arrowstyle='->', color='red', lw=1.2)
    )

plt.tight_layout()
plt.savefig('results/01_semantic_similarity.png', dpi=150, bbox_inches='tight')
print("    ✓ results/01_semantic_similarity.png")
plt.close()

# ── PLOT 2: Heatmap displacement ──────────────────────────────
print("[2/4] Plotting displacement heatmap...")

fig, ax = plt.subplots(figsize=(11, 5))
matrix  = np.array([
    [1 - (results[kw].get(y) or np.nan) for y in YEARS[1:]]
    for kw in KEYWORDS
])

sns.heatmap(
    matrix,
    annot=True, fmt='.3f',
    xticklabels=YEARS[1:],
    yticklabels=KEYWORDS,
    cmap='YlOrRd',
    vmin=0, vmax=0.65,
    linewidths=0.5,
    ax=ax
)
ax.set_title(
    'Semantic Displacement Score (vs Baseline 2017)\n'
    'Semakin merah = semakin besar pergeseran makna',
    fontsize=12, fontweight='bold'
)
ax.set_xlabel('Tahun')
ax.set_ylabel('Kata')

plt.tight_layout()
plt.savefig('results/02_heatmap_displacement.png', dpi=150, bbox_inches='tight')
print("    ✓ results/02_heatmap_displacement.png")
plt.close()

# ── PLOT 3: Comparison 3 focus keywords ───────────────────────
print("[3/4] Plotting displacement comparison...")

fig, ax = plt.subplots(figsize=(11, 6))

for kw, color in zip(FOCUS_KW, COLORS[:3]):
    years_plt = YEARS[1:]
    sims      = [results[kw].get(y) for y in years_plt]
    valid     = [(y, 1 - s) for y, s in zip(years_plt, sims) if s is not None]
    if not valid:
        continue
    ys, ds = zip(*valid)
    ax.plot(ys, ds, 'o-', color=color, linewidth=2.5, markersize=9, label=f'"{kw}"')
    ax.fill_between(ys, ds, alpha=0.08, color=color)

ax.axvspan(2019.5, 2021.5, alpha=0.07, color='gray')
ax.text(2020.5, ax.get_ylim()[0] + 0.01, 'COVID-19\nPandemic',
        ha='center', fontsize=8, color='gray', style='italic')

ax.set_title(
    'Semantic Displacement: "healing" vs "toxic" vs "burnout"\n'
    'Twitter Mental Health Dataset (2017–2023)',
    fontsize=13, fontweight='bold'
)
ax.set_xlabel('Tahun', fontsize=12)
ax.set_ylabel('Displacement Score (1 − cosine similarity)', fontsize=12)
ax.legend(fontsize=11)
ax.set_xticks(YEARS[1:])
ax.set_ylim(0, 0.75)

plt.tight_layout()
plt.savefig('results/03_displacement_comparison.png', dpi=150, bbox_inches='tight')
print("    ✓ results/03_displacement_comparison.png")
plt.close()

# ── PLOT 4: t-SNE ─────────────────────────────────────────────
print("[4/4] Building t-SNE plot (this may take a moment)...")

YEARS_SAMPLE  = [2017, 2019, 2021, 2023]
TSNE_COLORS   = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
TOP_N         = 15

fig, axes = plt.subplots(1, 3, figsize=(21, 7))
fig.suptitle(
    'Pergerakan Makna Kata di Ruang Semantik (t-SNE)\n'
    'Twitter Mental Health 2017–2023',
    fontsize=14, fontweight='bold'
)

for ax_idx, kw in enumerate(FOCUS_KW):
    ax          = axes[ax_idx]
    all_words   = []
    all_vectors = []
    all_years   = []

    for year in YEARS_SAMPLE:
        model = Word2Vec.load(f'models/word2vec_{year}.model')
        if kw not in model.wv:
            continue
        neighbors     = [w for w, _ in model.wv.most_similar(kw, topn=TOP_N)]
        words_to_plot = [kw] + neighbors
        for word in words_to_plot:
            if word in model.wv:
                all_words.append(word)
                all_vectors.append(model.wv[word])
                all_years.append(year)

    if not all_vectors:
        continue

    vectors_arr = np.array(all_vectors)
    perplexity  = min(30, len(vectors_arr) - 1)
    tsne        = TSNE(n_components=2, random_state=42,
                       perplexity=perplexity, max_iter=1000)
    coords      = tsne.fit_transform(vectors_arr)

    for year, color in zip(YEARS_SAMPLE, TSNE_COLORS):
        mask   = [i for i, y in enumerate(all_years) if y == year]
        xs     = coords[mask, 0]
        ys_c   = coords[mask, 1]
        words  = [all_words[i] for i in mask]

        ax.scatter(xs, ys_c, c=color, label=str(year), s=80, alpha=0.7, zorder=3)
        for i, word in enumerate(words):
            ax.annotate(
                word, (xs[i], ys_c[i]),
                fontsize=11 if word == kw else 7,
                fontweight='bold' if word == kw else 'normal',
                alpha=0.9
            )

    ax.set_title(f'"{kw}"', fontsize=13, fontweight='bold')
    ax.legend(title='Tahun', fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_xlabel('t-SNE dim 1')
    ax.set_ylabel('t-SNE dim 2')

plt.tight_layout()
plt.savefig('results/04_tsne_plot.png', dpi=150, bbox_inches='tight')
print("    ✓ results/04_tsne_plot.png")
plt.close()

# ── Nearest Neighbors Summary ─────────────────────────────────
print("\n" + "=" * 55)
print("  NEAREST NEIGHBORS PER TAHUN")
print("=" * 55)

for kw in FOCUS_KW:
    print(f"\n  '{kw}':")
    for year in [2017, 2019, 2021, 2023]:
        model = Word2Vec.load(f'models/word2vec_{year}.model')
        if kw in model.wv:
            neighbors = [w for w, _ in model.wv.most_similar(kw, topn=8)]
            print(f"    {year}: {', '.join(neighbors)}")

print(f"\n✓ Semua visualisasi tersimpan di results/")
