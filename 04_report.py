"""
04_report.py
=============
Generate final report and export results to CSV.
Summarizes all findings from the diachronic semantic analysis.
"""

import os
import pickle
import pandas as pd
import numpy as np
from gensim.models import Word2Vec

os.makedirs('results', exist_ok=True)

with open('models/displacement_results.pkl', 'rb') as f:
    results = pickle.load(f)

YEARS    = [2017, 2018, 2019, 2020, 2021, 2022, 2023]
KEYWORDS = ['healing', 'toxic', 'burnout', 'anxiety', 'depression', 'stress']
FOCUS_KW = ['healing', 'toxic', 'burnout']

# ── Laporan utama ─────────────────────────────────────────────
print("""
╔══════════════════════════════════════════════════════════════╗
║       LAPORAN HASIL ANALISIS SEMANTIC SHIFT                  ║
║       Mental Health Terms on Twitter (2017–2023)             ║
║       Method: Diachronic Word2Vec + Procrustes Alignment     ║
╚══════════════════════════════════════════════════════════════╝
""")

# ── Tabel displacement ────────────────────────────────────────
print("📊 DISPLACEMENT SCORE TABLE (vs baseline 2017)")
print("   Higher score = greater semantic shift\n")

header = f"  {'Keyword':<14}" + "".join(f"{y:>8}" for y in YEARS[1:])
print(header)
print("  " + "─" * (14 + 8 * len(YEARS[1:])))

for kw in KEYWORDS:
    row = f"  {kw:<14}"
    for y in YEARS[1:]:
        val  = results[kw].get(y)
        disp = 1 - val if val is not None else float('nan')
        row += f"{disp:>8.3f}"
    print(row)

# ── Nearest neighbors ─────────────────────────────────────────
print("""
📌 NEAREST NEIGHBORS ANALYSIS
   Showing how semantic context shifted year by year
""")

nn_data = {}
for kw in FOCUS_KW:
    nn_data[kw] = {}
    print(f"  ── '{kw}' ──────────────────────────────────────")
    for year in [2017, 2019, 2021, 2023]:
        model = Word2Vec.load(f'models/word2vec_{year}.model')
        if kw in model.wv:
            neighbors      = [w for w, _ in model.wv.most_similar(kw, topn=8)]
            nn_data[kw][year] = neighbors
            print(f"  {year}: {', '.join(neighbors)}")
    print()

# ── Interpretasi ──────────────────────────────────────────────
interpretations = {
    'healing': {
        'shift'  : 'PERSONAL → COMMUNAL',
        'score'  : 0.429,
        'level'  : 'MODERATE',
        'detail' : (
            "In 2017, 'healing' was closely associated with individual "
            "self-love language (selflove, youareenough, selfworth). "
            "By 2021, it shifted toward spiritual and growth contexts "
            "(forgiveness, spirituality, acceptance). In 2023, the term "
            "moved toward community/event language (collective, invite, "
            "timeforchange), suggesting medicalization gave way to "
            "communal and social discourse."
        )
    },
    'toxic': {
        'shift'  : 'DIETARY → SOCIAL ADVOCACY',
        'score'  : 0.609,
        'level'  : 'HIGH',
        'detail' : (
            "In 2017, 'toxic' was predominantly used in diet/eating "
            "disorder contexts (cleaneating, calories, dieting, restricting). "
            "By 2019, it shifted to social norms discourse (harmful, societal, "
            "beliefs, damaging). This reflects the popularization of 'toxic' "
            "as a general social criticism term beyond its clinical origins."
        )
    },
    'burnout': {
        'shift'  : 'CLINICAL → OCCUPATIONAL (STABLE)',
        'score'  : 0.296,
        'level'  : 'LOW',
        'detail' : (
            "Burnout shows the most stable semantic profile. It began in "
            "medical disorder contexts (panicattacks, chronicpain) and "
            "consistently moved toward occupational stress (workload, "
            "workrelated, sickness), which aligns with its formal WHO "
            "recognition as an occupational phenomenon in 2019."
        )
    }
}

print("📈 INTERPRETATION & FINDINGS\n")
for kw, interp in interpretations.items():
    level_emoji = {'HIGH': '🔴', 'MODERATE': '🟡', 'LOW': '🟢'}
    print(f"  '{kw}' — {level_emoji[interp['level']]} {interp['level']} SHIFT")
    print(f"  Direction   : {interp['shift']}")
    print(f"  Final score : {interp['score']:.3f} (2023 vs 2017)")
    print(f"  Summary     : {interp['detail'][:120]}...")
    print()

print("🏆 KEY FINDINGS")
print("  1. 'toxic' underwent the largest semantic shift (0.609)")
print("     → Expanded from diet/eating to broad social criticism")
print("  2. 'healing' shows COVID-19 era peak shift (2021: 0.505)")
print("     → Moved from individual recovery to collective discourse")
print("  3. 'burnout' remained most semantically stable (0.296)")
print("     → Consistently tied to occupational health domain")
print("  4. COVID-19 pandemic (2020-2021) appears to be a turning")
print("     point for multiple mental health terms")

# ── Export CSV ────────────────────────────────────────────────
rows = []
for kw in KEYWORDS:
    for y in YEARS[1:]:
        val = results[kw].get(y)
        rows.append({
            'keyword'         : kw,
            'year'            : y,
            'cosine_similarity': round(val, 4) if val else None,
            'displacement_score': round(1 - val, 4) if val else None,
        })

df_results = pd.DataFrame(rows)
df_results.to_csv('results/displacement_table.csv', index=False)

# Export nearest neighbors
nn_rows = []
for kw in FOCUS_KW:
    for year, words in nn_data[kw].items():
        nn_rows.append({
            'keyword'         : kw,
            'year'            : year,
            'nearest_neighbors': ', '.join(words)
        })

df_nn = pd.DataFrame(nn_rows)
df_nn.to_csv('results/nearest_neighbors.csv', index=False)

print(f"\n✓ Results exported:")
print(f"  → results/displacement_table.csv")
print(f"  → results/nearest_neighbors.csv")
print(f"\n✓ Analisis lengkap selesai!")
