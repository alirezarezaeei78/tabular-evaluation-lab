"""Original measured static figure; values are loaded from saved aggregates."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent / 'experiments/2026-10-29'
report = json.loads((ROOT / 'results.json').read_text(encoding='utf-8'))
rows = report['summary_rows']
navy, teal, amber = '#183247', '#087f83', '#bd650a'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 13, 'text.color': navy,
                     'axes.labelcolor': navy, 'xtick.color': navy, 'ytick.color': navy,
                     'svg.fonttype': 'none'})
fig = plt.figure(figsize=(12, 12.7), facecolor='white')
fig.text(.06, .956, 'Refitting after feature removal', fontsize=25, weight='bold')
fig.text(.06, .925, 'UCI Iranian Churn · fixed model settings · original partitions unchanged', fontsize=13)
ax = fig.add_axes([.24, .565, .69, .285])
names = ['without_quality', 'without_usage', 'without_subscription', 'without_age']
labels = ['No quality group', 'No usage group', 'No subscription group', 'No age group']
for seed, color, marker, offset in [(13, teal, 'o', -.16), (29, amber, 's', 0), (47, '#6850a1', '^', .16)]:
    values = [next(r['delta_ap_vs_full'] for r in rows if r['seed'] == seed and r['variant'] == n) for n in names]
    ax.scatter(values, np.arange(4) + offset, s=72, color=color, marker=marker, label=f'Seed {seed}', zorder=3)
ax.axvline(0, color=navy, lw=1)
ax.set(yticks=range(4), yticklabels=labels, xlim=(-.255, .009), ylim=(3.5, -.55),
       xlabel='Change in AP versus the same seed’s full model')
ax.set_title('All four removals reduced AP in every declared split', loc='left', fontsize=14, pad=19)
ax.grid(axis='x', alpha=.14); ax.legend(loc='upper left', fontsize=11, frameon=False)
for spine in ['top', 'right', 'left']: ax.spines[spine].set_visible(False)
ax.tick_params(axis='y', length=0)

tax = fig.add_axes([.055, .207, .895, .275]); tax.axis('off')
tax.set_title('Primary seed 13 · same 631 evaluation rows · 101 churn labels', fontsize=14, loc='left', pad=13)
order = ['full', 'without_quality', 'without_usage', 'without_subscription', 'without_age', 'single_feature', 'prior']
display = ['Full model (11)', 'No quality (9)', 'No usage (7)', 'No subscription (8)', 'No age (9)', 'Complains only (1)', 'Training prior (0)']
cells = []
for name, label in zip(order, display):
    row = next(r for r in rows if r['seed'] == 13 and r['variant'] == name)
    matches = row['feature_equivalent_fit_rows']
    cells.append([label, f"{row['ap']:.4f}", f"{row['brier']:.5f}", f"{row['tp']}/64", '—' if matches is None else str(matches)])
table = tax.table(cellText=cells, colLabels=['Feature set (count)', 'AP ↑', 'Brier ↓', 'True positives', 'Fit matches*'],
                  cellLoc='center', colWidths=[.36, .14, .17, .18, .15], bbox=[0, 0, 1, 1])
table.auto_set_font_size(False); table.set_fontsize(12)
for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor('white'); cell.set_linewidth(1.6)
    if r == 0: cell.set_facecolor(navy); cell.set_text_props(color='white', weight='bold')
    else: cell.set_facecolor('#edf4f7' if r % 2 else '#f6f8fa')
    if c == 0 and r: cell.set_text_props(ha='left')

fig.text(.06, .158, '*Evaluation feature tuples also found in fitting rows after removing columns.', fontsize=11)
fig.text(.06, .137, 'Original 11-feature duplicate groups stay separate; reduced features can create new matches.', fontsize=11)
fig.text(.06, .103, 'Exploratory reuse of 21/25 October splits · overlapping seed checks, no confidence intervals', fontsize=11)
fig.text(.06, .082, 'Refit comparison, not causal importance or evidence of retention benefit. Feature costs unmeasured.', fontsize=11)
fig.text(.06, .049, 'Data: Iranian Churn (2020), UCI · DOI 10.24432/C5JW3Z · CC BY 4.0', fontsize=10)
fig.savefig(ROOT / 'feature-ablation.png', dpi=150)
fig.savefig(ROOT / 'feature-ablation.svg')
p = ROOT / 'feature-ablation.svg'
p.write_text('\n'.join(line.rstrip() for line in p.read_text(encoding='utf-8').splitlines()) + '\n', encoding='utf-8')
print('Saved original measured PNG/SVG from all 21 aggregate rows; no simulated outcomes.')
