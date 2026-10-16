"""Render static technical figures from the saved split-audit outputs."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'experiments/2026-10-16'
NAVY, TEAL, AMBER, LIGHT = '#142b49', '#087f8c', '#b26b00', '#edf3f8'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 16, 'svg.fonttype': 'none',
                     'svg.hashsalt': 'split-audit-2026-10-16', 'text.color': NAVY})


def save(fig, name):
    fig.savefig(OUT/(name+'.png'), dpi=150, facecolor='white')
    fig.savefig(OUT/(name+'.svg'), facecolor='white', metadata={'Date': None})
    p = OUT/(name+'.svg')
    p.write_bytes(b'\n'.join(line.rstrip() for line in p.read_bytes().splitlines())+b'\n')
    plt.close(fig)


def main():
    results = json.loads((OUT/'results.json').read_text())
    rows = [r for r in results['summaries'] if r['seed'] == 13 or r['method'] == 'independent_common_values']
    labels = {'random_rows': 'Random rows', 'feature_partition': 'Feature-key\npartition',
              'group_partition': 'Entity-group\npartition', 'independent_common_values': 'Distinct entities,\ncommon features'}
    cells = [[labels[r['method']], f"{r['train_rows']} / {r['test_rows']}",
              f"{r['overlapping_test_rows']} / {r['test_rows']}",
              f"{r['test_rows_in_shared_groups']} / {r['test_rows']}"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 6.8))
    ax.axis('off')
    table = ax.table(cellText=cells,
                     colLabels=['Partition / fixture', 'Train / test\nrows', 'Test rows:\nfeatures seen', 'Test rows:\ngroup seen'],
                     colWidths=[.31, .20, .245, .245], cellLoc='center', bbox=[.025, .29, .95, .66])
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#cad6e1')
        if r == 0:
            cell.set_facecolor(NAVY)
            cell.get_text().set_color('white')
        else:
            cell.set_facecolor('white' if r % 2 else LIGHT)
            if c >= 2 and int(cells[r-1][c].split('/')[0]) > 0:
                cell.set_facecolor('#fff0d6')
    ax.text(.04, .22, 'First 3 rows: 600 synthetic records / 200 entities; split seed 13.', fontsize=11.8)
    ax.text(.04, .175, 'Each entity: 2 repeated-feature records + 1 distinct-feature record.', fontsize=11.2)
    ax.text(.04, .13, 'Last row: separate 4-record fixture; shared features, disjoint entities.', fontsize=11)
    ax.text(.04, .08, 'Counts from executed reports; selected features exclude target and IDs.', fontsize=11)
    ax.text(.04, .035, 'Overlap is a review signal; no overlap is not a safety certificate.', fontsize=11.3, color=TEAL)
    fig.subplots_adjust(left=.02, right=.98, top=.98, bottom=.02)
    save(fig, 'split-overlap-counts')

    fig, ax = plt.subplots(figsize=(8, 6.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    boxes = [(.05,.76,.39,.15,'Train / test CSV snapshots\nExplicit feature + group roles'),
             (.56,.76,.39,.15,'Validate headers / rows\nReject malformed numeric cells'),
             (.56,.48,.39,.15,'Canonical feature tuple\nNFC text + exact finite decimals'),
             (.05,.48,.39,.15,'Compare complete keys\nSHA-256 labels + row positions'),
             (.05,.18,.39,.15,'Selected-feature matches\nShared group IDs (separate)'),
             (.56,.18,.39,.15,'JSON report + CSV flags\nExit 0 / 1 / 2 for CI')]
    for x,y,w,h,label in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012',facecolor=LIGHT,edgecolor=TEAL,linewidth=1.5))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=12.7)
    for a,b in [((.45,.835),(.54,.835)),((.755,.745),(.755,.645)),((.545,.555),(.455,.555)),
                ((.245,.465),(.245,.345)),((.455,.255),(.545,.255))]:
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=15,color=TEAL,lw=1.6))
    ax.text(.5,.075,'Schematic of the executed local CLI/API; no automatic row deletion.',ha='center',fontsize=11)
    ax.text(.5,.025,'Exact equality only; fingerprints do not anonymize private values.',ha='center',fontsize=11)
    fig.subplots_adjust(left=.02,right=.98,top=.98,bottom=.02)
    save(fig, 'audit-workflow')

    captions = {
        'split-overlap-counts': {'caption': 'Actual selected-feature/group overlap counts on declared synthetic CSV fixtures. Seed 13 shown; complete seeds 13/29/47 and all reports retained. The partition methods have different test rows and sizes, not matched predictive accuracy.',
          'alt': 'Measured table: random row split has 420 train and 180 test rows, 85 of 180 test rows with feature tuples seen in train and 162 of 180 with a shared entity. Feature-key partition has 430 train and 170 test rows, zero feature matches but 104 of 170 shared-entity rows. Entity partition has 420/180 rows and zero of both. A separate distinct-entity common-feature fixture has 2/2 rows, one of two feature matches and zero shared entities. Synthetic functionality test, no predictive-performance claim.'},
        'audit-workflow': {'caption': 'Implemented CSV input-validation, exact canonicalization, independent feature/group intersection and JSON/CSV export flow. Schematic, not a timing measurement.',
          'alt': 'Local CSV snapshots and explicit column roles lead to header and row validation, canonical feature tuples with NFC text and exact finite decimals, complete-key equality, separate feature and group matches, then JSON evidence, CSV row flags and CI exit codes. No automatic deletion.'}}
    (OUT/'figure-captions.json').write_bytes((json.dumps(captions,indent=2)+'\n').encode())


if __name__ == '__main__':
    main()
