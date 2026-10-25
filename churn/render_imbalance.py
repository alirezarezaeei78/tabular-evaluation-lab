"""Render actual measured outputs; original static PNG/SVG only."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent/'experiments/2026-10-25'
data=json.loads((root/'results.json').read_text());run=data['runs'][0]
fig,(ax,table_ax)=plt.subplots(2,1,figsize=(11,12),gridspec_kw={'height_ratios':[1.5,1]},layout='constrained')
labels={'prior':'Training-prior reference','unweighted':'Unweighted LR','balanced':'Balanced-weight LR','oversampled':'Normalized oversampling LR'}
colors=['#8895a5','#087f83','#bd650a','#6850a1']
for (name,result),color in zip(run['models'].items(),colors):
 c=result['pr_curve'];ax.step(c['recall'],c['precision'],where='post',label=labels[name],color=color,lw=2)
ax.set(xlim=(0,1),ylim=(0,1.02),xlabel='Recall',ylabel='Precision',title='Same631 evaluation rows ·101 churn labels · seed13')
ax.legend(loc='lower left',fontsize=11);ax.grid(alpha=.15)
table_ax.axis('off');rows=[]
for name,v in run['models'].items():
 b=v['policies']['exact_10_percent'];c=v['policies']['cutoff_0.5']
 rows.append([labels[name],f"{v['ap']:.4f}",f"{v['brier']:.4f}",str(b['tp']),str(b['fp']),str(c['tp']),str(c['fp'])])
t=table_ax.table(cellText=rows,colLabels=['Method','AP ↑','Brier ↓','TP /64','FP /64','TP @.5','FP @.5'],loc='center',colWidths=[.31,.11,.12,.11,.11,.12,.12]);t.auto_set_font_size(False);t.set_fontsize(11);t.scale(1,2.1)
for (i,j),cell in t.get_celld().items():
 cell.set_edgecolor('#dae2ec')
 if i==0:cell.set_facecolor('#e9f0f8');cell.set_text_props(weight='bold')
table_ax.set_title('Fixed64-record review budget versus a0.5 cutoff',fontsize=14,pad=0)
fig.suptitle('UCI Iranian Churn · training-only imbalance strategies',fontsize=17,weight='bold')
fig.supxlabel('Exploratory reuse of21October grouped splits; no independent validation.\nUCI Iranian Churn(2020), DOI10.24432/C5JW3Z · CC BY4.0. All three seeds in saved results.',fontsize=10)
fig.savefig(root/'imbalance-comparison.png',dpi=150);fig.savefig(root/'imbalance-comparison.svg')
p=root/'imbalance-comparison.svg';p.write_text('\n'.join(l.rstrip() for l in p.read_text().splitlines())+'\n',encoding='utf-8')
