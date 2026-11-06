"""Original static evidence table generated only from actual frozen experiment output."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent.parent
d=json.loads((ROOT/'results/results.json').read_text());r=next(r for r in d['runs'] if r['seed']==13)
get=lambda condition,policy:next(c for c in r['conditions'] if c['condition']==condition and c['policy']==policy)
fig,ax=plt.subplots(figsize=(7.2,9),dpi=150,facecolor='#F7F9FB');fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
navy,teal,gray='#152A44','#087F82','#536479'
plt.rcParams['svg.fonttype']='none'
ax.text(.06,.95,'A missing field is\nan input-policy decision',fontsize=23,weight='bold',color=navy,va='top')
ax.text(.06,.825,'Same 631 records | 101 churn labels',fontsize=16,color=gray)
ax.text(.06,.785,'Frozen booster | primary split, seed 13',fontsize=16,color=gray)
ax.text(.06,.72,'Usage input',fontsize=16,weight='bold',color=navy)
ax.text(.57,.72,'Native AP',fontsize=16,ha='center',weight='bold',color=navy)
ax.text(.84,.72,'Median AP',fontsize=16,ha='center',weight='bold',color=navy)
ax.plot([.06,.95],[.70,.70],color='#CBD6E0')
conditions=[('clean','Clean'),('usage_missing_25pct','25% missing*'),('usage_missing_all','All missing*'),('usage_times2','Values x2*')]
for i,(condition,label) in enumerate(conditions):
 y=.65-.067*i
 ax.text(.06,y,label,fontsize=18,color=navy)
 for policy,pos in [('native',.57),('median',.84)]:
  value=get(condition,policy)['all_row_policy_metrics']['ap']
  ax.text(pos,y,f'{value:.3f}',fontsize=20,ha='center',color=teal)
ax.text(.06,.36,'When all usage fields are missing:',fontsize=17,weight='bold',color=navy)
native=get('usage_missing_all','native')['all_row_policy_metrics']['brier'];median=get('usage_missing_all','median')['all_row_policy_metrics']['brier']
ax.text(.06,.32,f'Brier: native {native:.3f} | median {median:.3f}',fontsize=18,color=teal)
ax.text(.06,.267,'Guarded policy rejects all 631 records.',fontsize=16.5,color=gray)
ax.text(.06,.232,'Prior fallback is labelled separately.',fontsize=16.5,color=gray)
ax.text(.06,.19,'Native NaN routing is not robustness.',fontsize=16.5,color=navy,weight='bold')
ax.text(.06,.15,'*Simulated usage-field perturbations.',fontsize=15.5,color=gray)
ax.text(.06,.115,'Known reused splits; exploratory results.',fontsize=15.5,color=gray)
ax.text(.06,.067,'UCI Iranian Churn (2020) | CC BY 4.0',fontsize=14,color=gray)
ax.text(.06,.034,'DOI 10.24432/C5JW3Z | Original measured table',fontsize=13,color=gray)
for ext in ['png','svg']:fig.savefig(ROOT/f'missing-input-stress.{ext}',dpi=150,facecolor=fig.get_facecolor())
plt.close(fig)
