"""Original measured comparison table driven only by completed experiment JSON."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BASE=Path(__file__).parent/'retail-extremes'/'2026-11-12'
r=json.loads((BASE/'results.json').read_text(encoding='utf-8'))
assert r['status']=='completed' and r['model_fits_completed']==20
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none'})
fig=plt.figure(figsize=(10.8,14),dpi=100,facecolor='#f5f2e9')
ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,1080);ax.set_ylim(1400,0);ax.axis('off')
ink,muted,blue='#182e3b','#465660','#126777'
def txt(x,y,s,size=23,color=ink,bold=False,**kw):
    ax.text(x,y,s,fontsize=size*.72,color=color,weight='bold' if bold else 'normal',va='top',**kw)
def box(x,y,w,h,color):ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=12',facecolor=color,edgecolor='none'))
txt(54,42,'RETAIL SEGMENTATION • MEASURED SENSITIVITY',20,blue,True)
txt(54,87,'Does robust scaling remove',40,bold=True)
txt(54,141,'extreme-spend sensitivity?',40,bold=True)
txt(54,212,f"{r['matched_customers']:,} matched customers • k=2 • 20 fixed-budget fits",25)
txt(54,251,'RFM: recency, invoice frequency, gross purchase pounds',23,color=muted)
box(54,309,972,157,'#e1e8df')
txt(76,326,'THREE DISTINCT CHECKS',20,bold=True)
txt(76,365,'Frozen input: 17 comparison spenders, M × 10',23)
txt(76,400,'Refit: 30 development spenders, M × 10; score clean rows',23)
txt(76,435,'Sampling: the same three 80% development subsets per method',21)

xs=[74,388,556,720,921]
headers=['Preprocessing','Clean\nsizes','Frozen\nchanges','Refit\nARI','80% samples\nARI range']
for x,h in zip(xs,headers):txt(x,494,h,21,bold=True,ha='left' if x==74 else 'center')
labels={'raw_standard':'Raw + standard','log_standard':'Log + standard','raw_robust':'Raw + robust','log_robust':'Log + robust'}
for i,method in enumerate(r['preprocessing_methods']):
    y=580+i*110;m=r['methods'][method]
    box(54,y-18,972,94,'#ffffff' if i%2==0 else '#e9ece5')
    supports=sorted(m['clean_supports'],reverse=True)
    vals=[labels[method],f'{supports[0]:,} / {supports[1]:,}',f"{m['frozen_input_stress']['changed_fixed_labels']} / 17",f"{m['training_stress_refit']['ari']:.4f}",f"{min(m['sampling_aris']):.4f}–{max(m['sampling_aris']):.4f}"]
    for j,(x,v) in enumerate(zip(xs,vals)):txt(x,y+7,v,24 if j==0 else 23,color=blue if j==0 else ink,bold=j==0,ha='left' if j==0 else 'center')
box(54,1041,972,166,'#e1e8df')
txt(76,1058,'READ STABILITY WITH THE SUPPORTS',21,bold=True)
txt(76,1096,'ARI compares partitions; it is not customer-type accuracy.',23)
txt(76,1134,'Raw + robust refit: all 1,664 comparison rows in one group.',22)
txt(76,1170,'High sampling stability does not establish useful segments.',21)
txt(54,1240,'Same known historical snapshot; exploratory, no ground truth.',22,bold=True)
txt(54,1278,'Log = log1p. Robust = median/IQR. Neither removes outliers.',21,color=muted)
txt(54,1315,'Simulated monetary stress; no real spend, accuracy or causal claim.',20,color=muted)
txt(54,1351,'Data: Chen (2015), UCI Online Retail • CC BY 4.0',19,color=muted)
fig.savefig(BASE/'retail-extreme-sensitivity.png',dpi=100,facecolor=fig.get_facecolor())
fig.savefig(BASE/'retail-extreme-sensitivity.svg',facecolor=fig.get_facecolor())
p=BASE/'retail-extreme-sensitivity.svg';p.write_text('\n'.join(x.rstrip() for x in p.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
print('Rendered original 1080x1400 measured table from completed results.')
