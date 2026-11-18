"""Original static workflow/table from the executed snapshot report."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).parent/'snapshot-comparison/2026-11-18'
def main():
    r=json.loads((ROOT/'results.json').read_text())['provider_report'];a=r['accounting']
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.hashsalt':'snapshot-20261118'})
    fig=plt.figure(figsize=(10.8,13.8),dpi=100,facecolor='#f4f3ee')
    ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,1080);ax.set_ylim(1380,0);ax.axis('off')
    navy='#152b3b';teal='#067568';muted='#465561'
    def text(x,y,s,size=22,color=navy,weight='normal',**kw):ax.text(x,y,s,fontsize=size,color=color,weight=weight,va='top',**kw)
    def box(x,y,w,h,color):ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=14',facecolor=color,edgecolor='none'))
    text(58,48,'FROZEN SEGMENT COMPARISON',16,teal,'bold')
    text(58,93,'Keep the model fixed.\nAccount for the cohort.',37,navy,'bold',linespacing=1.15)
    text(58,222,'Local CLI • RFM snapshots • matching JSON / CSV',20,muted)
    for x,title,body in [(58,'1  Load','Two RFM CSVs'),(390,'2  Freeze','Same model + rule'),(722,'3  Compare','Match customer IDs')]:
        box(x,285,300,110,'#dfe8e5');text(x+20,302,title,21,teal,'bold');text(x+20,345,body,18)
    text(58,432,'ACTUAL COHORT ACCOUNTING',17,teal,'bold')
    box(58,474,964,236,'#ffffff')
    for y,label,value in [(495,'180-day snapshot',a['left']),(544,'30-day snapshot',a['right']),(593,'Matched on both sides',a['matched']),(642,'180-day only / 30-day only',f"{a['left_only']:,} / {a['right_only']:,}")]:
        text(82,y,label,21);text(990,y,f'{value:,}' if isinstance(value,int) else value,24,navy,'bold',ha='right')
    text(58,748,'MATCHED CUSTOMERS ONLY  •  SAME FROZEN LABELS',16,teal,'bold')
    box(58,794,964,217,'#ffffff')
    for x,label in [(82,'Method'),(650,'Changed labels'),(966,'ARI')]:text(x,813,label,17,muted,'bold',ha='right' if x>600 else 'left')
    for y,label,key in [(861,'Frozen K-means','kmeans'),(935,'Median rule','median_rule')]:
        m=r['methods'][key];text(82,y,label,20,navy,'bold');text(650,y,f"{m['changed_fixed_labels']:,} / {m['matched']:,}",21,navy,'bold',ha='right');text(966,y,f"{m['ari']:.4f}",22,navy,'bold',ha='right')
        text(650,y+30,f"{m['changed_fixed_label_share']:.2%}",17,muted,ha='right')
    box(58,1050,964,166,'#152b3b')
    text(82,1072,'Different windows, one cutoff: 1 December 2011.',21,'#ffffff','bold')
    text(82,1116,'This is window sensitivity, not temporal migration.\nARI measures partition agreement, not segment accuracy.\nFewer label changes do not establish a better method.',19,'#ffffff',linespacing=1.45)
    text(58,1251,'UCI Online Retail • Chen (2015) • CC BY 4.0',17,muted)
    text(58,1285,'Known historical cohort; exploratory reuse. Positive purchases only.\nDevelopment and comparison windows overlap. No causal claim.',16,muted,linespacing=1.45)
    fig.savefig(ROOT/'snapshot-comparison.png',dpi=100,facecolor=fig.get_facecolor())
    with (ROOT/'snapshot-comparison.svg').open('w',encoding='utf-8',newline='\n') as handle:fig.savefig(handle,format='svg',metadata={'Date':None})
    svg=ROOT/'snapshot-comparison.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')
    plt.close(fig)
if __name__=='__main__':main()
