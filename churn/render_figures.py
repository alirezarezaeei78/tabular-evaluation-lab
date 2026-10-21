"""Render three original static technical figures from saved aggregate measurements."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE=Path(__file__).resolve().parent
OUT=HERE/'experiments/2026-10-21'
DATA=json.loads((OUT/'results.json').read_text())
PRIMARY=next(r for r in DATA['runs'] if r['seed']==13)
INK='#17324d';BLUE='#286e9c';ORANGE='#ae4e18';GREY='#536775';PALE='#edf4f8'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':15,'text.color':INK,
                     'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,'svg.fonttype':'none'})


def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=160,facecolor='white')
    svg=OUT/(name+'.svg')
    fig.savefig(svg,facecolor='white')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    plt.close(fig)


def workflow():
    fig,ax=plt.subplots(figsize=(10,5.625));fig.subplots_adjust(0,0,1,1);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    ax.text(.055,.93,'FIT / CALIBRATE / EVALUATE',fontsize=23,weight='bold')
    ax.text(.055,.875,'Exact predictor groups stay together · primary seed 13',fontsize=14,color=GREY)
    def box(x,y,w,h,heading,body,face=PALE):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=.012',linewidth=1.4,edgecolor='#a9bfce',facecolor=face))
        ax.text(x+.018,y+h-.047,heading,fontsize=17,weight='bold',va='top')
        ax.text(x+.018,y+h-.102,body,fontsize=14,va='top',linespacing=1.5)
    fit,cal,test=(PRIMARY['partitions'][x] for x in ['fit','calibration','evaluation'])
    box(.055,.57,.27,.24,f"FIT · {fit['rows']:,} rows",f"Train models + scaler\n{fit['positives']} observed churn labels")
    box(.365,.57,.27,.24,f"CAL · {cal['rows']:,} rows",f"Fit sigmoid; fix cutoff\n{cal['positives']} observed churn labels")
    box(.675,.57,.27,.24,f"EVAL · {test['rows']:,} rows",f"Measure on held-out rows\n{test['positives']} observed churn labels")
    for x in [.33,.64]:ax.add_patch(FancyArrowPatch((x,.69),(x+.028,.69),arrowstyle='-|>',mutation_scale=18,color=BLUE))
    ax.text(.055,.49,'Same boosting scores → different review rules',fontsize=19,weight='bold')
    raw,calibrated=(PRIMARY['models'][x]['policies'] for x in ['boosting','boosting_sigmoid'])
    exact=raw['exact_10_percent'];assert exact==calibrated['exact_10_percent']
    box(.055,.17,.42,.25,'Fixed 0.5 cutoff',f"Uncalibrated: {raw['cutoff_0.5']['selected']} flagged\nAfter sigmoid: {calibrated['cutoff_0.5']['selected']} flagged",face='#fff1e6')
    box(.525,.17,.42,.25,'Exact 10% budget',f"{exact['selected']} flagged with either score\n{exact['tp']} positives · {exact['fp']} false positive")
    ax.text(.055,.092,f"Budget = ceil(0.10 × {test['rows']}). Cutoffs do not guarantee capacity.",fontsize=13,color=GREY)
    ax.text(.055,.045,'Source: UCI Iranian Churn (CC BY 4.0) · local retrospective experiment, 21 Oct 2026',fontsize=10.5,color=GREY)
    save(fig,'workflow')


def calibration():
    fig=plt.figure(figsize=(7.5,9.5));gs=fig.add_gridspec(3,1,height_ratios=[1.05,1,.45],hspace=.65)
    fig.subplots_adjust(left=.14,right=.94,top=.91,bottom=.07)
    fig.text(.1,.965,'Probability error after sigmoid fitting',fontsize=19,weight='bold')
    ax=fig.add_subplot(gs[0]);width=.31
    for offset,name,label,color in [(-width/2,'boosting','Boosting',BLUE),(width/2,'boosting_sigmoid','Boosting + sigmoid',ORANGE)]:
        xs=[i+offset for i in range(3)];vals=[r['models'][name]['brier'] for r in DATA['runs']]
        ax.bar(xs,vals,width=width,label=label,color=color,hatch='//' if offset>0 else None,edgecolor=color)
        for x,y in zip(xs,vals):ax.text(x,y+.001,f'{y:.5f}',ha='center',va='bottom',fontsize=10.5)
    ax.set(xticks=range(3),xticklabels=['Seed 13','Seed 29','Seed 47'],ylim=(0,.047),ylabel='Brier loss ↓')
    ax.legend(frameon=False,fontsize=11,loc='upper right');ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    ax=fig.add_subplot(gs[1]);ax.plot([0,1],[0,1],ls='--',color=GREY,lw=1,label='Identity')
    for name,label,color,marker in [('boosting','Boosting',BLUE,'o'),('boosting_sigmoid','+ sigmoid',ORANGE,'s')]:
        bins=PRIMARY['models'][name]['reliability_bins']
        ax.plot([x['mean_score'] for x in bins],[x['event_fraction'] for x in bins],marker=marker,color=color,label=label,lw=1.5)
    ax.set(xlim=(0,1),ylim=(0,1.05),xlabel='Mean predicted probability',ylabel='Observed fraction')
    ax.set_title('Seed 13 · ten fixed-width reliability bins',fontsize=13,pad=12)
    ax.legend(frameon=False,fontsize=10,loc='lower right');ax.spines[['top','right']].set_visible(False)
    ax=fig.add_subplot(gs[2]);ax.axis('off')
    ax.text(0,.95,'AP was unchanged in each split:',fontsize=13,weight='bold')
    assert all(r['models']['boosting']['ap']==r['models']['boosting_sigmoid']['ap'] for r in DATA['runs'])
    ax.text(0,.55,'    '.join(f"{r['seed']}: {r['models']['boosting']['ap']:.4f}" for r in DATA['runs']),fontsize=12)
    raw_n=[x['n'] for x in PRIMARY['models']['boosting']['reliability_bins']]
    cal_n=[x['n'] for x in PRIMARY['models']['boosting_sigmoid']['reliability_bins']]
    ax.text(0,.12,f"Bin supports: {min(raw_n)}–{max(raw_n)} rows (raw), {min(cal_n)}–{max(cal_n)} (+ sigmoid).\nSmall bins are noisy; Brier is not pure calibration error.",fontsize=10.5,color=GREY,va='top')
    fig.text(.1,.015,'UCI Iranian Churn · grouped retrospective splits · CC BY 4.0\nAll local seed outputs retained; no confidence intervals or latency claim.',fontsize=9,color=GREY)
    save(fig,'calibration')


def policies():
    fig,ax=plt.subplots(figsize=(7.5,9.5));fig.subplots_adjust(0,0,1,1);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    test=PRIMARY['partitions']['evaluation']
    ax.text(.06,.955,f"PRIMARY SPLIT · {test['rows']} ROWS / {test['positives']} POSITIVES",fontsize=16,weight='bold')
    ax.text(.06,.907,'Same rows, four probability models',fontsize=16,weight='bold')
    rows=[]
    labels={'dummy':'Prevalence','logistic':'Logistic','boosting':'Boosting','boosting_sigmoid':'Boosting + sigmoid'}
    for k,v in PRIMARY['models'].items():rows.append([labels[k],f"{v['ap']:.4f}",f"{v['brier']:.5f}"])
    tab=ax.table(cellText=rows,colLabels=['Model','AP ↑','Brier ↓'],bbox=[.06,.645,.88,.23],colWidths=[.46,.20,.22],cellLoc='left',colLoc='left')
    style_table(tab,14)
    ax.text(.06,.59,'Boosting review policies',fontsize=16,weight='bold')
    rows=[]
    for key,label in [('cutoff_0.5','0.5 cutoff'),('development_fixed_cutoff','Dev cutoff'),('exact_10_percent','Exact 10%')]:
        for name,short in [('boosting','Raw'),('boosting_sigmoid','Sigmoid')]:
            x=PRIMARY['models'][name]['policies'][key]
            rows.append([label,short,x['selected'],x['tp'],x['fp'],x['fn']])
    t=ax.table(cellText=rows,colLabels=['Policy','Score','Flags','TP','FP','Missed'],bbox=[.06,.22,.88,.33],
               colWidths=[.24,.18,.12,.1,.1,.14],cellLoc='left',colLoc='left');style_table(t,12)
    exact=PRIMARY['models']['boosting']['policies']['exact_10_percent']
    dev=PRIMARY['models']['boosting']['policies']['development_fixed_cutoff']
    ax.text(.06,.16,f"Exact budget: {exact['tp']}/{exact['selected']} precision; {exact['tp']}/{test['positives']} recall.\nDevelopment cutoff flags {dev['selected']} rows ({dev['tp']} TP, {dev['fp']} FP).",fontsize=13,linespacing=1.6)
    ax.text(.06,.065,'10% is a hypothetical review capacity, not retention impact.\nDev cutoff fixed from calibration; evaluation share may drift.\nUCI Iranian Churn (CC BY 4.0) · seed 13 · local outputs',fontsize=10.5,color=GREY,linespacing=1.5)
    save(fig,'policies')


def style_table(table,size):
    table.auto_set_font_size(False);table.set_fontsize(size)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor('#d4e0e7');cell.set_linewidth(.7)
        if r==0:cell.set_facecolor(PALE);cell.set_text_props(weight='bold',color=INK)
        else:cell.set_facecolor('white' if r%2 else '#f8fafb')


if __name__=='__main__':
    workflow();calibration();policies()
    (OUT/'figures-data.json').write_text(json.dumps({'source_results':'results.json','primary_seed':13,
       'data_sha256':DATA['data']['sha256'],'runs':[{'seed':r['seed'],'partitions':r['partitions'],'models':r['models']} for r in DATA['runs']]},indent=2)+'\n')
    print('Rendered workflow, calibration and policies as original PNG + editable SVG; no video.')
