"""Original educational infographic; motion replays saved fitted stages."""
from pathlib import Path
import json
import subprocess
import sys

import imageio_ffmpeg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'outputs'
NAVY, BLUE, AMBER, INK = '#102b56', '#245dd8', '#b86600', '#18324e'


def draw():
    data = np.load(OUT/'render-data.npz')
    fig = plt.figure(figsize=(10.8,13.5), dpi=100, facecolor='white')
    ax = fig.add_axes([0,0,1,1]); ax.set(xlim=(0,1), ylim=(0,1)); ax.axis('off')

    def text(x,y,s,size=15,weight='normal',color=INK,ha='left',va='top'):
        return ax.text(x,y,s,fontsize=size,fontweight=weight,color=color,ha=ha,va=va,linespacing=1.4)

    def box(x,y,w,h,label='',color='#eff5ff',edge='#b5c9e7',size=14):
        p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.006,rounding_size=0.012',facecolor=color,edgecolor=edge,lw=1.2)
        ax.add_patch(p)
        if label:text(x+w/2,y+h/2,label,size,ha='center',va='center')
        return p

    def arrow(start,end,color=BLUE):
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=17,color=color,lw=1.8))

    text(.5,.971,'LOGISTIC REGRESSION',31,'bold',NAVY,'center')
    text(.5,.929,'vs GRADIENT BOOSTING',31,'bold',NAVY,'center')
    text(.5,.881,'Two ways to turn features into a probability',17,ha='center')
    box(.035,.22,.452,.622,color='#f3f8ff')
    box(.513,.22,.452,.622,color='#fffbef',edge='#e8ca8e')
    text(.261,.824,'ONE LINEAR SCORE',18,'bold',BLUE,'center')
    text(.739,.824,'MANY SMALL TREES',18,'bold',AMBER,'center')
    text(.261,.789,r'$p=\sigma(w^{T}x+b)$',22,ha='center')
    text(.739,.803,r'$p=\sigma(F_0+\eta\sum_m T_m(x))$',18,ha='center')

    for x,col in [(.261,BLUE),(.739,AMBER)]:
        box(x-.15,.708,.30,.045,'Input features',color='white',size=14)
        arrow((x,.704),(x,.683),col)
    box(.111,.630,.30,.047,'Weighted sum',color='#e2edff',size=14)
    box(.566,.630,.346,.047,'Tree 1 + Tree 2 + ...',color='#fff0c9',edge='#e8ca8e',size=14)
    for x,col in [(.261,BLUE),(.739,AMBER)]:
        arrow((x,.626),(x,.604),col)
        box(x-.15,.553,.30,.047,'Sigmoid → probability',color='white',size=13)
    text(.064,.526,'• Linear in the supplied features\n• Curves need feature transforms\n• Coefficients are not causal effects',13.4)
    text(.542,.526,'• Adds trees to reduce a loss\n• Learns nonlinear interactions\n• Depth and rounds need validation',13.4)
    plots=[]; images=[]
    for pos,key,index in [([.076,.281,.37,.142],'logistic',None),([.554,.281,.37,.142],'boosting',-1)]:
        panel=fig.add_axes(pos)
        values=data[key] if index is None else data[key][index]
        im=panel.imshow(values,origin='lower',extent=[data['gx'][0],data['gx'][-1],data['gy'][0],data['gy'][-1]],cmap='coolwarm',vmin=0,vmax=1,aspect='auto',interpolation='nearest',alpha=.8)
        points=data['test_x']; labels=data['test_y']
        panel.scatter(points[:,0],points[:,1],c=labels,cmap='coolwarm',vmin=0,vmax=1,s=7,edgecolors='white',linewidths=.25)
        panel.set_xlabel('feature 1 (arbitrary units)',fontsize=9);panel.set_ylabel('feature 2',fontsize=9)
        panel.tick_params(labelsize=8);panel.set_title('Fitted probability surface',fontsize=12,color=INK)
        plots.append(panel);images.append(im)
    text(.5,.245,'Synthetic two-moons illustration • dots: held-out labels • blue 0 ↔ red 1',10,ha='center')
    text(.5,.229,'600 generated rows • 420 training / 180 held out • fixed presets • no churn benchmark',9.5,ha='center')
    box(.035,.106,.452,.098,color='#eaf2ff')
    box(.513,.106,.452,.098,color='#fff3d5',edge='#e8ca8e')
    text(.064,.192,'TRY AS A BASELINE',14,'bold',BLUE)
    text(.064,.166,'A compact scoring rule, with\ninspectable feature coefficients.',13.8)
    text(.542,.192,'TRY FOR EXTRA FLEXIBILITY',14,'bold',AMBER)
    text(.542,.166,'Nonlinear patterns and interactions,\nwith validation to control overfit.',13.8)
    text(.5,.084,'Compare on the same held-out rows.',15,'bold',NAVY,'center')
    text(.5,.060,'For churn: inspect PR behavior, calibration and review-capacity errors.',11.7,ha='center')
    text(.5,.039,'No universal accuracy or speed winner. Preprocessing depends on implementation.',10.5,ha='center')
    source_caption=text(.5,.019,'Sources: scikit-learn User Guide • Original diagrams; fitted synthetic illustration',9.5,ha='center')
    counter=text(.739,.460,'30 trees',10,'bold',AMBER,'center')
    dots=[]
    for x,c in [(.261,BLUE),(.739,AMBER)]:
        d=Circle((x,.693),.0055,color=c);ax.add_patch(d);dots.append(d)
    fig.savefig(OUT/'comparison-infographic.png',dpi=144,facecolor='white')
    fig.savefig(OUT/'comparison-infographic.svg',facecolor='white')
    # Only this explicit caption distinguishes replay motion from wall-clock training.
    source_caption.set_text('Schematic flow + fitted-stage replay, not training time • Sources: scikit-learn')
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    fps,seconds=12,20
    command=[ffmpeg,'-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1080x1350','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'comparison-explainer.mp4')]
    process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    try:
        for frame in range(fps*seconds):
            section=min(4,frame//(fps*4))
            images[1].set_data(data['boosting'][section])
            counter.set_text(f'{int(data["stages"][section])} / 30 fitted trees')
            for d in dots:d.center=(d.center[0],.703-.092*((frame%fps)/fps))
            fig.canvas.draw()
            rgba=np.asarray(fig.canvas.buffer_rgba())
            process.stdin.write(rgba[:,:,:3].tobytes())
            if frame in [0,fps*seconds//2,fps*seconds-1]:
                fig.savefig(OUT/f'animation-frame-{frame}.png',dpi=100,facecolor='white')
        process.stdin.close();err=process.stderr.read().decode('utf-8',errors='replace');returncode=process.wait(timeout=60)
        if returncode:raise RuntimeError(err)
    except BaseException:
        process.kill();raise
    plt.close(fig)
    meta={'duration_seconds':seconds,'fps':fps,'frames':fps*seconds,'width':1080,'height':1350,'codec':'H264','pixel_format':'yuv420p','audio':'none','stage_intervals_seconds':4,'stages':data['stages'].tolist(),'motion_meaning':'Dots are schematic data flow; probability surfaces are actual saved fitted stages, not measured runtime.','static_alt':'Side-by-side educational infographic: logistic regression forms one linear feature score and applies a sigmoid; gradient boosting adds tree scores before applying a sigmoid. Fitted probability surfaces on the same synthetic two-moons data show their different shapes. Guidance compares an inspectable baseline with extra nonlinear flexibility, with validation and no universal winner.','clip_description':'20-second silent educational comparison. The left model stays fixed; the right surface replays actual fitted boosting stages1,5,10,20,30. Moving dots illustrate data flow, not timing. Synthetic data, not churn results.'}
    (OUT/'media.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(meta))


if __name__=='__main__':draw()
