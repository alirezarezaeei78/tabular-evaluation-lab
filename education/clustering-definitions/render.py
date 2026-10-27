"""Original educational infographic from the executed fit and saved values."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

root=Path(__file__).resolve().parent
data=np.loadtxt(root/'outputs/fit-points.csv',delimiter=',',skiprows=1)
r=json.loads((root/'outputs/results.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'text.color':'#15304d',
    'axes.labelcolor':'#15304d','axes.spines.top':False,'axes.spines.right':False})
fig=plt.figure(figsize=(9,11.25),facecolor='#ffffff')
fig.text(.07,.952,'K-MEANS vs DBSCAN',fontsize=27,weight='bold')
fig.text(.07,.919,'Two different definitions of a cluster',fontsize=17,color='#526579')
axes=[fig.add_axes([.08,.57,.39,.26]),fig.add_axes([.57,.57,.39,.26])]
colors=['#087e8b','#d58125','#7553a5','#5874b7']
for ax,column,title in [(axes[0],4,'Nearest centroid'),(axes[1],5,'Density-connected groups')]:
    labels=data[:,column].astype(int)
    for label in sorted(set(labels)):
        points=data[labels==label,2:4]
        ax.scatter(points[:,0],points[:,1],s=17,color='#626e78' if label==-1 else colors[label%len(colors)],
            marker='x' if label==-1 else 'o',label='Noise (−1)' if label==-1 else f'Group {label+1}',alpha=.9)
    if column==4:
        centers=np.asarray(r['kmeans']['centers_scaled']);ax.scatter(centers[:,0],centers[:,1],s=170,marker='X',color='#15304d',edgecolors='white',linewidths=1)
    ax.set_xlim(-2.6,2.6);ax.set_ylim(-3.5,3.5);ax.set_aspect('equal',adjustable='box')
    ax.set_xlabel('Scaled feature 1',fontsize=11);ax.set_ylabel('Scaled feature 2',fontsize=11)
    ax.tick_params(labelsize=9);ax.grid(alpha=.12)
fig.text(.08,.87,'K-means · k = 2',fontsize=17,weight='bold')
fig.text(.57,.87,'DBSCAN',fontsize=17,weight='bold')
fig.text(.08,.844,'Nearest centroid',fontsize=12)
fig.text(.57,.844,'Density-connected groups',fontsize=12)
fig.text(.08,.508,'244 assigned · 0 noise',fontsize=14,weight='bold')
fig.text(.57,.508,f"{r['density']['core_components']} groups · {r['density']['noise_rows']} noise rows",fontsize=14,weight='bold')
fig.text(.08,.482,'10 starts · seed 13',fontsize=12,color='#526579')
fig.text(.57,.482,'eps = 0.22 · min_samples = 5',fontsize=12,color='#526579')

def box(x,y,w,h,title,lines,color):
    fig.patches.append(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.008',transform=fig.transFigure,
        facecolor=color,edgecolor='#d3dce5',linewidth=.8,zorder=-1))
    fig.text(x+.018,y+h-.036,title,fontsize=15,weight='bold')
    fig.text(x+.018,y+h-.075,lines,fontsize=12.5,linespacing=1.6,va='top')
box(.07,.295,.405,.175,'K-means asks…','Which centroid is nearest?\nChoose k in advance.\nEvery fit row gets a group.\nUnits and outliers matter.','#eff5fc')
box(.525,.295,.405,.175,'DBSCAN asks…','Is this a dense neighborhood?\nChoose eps and min_samples.\nSome fit rows can be noise.\nUnequal densities can be hard.','#edf8f5')
box(.07,.105,.86,.145,'Common mistake: changing units, keeping eps','A = (0, 0), B = (0.1, 0.1): distance 0.1414\nMultiply only feature 1 by 100: distance 10.0005\nAt eps = 0.22, the neighbor decision changes.','#fff5e6')
fig.text(.07,.063,'Authored synthetic example · same 244 fit rows and train-fitted scaling',fontsize=11)
fig.text(.07,.039,'No customer segments or accuracy winner claimed. DBSCAN has no built-in predict.',fontsize=10.5)
fig.text(.07,.018,'Sources: scikit-learn 1.4.2 K-means, DBSCAN, StandardScaler docs · code + outputs in GitHub',fontsize=9.5,color='#526579')
for ext in ['png','svg']:fig.savefig(root/f'clustering-guide.{ext}',dpi=160,facecolor='white')
