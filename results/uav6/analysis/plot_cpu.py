"""Zoomed CPU/load diagnostic from immutable parameter-only summary values."""
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parents[1]
summary=json.loads((root/'summary.json').read_text())
conditions=('allocated-1ghz','allocated-6ghz','uav-6ghz','cloud-6ghz')
colors=('#666666','#009E73','#0072B2','#D55E00')
labels=('Allocator 1 GHz','Allocator 6 GHz','All UAV 6 GHz','All cloud')
plt.rcParams.update({'font.family':'serif','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none',
    'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(9,3.4),layout='constrained')
for condition,color,label in zip(conditions,colors,labels):
    rows=sorted([r for r in summary if r['condition']==condition],key=lambda r:r['num_agents'])
    x=[r['num_agents'] for r in rows]
    for pool,style in (('drone','-'),('cloud','--')):
        axes[0].plot(x,[100*r[pool+'_utilization'] for r in rows],color=color,ls=style,marker='o',label=label if pool=='drone' else None)
        axes[1].plot(x,[r[pool+'_concurrent_max'] for r in rows],color=color,ls=style,marker='o')
axes[0].set(ylabel='Time-weighted CPU utilization (%)',ylim=(-.2,7))
axes[0].legend(frameon=False,fontsize=8)
axes[1].set(ylabel='Peak concurrent CPU jobs',ylim=(-.2,8.7),yticks=(0,2,4,6,8))
for ax in axes:
    ax.set(xlabel='Robots',xticks=(2,4,8));ax.grid(alpha=.2)
fig.suptitle('Parameter-only pilot results — solid UAV, dashed cloud',fontsize=11)
for ext in ('pdf','svg','png'):
    fig.savefig(root/'figures'/('cpu_load.'+ext),dpi=300,metadata={'CreationDate':None} if ext=='pdf' else None)
plt.close(fig)
