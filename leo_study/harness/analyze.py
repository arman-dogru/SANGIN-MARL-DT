"""Build the 7 study figures (PNG+PDF) and per-figure CSVs from runs/main/*/result.json."""
import json, glob, math, os, sys
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv

STUDY=Path('/workspace/leo_study');FIG=STUDY/'figures';FIG.mkdir(exist_ok=True)
CUTOFF=.25;INTERVAL=.35;AREF_CONFIGURED=1.0
COUNTS=(2,4,8,16,32);RNG=np.random.default_rng(20261027);B=1000
plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':.3,'figure.dpi':150})

def load():
    rows=[]
    for f in glob.glob(str(STUDY/'runs/main/*/result.json')):
        p=json.load(open(f));j=p['job'];r=p['row']
        rows.append(dict(job=j,row=r,tasks=p['tasks'],pair=p['pair_stats']))
    return rows

def sel(rows,**kw):
    out=[]
    for x in rows:
        j=x['job']
        if all((j.get(k)==v if not isinstance(v,(tuple,list,set)) else j.get(k) in v) for k,v in kw.items()):out.append(x)
    return out

def boot_ratio(num,den):
    num=np.asarray(num,float);den=np.asarray(den,float)
    if len(num)==0 or den.sum()==0:return (math.nan,math.nan,math.nan)
    est=num.sum()/den.sum()
    idx=RNG.integers(0,len(num),(B,len(num)));d=den[idx].sum(1);n=num[idx].sum(1)
    vals=np.where(d>0,n/np.maximum(d,1e-12),np.nan)
    return est,np.nanpercentile(vals,2.5),np.nanpercentile(vals,97.5)

def boot_mean(values):
    v=np.asarray([x for x in values if x is not None and not (isinstance(x,float) and math.isnan(x))],float)
    if len(v)==0:return (math.nan,math.nan,math.nan)
    idx=RNG.integers(0,len(v),(B,len(v)));m=v[idx].mean(1)
    return v.mean(),np.percentile(m,2.5),np.percentile(m,97.5)

def wilson(k,n,z=1.96):
    if n==0:return (math.nan,math.nan,math.nan)
    p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return p,c-h,c+h

def write_csv(name,records):
    if not records:return
    keys=list(dict.fromkeys(k for r in records for k in r))
    with open(FIG/(name+'.csv'),'w',newline='') as fh:
        w=csv.DictWriter(fh,keys);w.writeheader();w.writerows(records)

EXPECTED=10;PARTIAL={}
FIG_GROUPS={'fig1':('nominal','delay','baseline'),'fig2':('nominal','delay','baseline'),'fig3':('nominal','baseline','delay'),
    'fig4':('nominal','delay'),'fig5':('nominal',),'fig6':('nominal','delay'),'fig7':('nominal','routes'),'fig8':('nominal','routes','routes_speed')}
def partial_cells(rows):
    """Cells of the planned matrix with fewer than 10 completed episodes (run in progress, OOM-killed or trimmed)."""
    plan=[]
    for f in glob.glob(str(STUDY/'runs/main/*/job.json')):
        j=json.load(open(f));plan.append(j)
    have=Counter((x['job']['group'],x['job']['treatment'],x['job']['route'],dl(x),x['job']['num_agents']) for x in rows)
    cells=Counter()
    for j in plan:cells[(j['group'],j['treatment'],j['route'],int(round(j.get('downlink_seconds',0)*1000)),j['num_agents'])]+=0
    for m in ('matrix.json','matrix_extra.json'):
        mp=STUDY/'runs'/m
        if mp.exists():
            for j in json.load(open(mp)):
                if isinstance(j,dict):cells[(j['group'],j['treatment'],j['route'],int(round(j.get('downlink_seconds',0)*1000)),j['num_agents'])]+=0
    out=defaultdict(list)
    for c in sorted(cells):
        n=have.get(c,0)
        if n<EXPECTED:out[c[0]].append((c,n))
    return out
def footnote(fig,name):
    key=name.split('_')[0]
    groups=FIG_GROUPS.get(key,())
    items=[f"{g} {t}/{r}{'' if d==0 else f' +{d}ms'} N={N}: n={n}" for g in groups for (g_,t,r,d,N),n in PARTIAL.get(g,[])]
    if not items:return
    txt='Partial cells (n of 10 episodes; N=32 run in progress, OOM/trimmed = missing): '+'; '.join(items)
    import textwrap
    fig.text(.5,-.02,'\n'.join(textwrap.wrap(txt,230)),ha='center',va='top',fontsize=6,color='0.35')
def save(fig,name):
    footnote(fig,name)
    for ext in ('png','pdf'):fig.savefig(FIG/f'{name}.{ext}',bbox_inches='tight')
    plt.close(fig)

def err(ax,x,est,lo,hi,**kw):
    est,lo,hi=map(lambda a:np.asarray(a,float),(est,lo,hi))
    kw.setdefault('marker','o');kw.setdefault('ms',4)
    ax.errorbar(x,est,yerr=[np.clip(est-lo,0,None),np.clip(hi-est,0,None)],capsize=3,**kw)

DELAY_LABEL={0:'nominal LEO',150:'+150 ms DL',300:'+300 ms DL'}
def dl(x):return int(round(x['job'].get('downlink_seconds',0)*1000))

# ---------------- Fig 1: AoI vs priority-ordering inconsistency ----------------
def fig1(rows):
    recs=[];fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    groups=[('ideal network',dict(group='baseline',treatment='A4')),('nominal LEO',dict(group='nominal',treatment='A4')),
            ('+150 ms DL',dict(group='delay',treatment='A4',downlink_seconds=.15)),('+300 ms DL',dict(group='delay',treatment='A4',downlink_seconds=.3))]
    edges=np.array([0,.3,.45,.7,1.0,1.4,2.,3.,60.])
    for label,kw in groups:
        sub=[x for x in sel(rows,**kw) if x['job']['num_agents']>=8]
        if not sub:continue
        F=sub[0]['pair']['aoi_fields'];width=sub[0]['pair']['aoi_bin_seconds']
        for ai,(ax,relevant_only) in enumerate(zip(axes,(False,True))):
            n=np.zeros(len(edges)-1);e=np.zeros(len(edges)-1);per_ep=[]
            for x in sub:
                en=np.zeros(len(edges)-1);ee=np.zeros(len(edges)-1)
                for rec in x['pair']['aoi']:
                    d=dict(zip(F,rec))
                    if not d['active'] or (relevant_only and not d['relevant']):continue
                    age=(d['aoi_bin']+.5)*width;k=min(np.searchsorted(edges,age,side='right')-1,len(edges)-2)
                    en[k]+=d['count'];ee[k]+=d['count']*d['e2e_error']
                per_ep.append((en,ee));n+=en;e+=ee
            mids=np.sqrt(np.maximum(edges[:-1],.05)*edges[1:])
            est,lo,hi=[],[],[]
            for k in range(len(mids)):
                r=boot_ratio([p[1][k] for p in per_ep],[p[0][k] for p in per_ep]) if n[k]>=30 else (math.nan,)*3
                est.append(r[0]);lo.append(r[1]);hi.append(r[2])
                recs.append(dict(panel='FoV-relevant pairs' if relevant_only else 'all active pairs',condition=label,
                    aoi_bin_low_s=edges[k],aoi_bin_high_s=edges[k+1],pair_decisions=int(n[k]),e2e_errors=int(e[k]),
                    error_rate=r[0],ci_low=r[1],ci_high=r[2]))
            err(ax,mids,est,lo,hi,label=label)
    for ax,t in zip(axes,('All active directed pairs','FoV-relevant pairs (predicted conflict)')):
        ax.set_xscale('log');ax.set_xlabel('Age of information, max(self, peer) [s]  (Δ = 0.35 s)')
        ax.set_ylabel('E2E priority-ordering error rate');ax.set_title(t);ax.axvline(INTERVAL,color='grey',ls=':',lw=1)
    axes[0].legend(fontsize=7)
    fig.suptitle('Fig. 1  AoI vs priority-ordering inconsistency (A4 heterogeneous + cached fallback, N = 8/16/32 pooled; 95% case-bootstrap CI)',fontsize=9)
    save(fig,'fig1_aoi_vs_inconsistency')
    # Episode-level xi_AoI with the configured A_ref and with A_ref = Δ
    ep=[]
    for label,kw in groups:
        for n in COUNTS:
            sub=sel(rows,num_agents=n,**kw)
            if not sub:continue
            a=boot_ratio([x['row']['aoi_weighted_error_sum'] for x in sub],[x['row']['relevant_pairs'] for x in sub])
            ep.append(dict(condition=label,num_agents=n,episodes=len(sub),xi_aoi_Aref_1s=a[0],ci_low=a[1],ci_high=a[2],
                xi_aoi_Aref_interval=a[0]*AREF_CONFIGURED/INTERVAL if not math.isnan(a[0]) else math.nan,
                xi_fov=boot_ratio([x['row']['fov_errors'] for x in sub],[x['row']['relevant_pairs'] for x in sub])[0]))
    write_csv('fig1_aoi_vs_inconsistency',recs);write_csv('fig1_xi_aoi_by_condition',ep)
    return recs

# ---------------- Fig 2: UL / DL / E2E ----------------
def fig2(rows):
    recs=[];conds=[('ideal',dict(group='baseline',treatment='A4')),('nominal',dict(group='nominal',treatment='A4')),
        ('+150 ms',dict(group='delay',treatment='A4',downlink_seconds=.15)),('+300 ms',dict(group='delay',treatment='A4',downlink_seconds=.3))]
    fig,axes=plt.subplots(2,3,figsize=(12,6.2),sharex=True)
    for row_i,(scope,den,nums) in enumerate((('all directed pairs','directed_pairs',('uplink_errors','downlink_errors','ordering_errors')),
            ('FoV-relevant pairs','relevant_pairs',('fov_uplink_errors','fov_downlink_errors','fov_errors')))):
        for col,(name,num) in enumerate(zip(('UL (DT vs physical)','DL (cached vs DT)','E2E (cached vs physical)'),nums)):
            ax=axes[row_i,col]
            for ci,(label,kw) in enumerate(conds):
                xs,est,lo,hi=[],[],[],[]
                for i,n in enumerate(COUNTS):
                    sub=sel(rows,num_agents=n,**kw)
                    if not sub:continue
                    r=boot_ratio([x['row'][num] for x in sub],[x['row'][den] for x in sub])
                    xs.append(i+(ci-1.5)*.12);est.append(r[0]);lo.append(r[1]);hi.append(r[2])
                    recs.append(dict(scope=scope,error=name,condition=label,num_agents=n,episodes=len(sub),
                        errors=sum(x['row'][num] for x in sub),pairs=sum(x['row'][den] for x in sub),rate=r[0],ci_low=r[1],ci_high=r[2]))
                if xs:err(ax,xs,est,lo,hi,label=label,ls='none')
            ax.set_title(f'{name} — {scope}',fontsize=8);ax.set_xticks(range(len(COUNTS)));ax.set_xticklabels(COUNTS)
            if row_i==1:ax.set_xlabel('Robots N')
            if col==0:ax.set_ylabel('Ordering-error rate (errors / pair-decisions)')
    axes[0,0].legend(fontsize=7)
    fig.suptitle('Fig. 2  Physical vs DT/cached priority-ordering errors (A4; 95% case-bootstrap CI). 8/16/32 only for delay levels.',fontsize=9)
    save(fig,'fig2_ul_dl_e2e_errors');write_csv('fig2_ul_dl_e2e_errors',recs)

# ---------------- Fig 3: reversals vs population ----------------
def fig3(rows):
    recs=[];fig,axes=plt.subplots(1,3,figsize=(12,3.6))
    for label,kw,color in (('homogeneous (A3)',dict(group='nominal',treatment='A3'),'C0'),('heterogeneous (A4)',dict(group='nominal',treatment='A4'),'C3'),
                           ('homogeneous, ideal net',dict(group='baseline',treatment='A3'),'C0'),('heterogeneous, ideal net',dict(group='baseline',treatment='A4'),'C3')):
        ls='--' if 'ideal' in label else '-'
        for ax,(metric,fn) in zip(axes,(('movement reversals / episode',lambda s:boot_mean([x['row']['movement_priority_reversals'] for x in s])),
                ('reversals per 100 active pair-seconds',lambda s:boot_ratio([100*x['row']['movement_priority_reversals'] for x in s],[x['row']['active_pair_seconds'] for x in s])),
                ('FoV-relevant reversals / episode',lambda s:boot_mean([x['row']['relevant_movement_priority_reversals'] for x in s])))):
            xs,est,lo,hi=[],[],[],[]
            for n in COUNTS:
                sub=sel(rows,num_agents=n,**kw)
                if not sub:continue
                r=fn(sub);xs.append(n);est.append(r[0]);lo.append(r[1]);hi.append(r[2])
                recs.append(dict(metric=metric,setting=label,num_agents=n,episodes=len(sub),value=r[0],ci_low=r[1],ci_high=r[2]))
            if xs:err(ax,xs,est,lo,hi,label=label,color=color,ls=ls)
            ax.set_xscale('log',base=2);ax.set_xticks(COUNTS);ax.set_xticklabels(COUNTS);ax.set_xlabel('Robots N');ax.set_ylabel(metric)
    axes[0].legend(fontsize=7)
    fig.suptitle('Fig. 3  Physical priority reversals vs population, homogeneous vs heterogeneous speeds (cached fallback; 95% CI)',fontsize=9)
    save(fig,'fig3_reversals_vs_population');write_csv('fig3_reversals_vs_population',recs)

# ---------------- Fig 4: conflict detection ----------------
def conflict_counts(x):
    F=x['pair']['conflict_fields'];c=Counter()
    for rec in x['pair']['conflict']:
        d=dict(zip(F,rec));k=d['count']
        if d['physical_relevant'] and not d['cached_relevant']:c['missed_orig']+=k;c['missed_active']+=k*d['active']
        if d['cached_relevant'] and not d['physical_relevant']:c['fp_orig']+=k;c['fp_active']+=k*d['active']
        if d['cached_relevant'] and d['physical_relevant']:c['hit_active']+=k*d['active']
        if d['physical_relevant']:c['phys_active']+=k*d['active']
    act=sum(rec[-1] for rec in x['pair']['aoi'] if rec[1]);c['active_pairs']=act
    return c

def fig4(rows):
    recs=[];conds=[('ideal',dict(group='baseline',treatment='A4')),('nominal',dict(group='nominal',treatment='A4')),
        ('+150 ms',dict(group='delay',treatment='A4',downlink_seconds=.15)),('+300 ms',dict(group='delay',treatment='A4',downlink_seconds=.3))]
    fig,axes=plt.subplots(1,4,figsize=(14,3.6))
    for ci,(label,kw) in enumerate(conds):
        for ax,(metric,fn) in zip(axes,(
            ('missed conflicts per 1k active pair-decisions',lambda s:boot_ratio([1000*conflict_counts(x)['missed_active'] for x in s],[conflict_counts(x)['active_pairs'] for x in s])),
            ('false-positive conflicts per 1k active pair-decisions',lambda s:boot_ratio([1000*conflict_counts(x)['fp_active'] for x in s],[conflict_counts(x)['active_pairs'] for x in s])),
            ('detection recall (cached ∧ physical / physical)',lambda s:boot_ratio([conflict_counts(x)['hit_active'] for x in s],[conflict_counts(x)['phys_active'] for x in s])),
            ('actual attempted movement conflicts / episode',lambda s:boot_mean([x['row']['attempted_vertex_conflicts']+x['row']['attempted_edge_swaps'] for x in s])))):
            xs,est,lo,hi=[],[],[],[]
            for i,n in enumerate(COUNTS):
                sub=sel(rows,num_agents=n,**kw)
                if not sub:continue
                r=fn(sub);xs.append(i+(ci-1.5)*.12);est.append(r[0]);lo.append(r[1]);hi.append(r[2])
                cc=Counter();[cc.update(conflict_counts(x)) for x in sub]
                recs.append(dict(metric=metric,condition=label,num_agents=n,episodes=len(sub),value=r[0],ci_low=r[1],ci_high=r[2],
                    missed_original_def=cc['missed_orig'],missed_active_def=cc['missed_active'],fp_original_def=cc['fp_orig'],fp_active_def=cc['fp_active'],
                    physical_active=cc['phys_active'],active_pair_decisions=cc['active_pairs']))
            if xs:err(ax,xs,est,lo,hi,label=label,ls='none')
            ax.set_title(metric,fontsize=8);ax.set_xticks(range(len(COUNTS)));ax.set_xticklabels(COUNTS);ax.set_xlabel('Robots N')
    axes[0].legend(fontsize=7)
    fig.suptitle('Fig. 4  Cached vs physical conflict detection (A4; active pairs only, corrected definition; 95% CI)',fontsize=9)
    save(fig,'fig4_conflict_detection');write_csv('fig4_conflict_detection',recs)

# ---------------- Fig 5: latency CDF / CCDF ----------------
def fig5(rows):
    recs=[];fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    conds=[('allocator, nominal',dict(group='nominal',treatment='A4',route='allocated')),('all-UAV, nominal',dict(group='routes',route='edge')),
           ('all-cloud, nominal',dict(group='routes',route='cloud')),('allocator, +150 ms DL',dict(group='delay',treatment='A4',downlink_seconds=.15)),
           ('allocator, +300 ms DL',dict(group='delay',treatment='A4',downlink_seconds=.3))]
    for label,kw in conds:
        sub=sel(rows,**kw)
        lat=np.sort([t['cycle_latency_seconds'] for x in sub for t in x['tasks'] if t['cycle_latency_seconds'] is not None])
        undelivered=sum(t['cycle_latency_seconds'] is None for x in sub for t in x['tasks'])
        if len(lat)==0:continue
        total=len(lat)+undelivered
        cdf=np.arange(1,len(lat)+1)/total
        axes[0].step(lat*1000,cdf,where='post',label=label);axes[1].step(lat*1000,np.maximum(1-cdf,1/total/2),where='post',label=label)
        q=lambda p:float(np.quantile(lat,p))*1000
        recs.append(dict(condition=label,tasks=total,delivered=len(lat),undelivered=undelivered,p50_ms=q(.5),p95_ms=q(.95),p99_ms=q(.99),
            frac_after_250ms_incl_undelivered=(np.sum(lat>CUTOFF)+undelivered)/total))
    for ax in axes:
        ax.axvline(CUTOFF*1000,color='k',ls='--',lw=1,label='250 ms cutoff');ax.set_xlabel('Command delivery latency from cycle start [ms]')
    axes[0].set_ylabel('CDF (undelivered counted as > all)');axes[1].set_ylabel('CCDF  P(latency > x)');axes[1].set_yscale('log')
    axes[0].legend(fontsize=7)
    fig.suptitle('Fig. 5  Command-delivery latency CDF/CCDF (A4, all N pooled; delay levels at N = 8/16/32)',fontsize=9)
    save(fig,'fig5_latency_cdf_ccdf');write_csv('fig5_latency_summary',recs)
    # raw quantile grid for re-plotting
    grid=[]
    for label,kw in conds:
        lat=[t['cycle_latency_seconds'] for x in sel(rows,**kw) for t in x['tasks'] if t['cycle_latency_seconds'] is not None]
        if lat:
            for p in np.linspace(0,1,201):grid.append(dict(condition=label,quantile=round(p,3),latency_ms=float(np.quantile(lat,p))*1000))
    write_csv('fig5_latency_quantiles',grid)

# ---------------- Fig 6: fallback effectiveness ----------------
def fig6(rows):
    recs=[];fig,axes=plt.subplots(1,4,figsize=(15,3.6))
    pairs=[('heterogeneous',('A2','A4')),('homogeneous',('A1','A3'))]
    levels=[(0,'nominal'),(150,'+150 ms'),(300,'+300 ms')]
    xi=0;ticks=[];labels=[]
    for speed,(wait_t,cache_t) in pairs:
        for lv,lname in levels:
            for n in (8,16,32):
                kwbase=dict(group='nominal') if lv==0 else dict(group='delay',downlink_seconds=lv/1000)
                W=sel(rows,treatment=wait_t,num_agents=n,route='allocated',**kwbase);C=sel(rows,treatment=cache_t,num_agents=n,route='allocated',**kwbase)
                if not W or not C:continue
                for off,(name,sub,color) in enumerate((('WAIT',W,'C1'),('cached fallback',C,'C2'))):
                    succ=wilson(sum(x['row']['success'] for x in sub),len(sub))
                    rec_mv=boot_mean([x['row']['deadline_fallback_movements']+x['row']['extra_cached_movements'] for x in sub])
                    waits=boot_mean([x['row']['scheduled_cutoff_waits'] for x in sub])
                    miss=boot_mean([x['row']['scheduled_command_misses'] for x in sub])
                    mk=boot_mean([x['row']['makespan_seconds'] for x in sub if x['row']['success']])
                    xpos=xi+(off-.5)*.3
                    for ax,v in zip(axes,(succ,rec_mv,waits,mk)):
                        err(ax,[xpos],[v[0]],[v[1]],[v[2]],color=color,ls='none',label=name if xi==0 else None)
                    recs.append(dict(speed=speed,downlink_added_ms=lv,num_agents=n,policy=name,treatment=sub[0]['job']['treatment'],episodes=len(sub),
                        success=succ[0],success_ci_low=succ[1],success_ci_high=succ[2],recovered_movements=rec_mv[0],recovered_ci_low=rec_mv[1],recovered_ci_high=rec_mv[2],
                        cutoff_misses=miss[0],cutoff_waits=waits[0],cutoff_waits_ci_low=waits[1],cutoff_waits_ci_high=waits[2],
                        successful_makespan_s=mk[0]))
                # avoided waits = cutoff misses that did not become WAIT under cached fallback
                ticks.append(xi);labels.append(f"{'het' if speed=='heterogeneous' else 'hom'}\n{lname}\nN={n}");xi+=1
    for ax,t in zip(axes,('Mission success (Wilson 95% CI)','Recovered movements / episode\n(fallback moves after missed cutoff)',
                          'Cutoff-induced WAITs / episode\n(WAIT minus fallback = avoided WAITs)','Successful makespan [s]')):
        ax.set_title(t,fontsize=8);ax.set_xticks(ticks);ax.set_xticklabels(labels,fontsize=5.5)
    axes[0].legend(fontsize=7)
    fig.suptitle('Fig. 6  Fallback effectiveness: cached-path fallback (A3/A4) vs WAIT (A1/A2) across added downlink delay',fontsize=9)
    save(fig,'fig6_fallback_effectiveness')
    # avoided waits
    by={(r['speed'],r['downlink_added_ms'],r['num_agents'],r['policy']):r for r in recs}
    for r in recs:
        if r['policy']=='cached fallback':
            w=by.get((r['speed'],r['downlink_added_ms'],r['num_agents'],'WAIT'))
            if w:r['avoided_waits_vs_WAIT']=w['cutoff_waits']-r['cutoff_waits'];r['success_gain_vs_WAIT']=r['success']-w['success']
    write_csv('fig6_fallback_effectiveness',recs)

# ---------------- Fig 7: latency-energy-success ----------------
def fig7(rows):
    recs=[];fig,axes=plt.subplots(1,3,figsize=(13,3.8))
    routes=[('allocator',dict(group='nominal',treatment='A4',route='allocated'),'C0','o'),('all-UAV',dict(group='routes',route='edge'),'C1','s'),
            ('all-cloud',dict(group='routes',route='cloud'),'C2','^')]
    for label,kw,color,marker in routes:
        pts=[]
        for n in COUNTS:
            sub=sel(rows,num_agents=n,**kw)
            if not sub:continue
            lat=boot_mean([x['row']['task_latency_p95_seconds']*1000 for x in sub if x['row'].get('task_latency_p95_seconds') is not None])
            en=boot_mean([x['row']['modeled_energy_lower_j'] for x in sub])
            enm=boot_mean([x['row']['modeled_energy_lower_j']/max(1e-9,x['row']['observed_seconds'])*60 for x in sub])
            s=wilson(sum(x['row']['success'] for x in sub),len(sub))
            ontime=boot_ratio([x['row']['deadline_commands_on_time'] for x in sub],[x['row']['deadline_tasks_assessed'] or 0 for x in sub])
            recs.append(dict(route=label,num_agents=n,episodes=len(sub),latency_p95_ms=lat[0],latency_ci_low=lat[1],latency_ci_high=lat[2],
                energy_per_mission_j=en[0],energy_ci_low=en[1],energy_ci_high=en[2],energy_per_minute_j=enm[0],
                success=s[0],success_ci_low=s[1],success_ci_high=s[2],on_time_fraction=ontime[0]))
            pts.append((n,lat,en,s,enm))
        if not pts:continue
        x=[p[0] for p in pts]
        err(axes[0],[p[1][0] for p in pts],[p[4][0] for p in pts],[p[4][1] for p in pts],[p[4][2] for p in pts],color=color,marker=marker,label=label)
        for p in pts:axes[0].annotate(str(p[0]),(p[1][0],p[4][0]),fontsize=6,xytext=(3,3),textcoords='offset points')
        err(axes[1],x,[p[1][0] for p in pts],[p[1][1] for p in pts],[p[1][2] for p in pts],color=color,marker=marker,label=label)
        err(axes[2],x,[p[3][0] for p in pts],[p[3][1] for p in pts],[p[3][2] for p in pts],color=color,marker=marker,label=label)
    axes[0].set_xlabel('Per-episode p95 task latency [ms] (labels = N)');axes[0].set_ylabel('Modeled energy per minute of mission [J/min]')
    axes[1].set_ylabel('p95 task latency [ms]');axes[1].axhline(CUTOFF*1000,color='k',ls='--',lw=1)
    axes[2].set_ylabel('Mission success (Wilson 95% CI)')
    for ax in axes[1:]:ax.set_xscale('log',base=2);ax.set_xticks(COUNTS);ax.set_xticklabels(COUNTS);ax.set_xlabel('Robots N')
    axes[0].legend(fontsize=7)
    fig.suptitle('Fig. 7  Latency–energy–success: allocator vs all-UAV vs all-cloud (A4, nominal LEO; energy = comms + UAV compute)',fontsize=9)
    save(fig,'fig7_latency_energy_success');write_csv('fig7_latency_energy_success',recs)


# ---------------- Fig 8 (add-on): routes x speed setting ----------------
def fig8(rows):
    recs=[];metrics=[('Mission success',lambda s:wilson(sum(x['row']['success'] for x in s),len(s))),
        ('Successful completion time [s]',lambda s:boot_mean([x['row']['makespan_seconds'] for x in s if x['row']['success']])),
        ('Waiting steps / episode',lambda s:boot_mean([x['row']['waiting_steps'] for x in s])),
        ('Attempted collisions / episode',lambda s:boot_mean([x['row']['attempted_vertex_conflicts']+x['row']['attempted_edge_swaps'] for x in s])),
        ('Modeled energy / mission [J]',lambda s:boot_mean([x['row']['modeled_energy_lower_j'] for x in s])),
        ('p95 task latency [ms]',lambda s:boot_mean([x['row']['task_latency_p95_seconds']*1000 for x in s if x['row'].get('task_latency_p95_seconds') is not None])),
        ('E2E ordering-error rate (all pairs)',lambda s:boot_ratio([x['row']['ordering_errors'] for x in s],[x['row']['directed_pairs'] for x in s])),
        ('FoV ordering-error rate (xi_FoV)',lambda s:boot_ratio([x['row']['fov_errors'] for x in s],[x['row']['relevant_pairs'] for x in s]))]
    cells=[('allocator','same speed (A3)',dict(group='nominal',treatment='A3',route='allocated')),
           ('all-UAV','same speed (A3)',dict(group='routes_speed',treatment='A3',route='edge')),
           ('all-cloud','same speed (A3)',dict(group='routes_speed',treatment='A3',route='cloud')),
           ('allocator','mixed speed (A4)',dict(group='nominal',treatment='A4',route='allocated')),
           ('all-UAV','mixed speed (A4)',dict(group='routes',treatment='A4',route='edge')),
           ('all-cloud','mixed speed (A4)',dict(group='routes',treatment='A4',route='cloud'))]
    fig,axes=plt.subplots(2,4,figsize=(16,7.2));axes=axes.ravel();fig.subplots_adjust(hspace=.38,wspace=.28)
    colors={'allocator':'C0','all-UAV':'C1','all-cloud':'C2'}
    for ci,(route,speed,kw) in enumerate(cells):
        for ax,(name,fn) in zip(axes,metrics):
            xs,est,lo,hi=[],[],[],[]
            for i,n in enumerate(COUNTS):
                sub=sel(rows,num_agents=n,**kw)
                if not sub:continue
                r=fn(sub);xs.append(i+(ci-2.5)*.1);est.append(r[0]);lo.append(r[1]);hi.append(r[2])
                recs.append(dict(metric=name,route=route,speed=speed,num_agents=n,episodes=len(sub),value=r[0],ci_low=r[1],ci_high=r[2]))
            if xs:err(ax,xs,est,lo,hi,color=colors[route],ls='none',marker='o' if 'A3' in speed else '^',
                      mfc='white' if 'A3' in speed else colors[route],label=f'{route}, {speed}')
            ax.set_title(name,fontsize=8);ax.set_xticks(range(len(COUNTS)));ax.set_xticklabels(COUNTS);ax.set_xlabel('Robots N')
    axes[0].legend(fontsize=6)
    fig.suptitle('Fig. 8  Allocator vs all-UAV vs all-cloud under same speed (A3, open circles) vs mixed speed (A4, triangles); nominal LEO, cached fallback; 95% CI',fontsize=9)
    save(fig,'fig8_routes_by_speed');write_csv('fig8_routes_by_speed',recs)

def summary(rows):
    """Every episode-level number plus baseline deltas (ideal network, same N and speed setting)."""
    out=[];keys=('success','makespan_seconds','waiting_steps','attempted_vertex_conflicts','attempted_edge_swaps','post_resolution_conflicts',
        'modeled_energy_lower_j','modeled_energy_upper_j','task_latency_p95_seconds','command_cutoff_misses','cached_fallback_decisions',
        'fallback_movements','scheduled_cutoff_waits','movement_priority_reversals','ordering_errors','uplink_errors','downlink_errors',
        'directed_pairs','fov_errors','relevant_pairs','xi_fov','xi_aoi','speed_limited_nonmovement','completion_fraction','sim_wall_seconds')
    cells=defaultdict(list)
    for x in rows:
        j=x['job'];cells[(j['group'],j['treatment'],j['route'],j['network'],dl(x),j['num_agents'])].append(x)
    base={}
    for (g,t,r,net,d,n),sub in cells.items():
        if g=='baseline':base[('het' if t in ('A2','A4') else 'hom',n)]={k:np.nanmean([float(s['row'].get(k)) if s['row'].get(k) is not None else np.nan for s in sub]) for k in keys}
    for (g,t,r,net,d,n),sub in sorted(cells.items()):
        rec=dict(group=g,treatment=t,route=r,network=net,downlink_added_ms=d,num_agents=n,episodes=len(sub))
        b=base.get(('het' if t in ('A2','A4') else 'hom',n),{})
        for k in keys:
            vals=[float(s['row'].get(k)) if s['row'].get(k) is not None else np.nan for s in sub]
            m=np.nanmean(vals) if not all(np.isnan(vals)) else np.nan
            rec[k]=m
            if k in b:rec[k+'__delta_vs_ideal']=m-b[k]
        out.append(rec)
    write_csv('summary_by_cell_with_baseline_deltas',out)
    eps=[]
    for x in rows:
        e=dict(x['job']);e.pop('priority',None);e.update({k:x['row'].get(k) for k in keys});eps.append(e)
    write_csv('episodes',eps)

if __name__=='__main__':
    rows=load();print('episodes',len(rows))
    PARTIAL.update(partial_cells(rows))
    write_csv('partial_cells',[dict(group=g,treatment=c[1],route=c[2],downlink_ms=c[3],num_agents=c[4],episodes=n) for g,v in PARTIAL.items() for c,n in v])
    for f in (fig1,fig2,fig3,fig4,fig5,fig6,fig7,fig8,summary):
        try:f(rows);print('ok',f.__name__)
        except Exception as e:
            import traceback;traceback.print_exc();print('FAILED',f.__name__)
