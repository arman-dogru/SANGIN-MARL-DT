"""Orbit, RF, control protocol and measured transport visualization.

Outputs static scientific figures and a self-contained interactive orbital view.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skyfield.api import EarthSatellite,load
from skyfield.framelib import itrs
from datetime import datetime,UTC

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/"src"))
from panama_uav.leo import OrbitTimeline
from panama_uav.records import read_jsonl,write_json


def figure(fig,output,name):
    fig.savefig(output/(name+".png"),dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeline",default="data/leo/tremblant_full.npz")
    parser.add_argument("--output",default="output/leo/leo-full-20261009-01")
    args=parser.parse_args()
    timeline=OrbitTimeline.load(PROJECT/args.timeline)
    output=PROJECT/args.output
    output.mkdir(parents=True,exist_ok=True)
    s=timeline.series
    t=timeline.times
    fig,axes=plt.subplots(4,2,figsize=(14,12),sharex=True,layout="constrained")
    axes[0,0].plot(t/60,s["elevation_degrees"],label="UAV")
    axes[0,0].plot(t/60,s["gateway_elevation_degrees"],label="Gateway",alpha=.7)
    axes[0,0].set_ylabel("Elevation (degrees)")
    axes[0,1].plot(t/60,s["range_km"],label="UAV")
    axes[0,1].plot(t/60,s["gateway_range_km"],label="Gateway",alpha=.7)
    axes[0,1].set_ylabel("Space-leg range (km)")
    axes[1,0].plot(t/60,s["uplink_doppler_hz"]/1000,label="Uplink")
    axes[1,0].plot(t/60,s["downlink_doppler_hz"]/1000,label="Downlink")
    axes[1,0].set_ylabel("Raw Doppler (kHz)")
    axes[1,1].plot(t/60,timeline.propagation*1000,label="Two space legs + ground")
    axes[1,1].set_ylabel("One-way propagation + ground (ms)")
    for direction,label in (("up","Uplink"),("down","Downlink")):
        axes[2,0].plot(t/60,s["uav_"+direction+"_atmosphere_db"],label=label)
        axes[2,1].plot(t/60,s["uav_"+direction+"_cn_db"],label=label)
        cap=10 if direction=="up" else 50
        axes[3,0].plot(t/60,np.minimum(cap,s["satellite_"+direction+"_bps"]/1e6),label=label)
    axes[2,0].set_ylabel("ITU-R atmospheric loss (dB)")
    axes[2,1].set_ylabel("Engineering C/N (dB)")
    axes[3,0].set_ylabel("Allocated radio service ceiling (Mbps)")
    axes[3,1].plot(t/60,s["visible_satellites"],label="Visible to UAV + gateway")
    axes[3,1].set_ylabel("Candidate satellites")
    for ax in axes.flat:
        ax.grid(alpha=.2); ax.legend(fontsize=7); ax.margins(x=.01)
    for ax in axes[-1]: ax.set_xlabel("Archived orbital time (minutes)")
    figure(fig,output,"leo_geometry_radio")
    events=[event for event in timeline.events if 1000<=event["start"]<=1300]
    fig,axes=plt.subplots(2,1,figsize=(12,6),layout="constrained")
    selection=np.unique(s["satellite_ids"][(t>=1000)&(t<=1300)])
    lookup={sat:i for i,sat in enumerate(selection)}
    mask=(t>=1000)&(t<=1300)
    axes[0].step(t[mask]-1000,[lookup[x] for x in s["satellite_ids"][mask]],where="post")
    axes[0].set(yticks=list(lookup.values()),yticklabels=[str(int(x)) for x in lookup],ylabel="Serving NORAD ID",xlabel="Window time (s)")
    for event in events:
        axes[0].axvline(event["start"]-1000,color="#db7c34",alpha=.4)
    event=next(e for e in events if e.get("attachment_changed"))
    for index,control in enumerate(event["control_messages"]):
        left=(control["start"]-event["start"])*1000
        width=(control["end"]-control["start"])*1000
        color="#2c7eb5" if control.get("direction")=="uplink" else "#df7c36"
        axes[1].barh(index,width,left=left,color=color)
    axes[1].set(yticks=range(len(event["control_messages"])),yticklabels=[c["kind"] for c in event["control_messages"]],xlabel="Time since control event (ms)",ylabel="Protocol step")
    axes[1].invert_yaxis()
    for ax in axes: ax.grid(alpha=.2)
    fig.suptitle("Scheduling boundaries and an individual handshake · assumed protocol")
    figure(fig,output,"leo_handover_handshake")
    # Use every recorded delivery for ECDFs; the workload is transport-only.
    rows=read_jsonl(output/"events.jsonl")
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout="constrained")
    report={}
    for col,kind in enumerate(("experience","model")):
        for agents in (2,4,6,8):
            ages=np.array([r["age_seconds"] for r in rows if r["event"]=="delivered" and r["kind"]==kind and r["num_agents"]==agents and r["handovers"]])
            if not len(ages): continue
            ordered=np.sort(ages)
            axes[0,col].plot(ordered,np.arange(1,len(ordered)+1)/len(ordered),label=f"Stage {(2,4,6,8).index(agents)+1}")
            report[f"{agents}_{kind}"]={"count":len(ages),"mean_s":float(ages.mean()),"p95_s":float(np.percentile(ages,95)),"max_s":float(ages.max())}
        tx=[r for r in rows if r["event"]=="transmit" and r.get("kind") in (("model","model_relay") if kind=="model" else (kind,)) and r["num_agents"]==8 and r["handovers"]]
        fields=("queue_seconds","transmission_seconds","handover_seconds","frame_guard_seconds","propagation_seconds")
        axes[1,col].bar(range(len(fields)),[np.mean([r.get(key,0) for r in tx]) for key in fields])
        axes[1,col].set(xticks=range(len(fields)),xticklabels=["Queue","Serialize","Interrupt","Guard","Propagate"],ylabel="Mean per-hop component (s)")
        axes[0,col].set(xlabel="Complete-message delivery age (s)",ylabel="Cumulative fraction",title=kind.capitalize())
        axes[0,col].legend()
    for ax in axes.flat: ax.grid(alpha=.2)
    figure(fig,output,"leo_latency_components")
    # Queue traces show offered load, including unfinished transmissions.
    queue_rows=read_jsonl(output/"queue_timelines.jsonl")
    fig,axes=plt.subplots(2,1,figsize=(12,6),sharex=True,layout="constrained")
    group=[r for r in queue_rows if r["num_agents"]==8 and r["handovers"]]
    for key in ("satellite_up","satellite_down","access_up","access_down","transmitting_remaining","in_flight"):
        axes[0].plot([r["step"] for r in group],[r["queue_bytes"].get(key,0)/1e6 for r in group],label=key)
    axes[0].set_ylabel("Pending message bytes (MB)"); axes[0].legend(fontsize=8)
    for kind in ("experience","model"):
        received=[r for r in rows if r["event"]=="delivered" and r["kind"]==kind and r["num_agents"]==8 and r["handovers"]]
        axes[1].scatter([r["time"] for r in received],[r["age_seconds"] for r in received],s=5,label=kind)
    axes[1].set(xlabel="Workload simulated time (s)",ylabel="Message latency (s)"); axes[1].legend()
    for ax in axes: ax.grid(alpha=.2)
    figure(fig,output,"leo_queues_latency")
    # Library-produced orbits of satellites serving during a ten-minute window.
    window=np.arange(1000,1601,dtype=float)
    ids={int(x) for x in s["satellite_ids"][1000:1601] if x>=0}
    records=json.loads((PROJECT/"data/leo/starlink_omm.json").read_text())
    ts=load.timescale(builtin=True)
    start=datetime.fromisoformat(timeline.metadata["simulation_start_utc"]).astimezone(UTC)
    sky_times=ts.tt_jd(ts.from_datetime(start).tt+window/86400)
    satellites=[]
    for record in records:
        if int(record["NORAD_CAT_ID"]) in ids:
            xyz,_=EarthSatellite.from_omm(ts,record).at(sky_times).frame_xyz_and_velocity(itrs)
            satellites.append({"id":int(record["NORAD_CAT_ID"]),"xyz":np.round(xyz.km.T,2).tolist()})
    indices=window.astype(int)
    fields=("satellite_ids","elevation_degrees","range_km","uplink_doppler_hz","downlink_doppler_hz")
    state={name:np.round(s[name][indices],3).tolist() for name in fields}
    state["latency_ms"]=np.round(timeline.propagation[indices]*1000,3).tolist()
    payload={"satellites":satellites,"state":state,"events":[e for e in timeline.events if 1000<=e["start"]<=1600],"time_origin":1000}
    html=HTML.replace("__DATA__",json.dumps(payload,separators=(",",":")))
    (output/"leo_motion.html").write_text(html,encoding="utf-8")
    names=[("leo_handover_handshake","Attachments and control exchange"),
           ("leo_latency_components","Complete-message latency and components"),
           ("leo_queues_latency","Queues and delivery timelines"),
           ("leo_geometry_radio","Orbital geometry and ITU-R radio budget"),
           ("measured_one_way_delay","Independent Karlstad one-way delay measurements")]
    dashboard='<!doctype html><meta charset="utf-8"><title>LEO communication study</title><style>body{font:17px system-ui;max-width:1250px;margin:30px auto;padding:20px;color:#17354a}img{max-width:100%}iframe{width:100%;height:880px;border:0}small{color:#536977}a{color:#176cab}</style><h1>LEO communication simulation</h1><p>Historical orbital geometry and measured transport workload. Policy training runs are separate.</p><p><a href="leo_motion.html">Open interactive satellite view</a></p><iframe src="leo_motion.html"></iframe>'
    for name,label in names: dashboard+=f'<h2>{label}</h2><a href="{name}.png"><img src="{name}.png" alt="{label}"></a>'
    dashboard+='<small>CelesTrak OMM, Skyfield/SGP4, ITU-Rpy, SimPy; measurement-informed SATPIPE scheduling. RF allocations, gateway and control protocol remain stated simulation assumptions. No synthetic extension of measured results.</small>'
    (output/"index.html").write_text(dashboard,encoding="utf-8")
    write_json(output/"full_simulation_summary.json",{
        "transport_only":True,"delivery_statistics":report,"orbit":timeline.metadata,
        "mean_propagation_ground_ms":float(np.nanmean(timeline.propagation)*1000),
        "view":"library-propagated ECEF satellites; ten-minute historical window; no inferred handovers",
        "trace_comparison":"Karlstad packet-delay dataset plotted separately, not added to modeled latency"})
    print(json.dumps({"output":str(output),"delivery":report},indent=2))


HTML=r'''<!doctype html><html><head><meta charset="utf-8"><title>Mont-Tremblant LEO simulation</title><style>
body{font:16px system-ui;background:#0b1825;color:#e9f1f6;margin:25px}main{max-width:1100px;margin:auto}canvas{background:#102335;border-radius:12px;width:100%}button,input{margin:8px;font:inherit}#detail{line-height:1.8}small{color:#b4c5d2}</style></head><body><main><h1>Mont-Tremblant satellite route</h1><canvas id="view" width="1100" height="580"></canvas><div><button id="play">Play</button><button id="next">Next handshake</button><label>Speed <select id="speed"><option>1</option><option selected>5</option><option>15</option></select>×</label><input id="slider" type="range" min="0" max="600" step="0.01" value="0" style="width:60%"></div><div id="detail"></div><small>Historical CelesTrak OMM · Skyfield/SGP4 motion · ITU-Rpy atmospheric model. The attachment policy, handshake and framed round-robin scheduler are simulation assumptions. Orange = serving LEO; blue = other satellites that serve during this window. Ground projection is geocentric. This visualization does not report policy success.</small></main><script>
const data=__DATA__;const cv=document.getElementById('view'),ctx=cv.getContext('2d'),sl=document.getElementById('slider');let running=false,last=0;
const site=[46.2,-74.6];function world(lat,lon,r){let a=lat*Math.PI/180,b=lon*Math.PI/180;return[r*Math.cos(a)*Math.cos(b),r*Math.cos(a)*Math.sin(b),r*Math.sin(a)];}
function project(p){let lon=site[1]*Math.PI/180,lat=site[0]*Math.PI/180;let e=[-Math.sin(lon),Math.cos(lon),0],n=[-Math.sin(lat)*Math.cos(lon),-Math.sin(lat)*Math.sin(lon),Math.cos(lat)],u=world(...site,1);return[375+p.reduce((v,x,i)=>v+x*e[i],0)*.042,285-p.reduce((v,x,i)=>v+x*n[i],0)*.042,p.reduce((v,x,i)=>v+x*u[i],0)];}
function dot(p,color,r){let q=project(p);if(q[2]<0)return;ctx.beginPath();ctx.arc(q[0],q[1],r,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();}
function draw(){let sec=+sl.value,i=Math.min(599,Math.floor(sec)),fraction=sec-i;let id=data.state.satellite_ids[i];ctx.clearRect(0,0,1100,580);ctx.beginPath();ctx.arc(375,285,6371*.042,0,Math.PI*2);ctx.fillStyle='#174969';ctx.fill();ctx.strokeStyle='#406378';
for(let lat=-60;lat<=60;lat+=30){ctx.beginPath();let first=true;for(let lon=-180;lon<=180;lon+=2){let q=project(world(lat,lon,6371));if(q[2]<0){first=true;continue;}if(first){ctx.moveTo(q[0],q[1]);first=false;}else ctx.lineTo(q[0],q[1]);}ctx.stroke();}dot(world(...site,6371),'#6bea9a',6);
for(const satellite of data.satellites){let p=satellite.xyz[i].map((x,k)=>x+(satellite.xyz[i+1][k]-x)*fraction);dot(p,satellite.id===id?'#ffac50':'#70bdf9',satellite.id===id?7:3);if(satellite.id===id){let a=project(world(...site,6371)),b=project(p);ctx.strokeStyle='#ffac50';ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();}}
let now=sec+data.time_origin,event=data.events.find(e=>e.start<=now&&now<e.end);ctx.fillStyle='#e9f1f6';ctx.font='22px system-ui';ctx.fillText('Serving LEO '+id,750,85);ctx.font='17px system-ui';['Window time: '+sec.toFixed(2)+' s','Elevation: '+data.state.elevation_degrees[i].toFixed(1)+'°','Range: '+data.state.range_km[i].toFixed(1)+' km','One-way propagation + ground: '+data.state.latency_ms[i].toFixed(2)+' ms','Raw uplink Doppler: '+(data.state.uplink_doppler_hz[i]/1000).toFixed(1)+' kHz','Raw downlink Doppler: '+(data.state.downlink_doppler_hz[i]/1000).toFixed(1)+' kHz',event?'HANDSHAKE / RECONFIGURATION':'Backhaul available'].forEach((x,j)=>ctx.fillText(x,750,135+j*40));
document.getElementById('detail').textContent=event?event.control_messages.filter(x=>x.start<=now&&now<x.end).map(x=>'Control phase: '+x.kind).join(' · '):'Robots continue acting while complete messages and model snapshots travel.';}
sl.oninput=draw;document.getElementById('play').onclick=()=>{running=!running;document.getElementById('play').textContent=running?'Pause':'Play';};document.getElementById('next').onclick=()=>{running=false;let e=data.events.find(e=>e.start>+sl.value+1000+.05);if(e)sl.value=e.start-1000+.001;draw();};function tick(t){if(running&&last){sl.value=(+sl.value+(t-last)/1000*+document.getElementById('speed').value)%600;draw();}last=t;requestAnimationFrame(tick);}draw();requestAnimationFrame(tick);
</script></body></html>'''

if __name__=="__main__":main()
