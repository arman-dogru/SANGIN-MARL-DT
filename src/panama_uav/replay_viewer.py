"""Offline browser replay of two recorded episodes; no browser dependency."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from .evaluation import case_bank, run_episode
from .models import choose_device
from .records import sha256, write_json
from .training import load_model


def create_replay(
    config,
    model_path,
    output,
    agents=2,
    delay=10,
    route="cloud",
    diagnostic=False,
    device_name="auto",
):
    device = choose_device(device_name)
    import torch

    torch.set_num_threads(1)
    model, payload = load_model(
        model_path, config, device, require_qualified=not diagnostic
    )
    case = case_bank(config, 1, (agents,))[0]
    configs = [
        replace(
            config,
            communication=replace(
                config.communication, profile="perfect", stale_information=()
            ),
        ),
        replace(
            config,
            communication=replace(
                config.communication,
                profile="physical",
                mission_route=route,
                stale_information=("task", "intentions", "priorities"),
                delay_steps=delay,
                handover_enabled=True,
                contention_enabled=True,
            ),
        ),
    ]
    traces = []
    for variant in configs:
        row, trace = run_episode(
            variant, model, case, device, payload.get("learner_updates", 0)
        )
        trace["result"] = row
        traces.append(trace)
    data = {
        "label": "Unqualified smoke model · transport demonstration"
        if diagnostic
        else "Qualified frozen clean policy",
        "route": route,
        "delay": delay,
        "model_sha256": sha256(model_path),
        "traces": traces,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output.with_suffix(".json"), data)
    encoded = json.dumps(data, separators=(",", ":"), allow_nan=False).replace(
        "</", "<\\/"
    )
    html = HTML.replace("__DATA__", encoded)
    output.write_text(html, encoding="utf-8")
    return output


HTML = r"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>UAV–PANAMA communication replay</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#111821;color:#e2eaf3;font:15px system-ui}main{max-width:1300px;margin:auto;padding:24px}
h1{font-size:25px;margin:0 0 8px}p{color:#a9b9ca}button,select{background:#253648;color:inherit;border:1px solid #52657a;border-radius:6px;padding:8px}
.controls{display:flex;gap:14px;align-items:center;margin:20px 0}input{flex:1}.panels{display:grid;grid-template-columns:1fr 1fr;gap:24px}
.panel{background:#1a2633;padding:16px;border-radius:12px}canvas{width:100%;aspect-ratio:1;background:#0d131b}
h2{font-size:18px}pre{white-space:pre-wrap;font-size:12px;max-height:220px;overflow:auto}table{width:100%;font-size:12px;border-collapse:collapse}
td,th{padding:7px;text-align:left;border-bottom:1px solid #384656}small{color:#afbbca}#timeline{aspect-ratio:8; margin-top:20px}
@media(max-width:750px){.panels{grid-template-columns:1fr}.controls{flex-wrap:wrap}}
</style><main>
<h1>Fresh and stale mission information</h1><p id="description"></p>
<div class="controls"><button id="play">Play</button><button id="reset">Reset</button><input id="step" type="range" min="0" value="0"><output id="clock"></output><label>Inspect agent <select id="agent"></select></label></div>
<div class="panels">
<section class="panel"><h2>Perfect communication</h2><canvas id="fresh" width="640" height="640"></canvas><div id="freshStats"></div></section>
<section class="panel"><h2>Satellite handover + resource contention</h2><canvas id="stale" width="640" height="640"></canvas><div id="staleStats"></div></section>
</div>
<canvas id="timeline" width="1200" height="150"></canvas>
<p><small>Solid circles: robots. Squares: known goals. Dashed line: inspected robot's cached absolute A* guidance. Cyan circle: UAV ground coverage at its recorded altitude. Red timeline marks: configured satellite reassignment/handshake interruption; yellow: scheduled reservation windows. Robots act continuously. This replay records neural policy actions; no stopping or recovery controller is added.</small></p>
</main><script>
const data=__DATA__;
const traces=data.traces,slider=document.querySelector('#step'),agentSelect=document.querySelector('#agent');
let tick=0,playing=false;
const colors=['#ef746a','#e1b448','#8baafa','#b486de','#6bc492','#ef8cbe','#9ccddd','#ddba94'];
document.querySelector('#description').textContent=data.label+' · '+data.route+' mission route · additional contention delay '+data.delay+' steps';
slider.max=Math.max(...traces.map(t=>t.frames.length))-1;
for(let a=0;a<traces[0].goals.length;a++){let o=document.createElement('option');o.value=a;o.textContent=a;agentSelect.append(o)}
function radius(height){const a=9.61,b=.16;for(let r=0;r<300;r+=.1){let alpha=height/Math.max(r,.001),p=1/(1+a*Math.exp(-b*(Math.atan(alpha)*180/Math.PI-a)));let loss=20*Math.log10(4*Math.PI*2.5e9/3e8)+20*Math.log10(Math.hypot(r,height))+p+(1-p)*20;if(loss>85)return r}return 300}
function draw(trace,name){let f=trace.frames[Math.min(tick,trace.frames.length-1)],a=+agentSelect.value,ctx=document.querySelector('#'+name).getContext('2d'),n=trace.grid.length,s=640/n;
ctx.clearRect(0,0,640,640);for(let r=0;r<n;r++)for(let c=0;c<n;c++){ctx.fillStyle=trace.grid[r][c]?'#42505e':'#13202a';ctx.fillRect(c*s,r*s,s-1,s-1)}
let u=f.uav_position;ctx.beginPath();ctx.arc(u[0]*s,u[1]*s,radius(u[2])*s,0,Math.PI*2);ctx.fillStyle='#59dbe314';ctx.fill();ctx.strokeStyle='#59dbe3';ctx.lineWidth=2;ctx.stroke();
ctx.fillStyle='#59dbe3';ctx.fillRect(u[0]*s-5,u[1]*s-5,10,10);
trace.goals.forEach((g,i)=>{ctx.strokeStyle=colors[i%8];ctx.lineWidth=2;ctx.strokeRect(g[1]*s+3,g[0]*s+3,s-6,s-6)});
let path=f.cached_guidance[a]||[];ctx.beginPath();ctx.moveTo((f.positions[a][1]+.5)*s,(f.positions[a][0]+.5)*s);path.forEach(p=>ctx.lineTo((p[1]+.5)*s,(p[0]+.5)*s));ctx.strokeStyle=colors[a%8];ctx.setLineDash([5,4]);ctx.stroke();ctx.setLineDash([]);
f.positions.forEach((p,i)=>{ctx.beginPath();ctx.arc((p[1]+.5)*s,(p[0]+.5)*s,s*.35,0,Math.PI*2);ctx.fillStyle=colors[i%8];ctx.fill();ctx.fillStyle='#111';ctx.font='bold 12px system-ui';ctx.fillText(i,(p[1]+.3)*s,(p[0]+.7)*s)});
ctx.strokeStyle='#59dbe3';ctx.lineWidth=2;ctx.beginPath();ctx.arc(u[0]*s,u[1]*s,12,0,Math.PI*2);ctx.moveTo(u[0]*s-18,u[1]*s);ctx.lineTo(u[0]*s+18,u[1]*s);ctx.moveTo(u[0]*s,u[1]*s-18);ctx.lineTo(u[0]*s,u[1]*s+18);ctx.stroke();
const ages=f.information_age[a],handover=trace.handover_windows.some(w=>w[0]<=f.time&&f.time<w[1]);
document.querySelector('#'+name+'Stats').innerHTML='<p>'+f.agents_at_goal+'/'+trace.goals.length+' at goals · UAV altitude '+u[2].toFixed(1)+' m · travel '+f.uav_travel_m.toFixed(1)+' m</p><table><tr><th>Cached input</th><th>Age (seconds)</th></tr>'+Object.entries(ages).map(([k,v])=>'<tr><td>'+k+'</td><td>'+v.toFixed(3)+'</td></tr>').join('')+'<tr><td>Deployed frozen model version</td><td>'+f.model_versions[a]+'</td></tr><tr><td>UAV position report age</td><td>'+f.uav_reports[a].age_seconds.toFixed(3)+'</td></tr></table><pre>Cached next waypoints: '+JSON.stringify(path.slice(0,5))+'\nSatellite handshake active at sampled time: '+handover+'\nFinal recorded success: '+trace.result.success+' · PAR-2 '+trace.result.par2+'</pre>';}
function timeline(){let t=traces[1],ctx=document.querySelector('#timeline').getContext('2d'),end=+slider.max||1;ctx.clearRect(0,0,1200,150);ctx.fillStyle='#152332';ctx.fillRect(0,0,1200,150);
for(let [s,e] of t.handover_windows){ctx.fillStyle='#f66d67';ctx.fillRect(s/end*1200,0,Math.max(2,(e-s)/end*1200),30)}
for(let w of t.transport_events.filter(e=>e.event==='scheduled_reservation'&&e.agent===+agentSelect.value)){ctx.fillStyle='#e1b448';ctx.fillRect(w.start/end*1200,30,(w.end-w.start)/end*1200,30)}
for(let [key,color] of [['task','#ef746a'],['intentions','#8baafa'],['priorities','#6bc492']]){ctx.strokeStyle=color;ctx.beginPath();t.frames.forEach((f,i)=>{let y=145-Math.min(80,f.information_age[agentSelect.value][key]*3);i?ctx.lineTo(f.time/end*1200,y):ctx.moveTo(0,y)});ctx.stroke()}
ctx.strokeStyle='#fff';ctx.beginPath();ctx.moveTo(tick/end*1200,0);ctx.lineTo(tick/end*1200,150);ctx.stroke();ctx.fillStyle='#eee';ctx.font='12px system-ui';ctx.fillText('Handover | reservations | information age curves (task red, intentions blue, priorities green)',10,18)}
function render(){slider.value=tick;document.querySelector('#clock').textContent='Step '+tick+' / '+slider.max;draw(traces[0],'fresh');draw(traces[1],'stale');timeline()}
slider.oninput=()=>{tick=+slider.value;render()};agentSelect.onchange=render;
document.querySelector('#play').onclick=()=>{playing=!playing;document.querySelector('#play').textContent=playing?'Pause':'Play'};
document.querySelector('#reset').onclick=()=>{tick=0;playing=false;document.querySelector('#play').textContent='Play';render()};
setInterval(()=>{if(playing){tick=Math.min(tick+1,+slider.max);if(tick===+slider.max){playing=false;document.querySelector('#play').textContent='Play'}render()}},120);
render();
</script></html>"""
