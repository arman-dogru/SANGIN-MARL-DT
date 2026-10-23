"""Disjoint procedural MAPF maps/cases; stress-pilot maps are evaluation-only."""
from collections import deque
import hashlib
import json
from pathlib import Path
import numpy as np

from .environment import astar_path
from .records import sha256, write_json
from .scenarios import load_map, Scenario
from .controlled_scenarios import geometry, FAMILIES as STRESS_FAMILIES

FAMILIES = ('random_obstacles','intersection','corridor','bottleneck','warehouse')
SPLITS = ('train','validation','test')


def canonical_map(grid):
    cells=np.argwhere(np.asarray(grid)==0)
    lo=cells.min(0); hi=cells.max(0)+1
    cropped=np.asarray(grid,dtype=np.uint8)[lo[0]:hi[0],lo[1]:hi[1]]
    encodings=[]
    for flipped in (cropped,np.fliplr(cropped)):
        for k in range(4):
            a=np.rot90(flipped,k)
            encodings.append(str(a.shape).encode()+a.tobytes())
    return hashlib.sha256(min(encodings)).hexdigest()


def connected(grid):
    cells=set(map(tuple,np.argwhere(grid==0)))
    largest=set()
    while cells:
        start=cells.pop(); component={start}; queue=deque([start])
        while queue:
            r,c=queue.popleft()
            for point in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                if point in cells:
                    cells.remove(point);component.add(point);queue.append(point)
        if len(component)>len(largest):largest=component
    result=np.ones_like(grid)
    for p in largest:result[p]=0
    return result


def generate_map(family,seed):
    rng=np.random.default_rng(seed); g=np.ones((32,32),np.uint8)
    center=int(rng.integers(13,19)); width=int(rng.choice([2,3,4]))
    if family=='random_obstacles':
        g[2:30,2:30]=(rng.random((28,28))<rng.uniform(.08,.22)).astype(np.uint8)
    elif family=='intersection':
        g[center:center+width,3:29]=0;g[3:29,center:center+width]=0
        for _ in range(12):
            r,c=map(int,rng.integers(4,27,2));g[r:r+int(rng.integers(2,5)),c:c+int(rng.integers(2,5))]=0
    elif family=='corridor':
        left=int(rng.integers(9,13));right=int(rng.integers(19,23))
        g[9:25,3:left]=0;g[8:26,right:29]=0;g[center:center+int(rng.choice([1,2,3])),3:29]=0
        for c in range(left,right,2):
            g[center-int(rng.integers(1,4)):center+int(rng.integers(2,5)),c:c+int(rng.integers(1,3))]=0
    elif family=='bottleneck':
        g[3:29,3:29]=0;g[3:29,center:center+2]=1
        door=int(rng.integers(12,20));g[door:door+int(rng.integers(1,4)),center:center+2]=0
        for _ in range(12):
            r,c=map(int,rng.integers(4,28,2))
            if abs(c-center)>3 and abs(r-door)>3:g[r,c]=1
    elif family=='warehouse':
        g[2:30,2:30]=0
        for r in range(4,27,int(rng.integers(4,7))):
            for c in range(4,27,int(rng.integers(4,7))):
                g[r:r+int(rng.integers(1,3)),c:c+int(rng.integers(1,4))]=1
    else:raise ValueError('Unknown MAPF family')
    return connected(np.rot90(g,int(rng.integers(4))))


def path_conflicts(grid,starts,goals,horizon=5,radius=7,paths=None):
    paths=paths if paths is not None else [astar_path(grid,tuple(a),tuple(b)) for a,b in zip(starts,goals)]
    if any(not p for p in paths):return 0
    paths=[(p+[p[-1]]*(horizon+1))[:horizon+1] for p in paths]
    count=0
    for i,a in enumerate(paths):
        for j,b in enumerate(paths):
            if i==j:continue
            visible=lambda p:max(abs(p[0]-starts[i][0]),abs(p[1]-starts[i][1]))<=radius
            vertex=any(a[k]==b[k] and visible(b[k]) for k in range(horizon+1))
            edge=any(a[k]==b[k+1] and a[k+1]==b[k] and a[k]!=a[k+1] and visible(b[k]) and visible(b[k+1]) for k in range(horizon))
            count+=bool(vertex or edge)
    return count


def case_for(grid,map_record,count,seed,coordinated=True):
    rng=np.random.default_rng(seed);free=np.argwhere(grid==0)
    center=np.array([16,16]);near=free[np.max(np.abs(free-center),axis=1)<=8]
    if len(near)<2*count:near=free
    for attempt in range(1000):
        cells=near if coordinated else free
        chosen=rng.choice(len(cells),2*count,replace=False)
        starts=cells[chosen[:count]].tolist();goals=cells[chosen[count:]].tolist()
        if coordinated:
            # Two opposing tasks create an actual timed vertex/edge opportunity.
            first=tuple(starts[0]); candidates=[tuple(map(int,p)) for p in near if 4<=sum(abs(int(p[k])-first[k]) for k in (0,1))<=9]
            if not candidates:continue
            second=candidates[int(rng.integers(len(candidates)))]
            starts[1]=list(second);goals[0]=list(second);goals[1]=list(first)
        if len(set(map(tuple,starts)))!=count or len(set(map(tuple,goals)))!=count:continue
        paths=[astar_path(grid,tuple(a),tuple(b)) for a,b in zip(starts,goals)]
        if any(not p or len(p)<3 or len(p)>55 for p in paths):continue
        exposure=path_conflicts(grid,starts,goals)
        if coordinated and not exposure:continue
        return dict(case_id=f"{map_record['map_id']}-n{count}-seed{seed}",map_id=map_record['map_id'],
            family=map_record['family'],split=map_record['split'],map_path=map_record['path'],
            map_canonical=map_record['canonical'],num_agents=count,seed=int(seed),starts=starts,goals=goals,
            designed_coordination=coordinated,initial_relevant_pairs=exposure)
    raise ValueError(f'Could not construct reachable MAPF case for {map_record["map_id"]}')


def prepare_dataset(root,seed=42,train_maps=20,eval_maps=5,eval_cases=40,*,excluded_canonicals=(),
                    populations=(2,4,6,8)):
    populations=tuple(populations)
    if (not populations or populations!=tuple(sorted(set(populations))) or
        any(type(n) is not int or n<2 for n in populations)):
        raise ValueError('Distinct increasing integer populations >=2 required')
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    reserved={canonical_map(geometry(f)[0]) for f in STRESS_FAMILIES}
    seen=set(reserved)|set(excluded_canonicals);maps=[];banks={s:[] for s in SPLITS}
    for split_index,split in enumerate(SPLITS):
        for family_index,family in enumerate(FAMILIES):
            for index in range(train_maps if split=='train' else eval_maps):
                map_seed=seed+100_000_000*(split_index+1)+family_index*1_000_000+index*100
                for retry in range(100):
                    grid=generate_map(family,map_seed+retry);canonical=canonical_map(grid)
                    if canonical not in seen and np.count_nonzero(grid==0)>100:break
                else:raise ValueError('Could not generate independent map')
                seen.add(canonical)
                path=root/f'{split}-{family}-{index:03d}.map'
                path.write_text('type octile\nheight 32\nwidth 32\nmap\n'+'\n'.join(''.join('@' if x else '.' for x in row) for row in grid)+'\n',encoding='utf-8')
                record=dict(map_id=f'{split}-{family}-{index:03d}',family=family,split=split,
                    seed=map_seed+retry,path=str(path.resolve()),sha256=sha256(path),canonical=canonical)
                maps.append(record)
            if split!='train':
                options=[r for r in maps if r['split']==split and r['family']==family]
                for count in populations:
                    for case in range(eval_cases):
                        record=options[case%len(options)];grid=load_map(record['path'])
                        value=case_for(grid,record,count,
                            seed+500_000_000*(split_index+1)+family_index*1_000_000+count*10000+case,
                            coordinated=case<eval_cases//2)
                        value['evaluation_index']=case
                        banks[split].append(value)
    dataset=dict(schema='clean_mapf_split_v1',seed=seed,maps=maps,
        reserved_stress_maps=sorted(reserved),banks=banks,
        exclusion='disjoint map content under cropping/translation/rotation/reflection; stress cases never enter training')
    if populations!=(2,4,6,8):
        dataset['populations']=list(populations)
    write_json(root/'dataset.json',dataset)
    verify_dataset(dataset)
    return dataset


def verify_dataset(dataset):
    seen=set(dataset['reserved_stress_maps'])
    map_by_id={}
    for record in dataset['maps']:
        grid=load_map(record['path'])
        if sha256(record['path'])!=record['sha256'] or canonical_map(grid)!=record['canonical'] or record['canonical'] in seen:
            raise ValueError('MAPF map leakage or checksum mismatch')
        seen.add(record['canonical']);map_by_id[record['map_id']]=record
    case_ids=set()
    for split,bank in dataset['banks'].items():
        for case in bank:
            if case['split']!=split or map_by_id[case['map_id']]['split']!=split or case['case_id'] in case_ids:
                raise ValueError('MAPF scenario split leakage')
            case_ids.add(case['case_id'])
    return True


def scenario_for(case):
    return Scenario(Path(case['case_id']),Path(case['map_path']),tuple(map(tuple,case['starts'])),tuple(map(tuple,case['goals'])))
