"""Same independent encounter RNG calls; cache task permutations and saved plan."""
from pathlib import Path
from types import SimpleNamespace
from ..core import require,digest
from ..artifact_io import read_payload,write_artifact
from ..smooth.definition import random_stream
from ..unified.config import case_index
from ..unified.data import OBSERVATION_COUNTS

def encounters(config,rows,*,progress=None):
    lookup={tuple(r['key']):r['case_id'] for r in rows};out={r['case_id']:[] for r in rows};tasks=config.training_tasks;cycle_cache={}
    for cursor in range(config.stream_updates):
        if progress and cursor%1000==0:progress('Dataset planning',planned_updates=cursor,total=config.stream_updates)
        used=set()
        for slot in range(4):
            ordinal=4*cursor+slot;cycle,offset=divmod(ordinal,len(tasks))
            if cycle not in cycle_cache:cycle_cache={cycle:random_stream('unified-task-order',config.seed,cycle).permutation(len(tasks))}
            p,g=tasks[int(cycle_cache[cycle][offset])];i=int(random_stream('unified-physical-index',config.seed,ordinal,p,g).integers(config.counts[0]))
            for _ in range(config.counts[0]):
                key=(p,g,'train',case_index(g,i))
                if key not in used:break
                i=(i+1)%config.counts[0]
            require(key not in used,'cannot form four independent physical cases');used.add(key)
            n=int(random_stream('unified-count',config.seed,cursor,slot).choice(OBSERVATION_COUNTS));out[lookup[key]].append((cursor,slot,n))
    return out

def saved_plan(config,rows,folder,*,progress=None):
    path=Path(folder)/'reference_cache/encounter-plan.json'
    if path.exists():
        r,p=read_payload(path,kind='diagnostic_bundle',required=('events',));require(r['dataset_identity']==config.identity and r['inventory_identity']==digest(rows),'encounter plan mismatch');return p['events'],path,True
    events=encounters(config,rows,progress=progress)
    path=write_artifact(path.parent,path.stem,{'artifact_type':'diagnostic_bundle','architecture_id':'REFERENCE_ONLY','run_id':'G1-encounter-plan','dataset_identity':config.identity,'inventory_identity':digest(rows)}, {'events':events},required=('events',))
    return events,path,False
