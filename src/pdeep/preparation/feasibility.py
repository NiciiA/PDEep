"""Balanced feasibility view of the SAME cases and independent reference draws.

Eight TRAIN/two validation/two test per task. Logical visits consume unique
per-case draws from an explicit parent reference plan, never a recycled bank.
This permits byte-exact reuse of completed research labels without solving their
physical cases again. Config/plan identities distinguish the new experiment.
"""
from dataclasses import asdict,replace
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from ..core import require,digest
from ..artifact_io import read_payload,write_artifact,sha256
from ..artifact_tensors import tree_identity
from ..unified.config import DatasetConfig,PDES,FAMILIES
from ..unified.dataset import inventory
from ..unified.data import BalancedSampler
from .planning import encounters
from .portable import read_index

SCHEMA='G1-balanced-feasibility-independent-parent-draws-v1'

def configuration(*,seed=42):
    return DatasetConfig(preset='pilot',counts=(8,2,2),seed=seed,stream_updates=3000,stream_queries=4096)

def is_configuration(config):return config.identity==configuration(seed=config.seed).identity

def parent_configuration(config):
    return replace(config,preset='research',counts=(128,32,64),stream_updates=100000)

def source_identity():
    paths=('feasibility.py','feasibility_training.py','provider.py','coordinates.py')
    return digest({p:sha256(Path(__file__).with_name(p)) for p in paths})

def make_plan(config,rows,parent_config,parent_events):
    require(config.seed==parent_config.seed and config.pdes==parent_config.pdes and config.families==parent_config.families and config.stream_queries==parent_config.stream_queries,'parent draw policy differs')
    require(all(a<=b for a,b in zip(config.counts,parent_config.counts)),'feasibility cases not a split-wise subset')
    logical=encounters(config,rows);events={};by_visit={};usage={}
    for cid,visits in logical.items():
        donors=parent_events.get(cid,[]);require(len(donors)>=len(visits),'not enough distinct parent draws for '+cid+'; no recycling or automatic case re-solve')
        events[cid]=donors[:len(visits)];usage[cid]=len(visits)
        for visit,donor in zip(visits,events[cid]):by_visit[tuple(visit[:2])]=(cid,tuple(donor))
    mapping=np.empty((config.stream_updates,4,3),np.int64);ids=[]
    for cursor in range(config.stream_updates):
        row=[]
        for slot in range(4):cid,donor=by_visit[(cursor,slot)];mapping[cursor,slot]=donor;row.append(cid)
        require(len(set(row))==4,'feasibility effective batch repeats a physical case');ids.append(tuple(row))
    identity=digest((SCHEMA,config.identity,parent_config.identity,tree_identity(torch.from_numpy(mapping)),ids))
    return {'events':events,'mapping':torch.from_numpy(mapping),'case_ids':ids,'mapping_identity':identity,'parent_config':asdict(parent_config),'usage':usage}

def save_plan(config,rows,folder,*,parent_manifest=None,progress=None,parent_config=None):
    folder=Path(folder);path=folder/'reference_cache/feasibility-plan.json'
    if path.exists():
        r,p=read_payload(path,kind='diagnostic_bundle',required=('events','mapping','case_ids','mapping_identity','parent_config'))
        require(r['dataset_identity']==config.identity and r['schema']==SCHEMA,'feasibility plan differs');return p,path,True
    parent_config=parent_config or parent_configuration(config);donors=None
    if parent_manifest:
        source,index=read_index(parent_manifest);actual=DatasetConfig(**index['config'])
        derived=index.get('feasibility',{}).get('parent_dataset_identity')==parent_config.identity
        require(actual.identity==parent_config.identity or derived,'selected donor/reference-parent configuration differs')
        parent_path=source.parent/'reference_cache/encounter-plan.json'
        if parent_path.is_file() and not derived:
            r,p=read_payload(parent_path,kind='diagnostic_bundle',required=('events',));require(r['dataset_identity']==parent_config.identity,'parent plan mismatch');donors=p['events']
    if donors is None:donors=encounters(parent_config,inventory(parent_config),progress=progress)
    p=make_plan(config,rows,parent_config,donors)
    path=write_artifact(path.parent,path.stem,{'artifact_type':'diagnostic_bundle','architecture_id':'REFERENCE_ONLY','run_id':'G1-feasibility-plan','schema':SCHEMA,'dataset_identity':config.identity,'parent_dataset_identity':parent_config.identity},p,required=tuple(p))
    return p,path,False
