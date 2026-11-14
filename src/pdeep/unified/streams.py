"""Pre-drawn continuous streams: NOT a finite bank sampled with replacement.

Every successful-update cursor/slot has its own original area/time/quantity draws.
Offline labels are evaluated once at those EXACT coordinates, with the FULL
CPU64 reference. Online regenerates the coordinates and independently adds the
original noise. Mathematically this is the same counter-keyed sampling law; no
trajectory interpolation, empirical query-bank recycling, or online PDE solve.
Only labels + recipes/checksums are stored, not all full trajectories/coordinates.
"""
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
import hashlib
import numpy as np
import torch
from ..core import digest,require
from ..artifact_io import read_payload,read_manifest,write_artifact
from ..smooth.bank import physical_design,grid
from ..smooth.definition import random_stream
from .config import SCHEMA
from .data import BalancedSampler,DEFAULT_COUNTS,OBSERVATION_COUNTS
from .reference import _space_payload,_sampling_space

STREAM_SCHEMA='G1-independent-continuous-encounters-v1'


def coordinates_id(x,t,q):
    h=hashlib.sha256()
    for a,dtype in ((x,np.float64),(t,np.float64),(q,np.int64)):
        a=np.ascontiguousarray(a,dtype=dtype);h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def encounters(config,rows,*,progress=None):
    lookup={tuple(r['key']):r['case_id'] for r in rows};output={r['case_id']:[] for r in rows}
    sampler=BalancedSampler(SimpleNamespace(data_seed=config.seed),SimpleNamespace(config=config),provider=object())
    for cursor in range(config.stream_updates):
        if progress and cursor%1000==0:progress('Dataset inspection',operation='planning independent continuous draws',planned_updates=cursor,total=config.stream_updates)
        for slot,key in enumerate(sampler.keys(cursor)):
            n=int(random_stream('unified-count',config.seed,cursor,slot).choice(OBSERVATION_COUNTS))
            output[lookup[key]].append((cursor,slot,n))
    return output


def designs(case,config,events):
    """Yield one design at a time, bounding host memory even for a long schedule."""
    seed=config.seed
    for cursor,slot,n in events:
        sid=(SCHEMA,'encounter',seed,cursor,slot)
        yield physical_design(case,n,observation=True,set_id=sid)
        yield physical_design(case,config.stream_queries,set_id=(SCHEMA,'supervised',seed,cursor,slot))
    # Bounded preflight on ANY physical case, at the configured count and128 sensors.
    if case.split=='train':
        sid=(SCHEMA,'encounter',seed,0,0)
        for n in set((*OBSERVATION_COUNTS,DEFAULT_COUNTS[case.pde])):yield physical_design(case,n,observation=True,set_id=sid)
        for n in {config.stream_queries,min(32,config.stream_queries)}:yield physical_design(case,n,set_id=(SCHEMA,'supervised',seed,0,0))
    # Fixed independent validation/evaluation designs. Training cases also receive
    # official sensors for optional inspection; their conditioning is never shared.
    sid=(SCHEMA,'fixed-evaluation',seed)
    for n in sorted(set((*OBSERVATION_COUNTS,DEFAULT_COUNTS[case.pde]))):yield physical_design(case,n,observation=True,set_id=sid)
    if case.split!='train':
        yield physical_design(case,config.stream_validation_queries,set_id=('unified-validation',seed))
        yield physical_design(case,128,observation=True,set_id=('unified-heldout-sensors',seed))
        yield physical_design(case,DEFAULT_COUNTS[case.pde],observation=True,set_id=(sid,'alternative-positions'))
        x,t,q,present=physical_design(case,DEFAULT_COUNTS[case.pde],observation=True,set_id=sid)
        yield x.astype(np.float32).astype(np.float64),t.astype(np.float32).astype(np.float64),q,present
        yield grid(case,config.stream_grid)


def prepare_labels(case,full,config,events,*,progress=None):
    output={}
    for i,design in enumerate(designs(case,config,events)):
        key=coordinates_id(*design[:3])
        if key in output:continue
        if progress and i%20==0:progress('Reference preparation',operation='direct continuous label evaluation',case=case.case_id,designs_completed=i,encounters=len(events))
        y=full.evaluate(*design[:3]);require(y.dtype==np.float64 and np.isfinite(y).all(),'reference labels must be finite CPU64')
        output[key]=torch.from_numpy(y.copy())
    return output


def save_sampling_space(full,root,execution_source):
    key='sampling-'+digest((full.reference.space.identity,execution_source));path=Path(root)/'spaces'/(key+'.json')
    if not path.exists():
        write_artifact(path.parent,key,{'artifact_type':'diagnostic_bundle','architecture_id':'REFERENCE_ONLY','run_id':'G1-unified-sampling',
            'family':full.case.family,'pde':'shared','execution_source':execution_source}, {'space':_space_payload(full.reference.space)},required=('space',),idempotent=True)
    return path


class PlannedReference:
    def __init__(self,dataset,*,max_cases=8):
        self.dataset=dataset;self.root=dataset.root;self.spaces={};self.loaded=OrderedDict();self.max_cases=max_cases
        from .reference import LinearCache
        self.optional=LinearCache(self.root/'responses',dataset.certificate,response_relative_tolerance=dataset.config.response_relative_tolerance)
        self.physics=self.optional.physics;self.execution_source=self.optional.execution_source
    def _payload(self,c):
        if c.case_id in self.loaded:self.loaded.move_to_end(c.case_id);return self.loaded[c.case_id]
        rel=self.dataset.index['records'].get(c.case_id);require(rel is not None,'unprepared or quarantined physical case: '+c.case_id)
        _,p=read_payload(self.root/rel,kind='diagnostic_bundle',required=('labels','qc','case'))
        self.loaded[c.case_id]=p
        while len(self.loaded)>self.max_cases:self.loaded.popitem(last=False)
        return p
    def evaluate(self,c,x,t,q,**kwargs):
        p=self._payload(c);key=coordinates_id(x,t,q)
        require(key in p['labels'],'Continuous draw not prepared for this declared schedule/evaluation design. Prepare the exact stream offline; no online PDE solve or recycled bank fallback is allowed.')
        y=p['labels'][key].numpy();require(len(y)==len(x),'label/coordinate size mismatch')
        return (y,np.zeros(len(y))) if kwargs.get('return_bound') else y
    def _get(self,c,*unused):
        if c.family not in self.spaces:
            payload=self._payload(c);path=self.root/payload['sampling_space'];r,p=read_payload(path,kind='diagnostic_bundle',required=('space',))
            require(r['execution_source']==self.execution_source,'sampling-space execution source changed')
            self.spaces[c.family]=_sampling_space(c.family,p['space'])
        return None
    def sensitivities(self,c,x,t,q):
        needed=[self.optional.asset_path(c,k,j) for k in c.fields for j in range(32)]
        if not all(p.is_file() for p in needed):raise BaselineUnavailable('Geometry-specific Gaussian control requires explicitly prepared verified unit responses; T1 D is not applicable to C3 geometry fields.')
        return self.optional.sensitivities(c,x,t,q)


class BaselineUnavailable(ValueError):pass
