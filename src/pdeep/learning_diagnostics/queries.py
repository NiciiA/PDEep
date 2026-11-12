"""Counter-keyed continuous TRAIN queries; fixed sensors/evaluation never resample."""
import time
from dataclasses import replace
import numpy as np
import torch
from ..core import digest,require
from ..geometric.data import Tokens,Batch
from ..smooth.definition import random_stream
from ..smooth.data import tokens
from ..smooth.bank import physical_design
from .data import token_id
from .reference_cache import FixedReference

MODES=('fixed','resampled','dense_resampled')

def coordinates(case,count,seed,cursor,policy='stratified',*,revision='continuous-queries-v2'):
    require(count>=2 and policy in {'stratified','mixture'},'invalid continuous query count/time policy')
    # Same existing area-weighted sqrt-barycentric geometry sampler as sensors.
    mesh=replace(case,conditioning_h=.08).problem.mesh
    x=mesh.sample(count,random_stream('continuous-TRAIN-query-v2',case.case_id,seed,cursor,'position'))
    stream=random_stream('continuous-TRAIN-query-v2',case.case_id,seed,cursor,'time')
    if case.pde=='poisson':t=np.zeros(count)
    elif policy=='mixture':
        from ..geometric.data import times
        t=times(count,stream,training=True)
    else:
        n0=int(round(count*.125));ne=int(round(count*.125));t=stream.random(count)
        t[:n0]=0;t[n0:n0+ne]=stream.uniform(np.nextafter(0.,1.),.1,ne);stream.shuffle(t)
    q=np.arange(count,dtype=np.int64)%2 if case.pde=='wave' else np.zeros(count,dtype=np.int64)
    random_stream('continuous-TRAIN-query-v2',case.case_id,seed,cursor,'quantity').shuffle(q)
    if revision=='continuous-queries-v3' and policy=='stratified' and case.pde!='poisson':
        # Versioned diagnostic risk; v2 streams above remain byte-identical.
        # Guarantee coverage independently for EACH quantity, not just overall.
        stream=random_stream('continuous-TRAIN-query-v3',case.case_id,seed,cursor,'time')
        for quantity in np.unique(q):
            ids=np.flatnonzero(q==quantity);n=len(ids)
            require(n>=8,'v3 stratification requires >=8 queries per dynamic quantity')
            n0=max(1,n//8);ne=max(1,n//8);nm=max(1,n//4);nl=max(1,n//4)
            values=np.concatenate((np.zeros(n0),stream.uniform(np.nextafter(0.,1.),.1,ne),
                stream.uniform(np.nextafter(.1,1.),.5,nm),stream.uniform(np.nextafter(.5,1.),1.,nl),
                stream.random(n-n0-ne-nm-nl)))
            stream.shuffle(values);t[ids]=values
    return x,t,q,case.pde!='poisson'


class QuerySampler:
    def __init__(self,data,config,*,reference_cache=None):
        self.data=data;self.config=config;self.reference=reference_cache if reference_cache is not None else data.cached_reference;self.last={}
        data.assert_frozen();self.versions=self._versions();self.measure_transfer=False
        # Keep fixed inputs on the neural device once. Targets remain CPU64 until evaluated.
        self.observations=data.observations.to(config.device);self.fixed_queries=data.training.to(config.device)
        self.independent=data.diagnostic.to(config.device)
        self.case_id=data.case.case_id;self.observation_id=token_id(data.observations);self.evaluation_id=token_id(data.diagnostic)
        ev=data.diagnostic;rows=np.column_stack((ev.position[0].numpy(),ev.physical_time[0].numpy(),ev.quantity_id[0].numpy())).astype(np.float32)
        self.excluded={r.tobytes() for r in rows}
        self.device_versions=tuple((id(v),v._version) for t in (self.observations,self.fixed_queries,self.independent) for v in vars(t).values() if isinstance(v,torch.Tensor))
    def _versions(self):
        return tuple((id(v),v._version) for t in (self.data.observations,self.data.training,self.data.diagnostic) for v in vars(t).values() if isinstance(v,torch.Tensor))
    def _reference(self):
        if self.reference is None:self.reference=FixedReference(self.data.case)
        require(self.reference.case.case_id==self.case_id,'reference cache belongs to a different physical case')
        return self.reference
    def batch(self,cursor):
        current_device=tuple((id(v),v._version) for t in (self.observations,self.fixed_queries,self.independent) for v in vars(t).values() if isinstance(v,torch.Tensor))
        require(self._versions()==self.versions and current_device==self.device_versions and self.data.case.case_id==self.case_id,'frozen case/observation/evaluation tensors changed');c=self.config;start=time.perf_counter()
        if c.query_mode=='fixed':query=self.fixed_queries;prep=0.;cpu_id=self.data.identities['training']
        else:
            ref=self._reference();design=coordinates(self.data.case,c.query_count,c.query_sampler_seed,cursor,c.query_time_policy,revision=c.query_revision)
            # Excluding a finite coordinate set has zero continuous area/time
            # measure. Explicitly reject even rare float32 tuple coincidences.
            x,t,q,present=design
            for retry in range(8):
                keys=np.column_stack((x,t,q)).astype(np.float32)
                bad=np.array([r.tobytes() in self.excluded for r in keys])
                if not bad.any():break
                mesh=replace(self.data.case,conditioning_h=.08).problem.mesh
                x[bad]=mesh.sample(int(bad.sum()),random_stream('query-eval-exclusion',self.case_id,c.query_sampler_seed,cursor,retry))
            require(not bad.any(),'failed to exclude independent query tuples')
            begin=time.perf_counter();clean=ref.evaluate(*design[:3]);prep=time.perf_counter()-begin
            query,_=tokens(self.data.case,c.query_count,self.data.normalizers,set_id=cursor,coordinates=design,clean=clean)
            cpu_id=token_id(query)
        events=None
        if self.measure_transfer and torch.device(c.device).type=='cuda':
            torch.cuda.synchronize(c.device);events=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True));events[0].record()
        transfer=time.perf_counter();query=query.to(c.device);transfer=time.perf_counter()-transfer
        if events:events[1].record();events[1].synchronize()
        self.last={'query_mode':c.query_mode,'query_identity':cpu_id,'cursor':cursor,'physical_case':self.case_id,'observation':self.observation_id,'independent':self.evaluation_id,'reference_identity':None if self.reference is None else self.reference.identity,'reference_evaluation_seconds':prep,'reference_preparation_seconds':0. if self.reference is None else self.reference.preparation_seconds,'H2D_wall_seconds':transfer,'H2D_CUDA_event_seconds':None if events is None else events[0].elapsed_time(events[1])/1000,'data_wall_seconds':time.perf_counter()-start,'time_policy':'frozen original' if c.query_mode=='fixed' else c.query_time_policy}
        return Batch((self.data.case,),self.observations,query,self.data.normalizers),self.last
    def state_dict(self,cursor):
        return {'schema':'continuous-TRAIN-query-sampler-v2','cursor':cursor,'config':self.config.identity,'fixed_data':self.data.identity,'physical_case':self.case_id,'observations':self.observation_id,'independent':self.evaluation_id,'query_mode':self.config.query_mode,'query_count':self.config.query_count,'seed':self.config.query_sampler_seed,'time_policy':self.config.query_time_policy,'ordering':'counter keyed; no mutable random generator or finite point bank'}
