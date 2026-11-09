"""Immutable matched T1 evaluation recipes. No model or hidden targets in a plan."""
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import numpy as np
import torch
from .core import decode, digest, require, require_immutable
from .data import (CaseDataset, CaseKey, EVALUATION_TIMES, VERSION, observations, persist_json)
from .batches import QueryBatch

METRICS = ("physical_MSE=mean(e^2)", "physical_RMSE=sqrt(MSE)", "physical_MAE=mean(abs(e))",
           "standardized_MSE=mean((e/training_RMS)^2)",
           "reduce:points->equal_quantities/case->equal_cases/PDE->equal_PDEs",
           "paired_intervals:resample_physical_cases_within_PDE;never_points",
           "coverage:50/90/95;latent_excludes_noise;measurement_includes_sigma^2")

@dataclass(frozen=True)
class EvaluationCase:
    pde: str
    case_id: int
    physical_id: str
    case_identity: str
    conditioning_set: str
    conditioning_identity: str
    heldout_set: str
    heldout_identity: str

@dataclass(frozen=True)
class EvaluationPlan:
    version: str
    benchmark: str
    inverse_track: str
    dataset_version: str
    dataset_identity: str
    root_seed: int
    split: str
    pdes: tuple[str, ...]
    cases: tuple[EvaluationCase, ...]
    spatial_points: tuple[tuple[float, float], ...]
    dynamic_times: tuple[float, ...]
    quantities: tuple[tuple[str, tuple[int, ...]], ...]
    diagnostic_points: tuple[tuple[float, float, float], ...]
    metrics: tuple[str, ...]
    normalizer_identity: str
    normalizer_version: str
    reference_precision: str
    evaluation_seed: int
    source_identity: str
    config_identity: str

    @property
    def identity(self): return digest(asdict(self))

    def validate(self, *, allow_final_test=False):
        require_immutable(self)
        require(self.version == 'matched-evaluation-v1' and self.benchmark == 'unit_square_modal_v1'
                and self.inverse_track == 'T1' and self.dataset_version == VERSION, 'unsupported evaluation benchmark')
        require(self.split == 'validation' or (self.split == 'test' and allow_final_test),
                'test evaluation requires explicit final-evaluation authorization')
        require(len(set(self.pdes)) == len(self.pdes) and set(self.pdes) <= {'poisson','heat','wave'} and self.pdes, 'invalid PDE selection')
        require(self.dynamic_times == EVALUATION_TIMES and self.reference_precision == 'CPU float64', 'canonical reference precision/times required')
        axis = np.linspace(0.,1.,64)
        require(self.spatial_points == tuple((float(x),float(y)) for x in axis for y in axis), 'canonical 64x64 grid required')
        require(self.quantities == tuple((p,(0,1) if p=='wave' else (0,)) for p in self.pdes), 'invalid quantity identities')
        require(self.normalizer_version == VERSION and self.metrics == METRICS and len(self.diagnostic_points) >= 8 and all(0<x<1 and 0<y<1 and 0<t<1 for x,y,t in self.diagnostic_points), 'invalid diagnostic/metric definitions')
        require(bool(self.source_identity) and bool(self.config_identity), 'source/config identity required')
        require({c.pde for c in self.cases} == set(self.pdes) and len({c.physical_id for c in self.cases}) == len(self.cases), 'duplicate or missing physical cases')
        for c in self.cases:
            require(CaseKey(self.split,c.pde,c.case_id).physical_id == c.physical_id and c.conditioning_set != c.heldout_set
                    and c.conditioning_identity != c.heldout_identity, 'case split or independent observation identity mismatch')
        require(CaseDataset(self.root_seed).identity == self.dataset_identity, 'dataset identity mismatch')
        return self

    def save(self, directory):
        return persist_json(Path(directory)/f'{self.identity}.json', {'identity':self.identity,'plan':asdict(self)})

    @classmethod
    def load(cls, path, *, allow_final_test=False):
        record = json.loads(Path(path).read_text()); plan = decode(cls,record['plan'])
        require(plan.identity == record['identity'], 'evaluation plan checksum mismatch')
        return plan.validate(allow_final_test=allow_final_test)

    def query(self, pde, quantity=0):
        require((pde, (0,1) if pde=='wave' else (0,)) in self.quantities and quantity in dict(self.quantities)[pde], 'unsupported query family/quantity')
        p=torch.tensor(self.spatial_points,dtype=torch.float64)
        if pde == 'poisson':
            p=p[None]; t=torch.zeros(1,len(self.spatial_points),1,dtype=torch.float64)
        else:
            p=p.repeat(len(self.dynamic_times),1)[None]
            t=torch.tensor(self.dynamic_times,dtype=torch.float64).repeat_interleave(len(self.spatial_points))[None,:,None]
        valid=torch.ones(p.shape[:2],dtype=torch.bool)
        return QueryBatch(p,t,valid & (pde!='poisson'),torch.full_like(valid,quantity,dtype=torch.int64),valid)

    def regenerate(self, entry, normalizers):
        require(entry in self.cases and normalizers.identity == self.normalizer_identity, 'plan/case/normalizer mismatch')
        ds=CaseDataset(self.root_seed); case=ds.case(self.split,entry.pde,entry.case_id)
        cond=observations(ds,case,normalizers,set_id=entry.conditioning_set)
        held=observations(ds,case,normalizers,set_id=entry.heldout_set)
        require((case.identity,cond.metadata.identity,held.metadata.identity)==
                (entry.case_identity,entry.conditioning_identity,entry.heldout_identity), 'regenerated plan identity mismatch')
        return case,cond,held


def make_plan(normalizers, *, source_id, config_id, case_ids=(0,), pdes=('poisson','heat','wave'),
              split='validation', seed=20261109, allow_final_test=False):
    require(split=='validation' or (split=='test' and allow_final_test), 'explicit final test authorization required')
    ds=CaseDataset(normalizers.root_seed); entries=[]
    for pde in pdes:
        for i in case_ids:
            case=ds.case(split,pde,i); name=f'eval-v1/{seed}'
            cond=observations(ds,case,normalizers,set_id=name+'/conditioning')
            held=observations(ds,case,normalizers,set_id=name+'/heldout')
            entries.append(EvaluationCase(pde,i,case.key.physical_id,case.identity,cond.metadata.set_id,
                                         cond.metadata.identity,held.metadata.set_id,held.metadata.identity))
    axis=np.linspace(0.,1.,64); rng=np.random.default_rng(seed)
    diag=tuple(tuple(float(v) for v in row) for row in rng.uniform(.003,.997,(24,3)))
    return EvaluationPlan('matched-evaluation-v1','unit_square_modal_v1','T1',VERSION,ds.identity,ds.root_seed,
                          split,tuple(pdes),tuple(entries),tuple((float(x),float(y)) for x in axis for y in axis),
                          EVALUATION_TIMES,tuple((p,(0,1) if p=='wave' else (0,)) for p in pdes),diag,METRICS,
                          normalizers.identity,normalizers.version,'CPU float64',seed,source_id,config_id).validate(allow_final_test=allow_final_test)
