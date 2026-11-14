"""Composition of established objectives, with per-case wave-only kinematics."""
import torch
from ..core import require
from ..train_types import LossReport
from ..geometric.physics import WeakConfig
from ..learning_diagnostics.query_loss import QueryAdapter
from ..learning_diagnostics.runtime import InputCache
from .config import MODELS


class UnifiedAdapter:
    def __init__(self,config):
        self.config=config;self.cursor=0;self.last={};self.cache=InputCache();self.amp='off';self.window=[];self.measure_gradients=False
        self.weak=WeakConfig(config.spatial_order,config.time_order,config.weak_tests,config.physics_chunk,
                             weak_weight=.001,initial_weight=.001,boundary_weight=.001,auxiliary_weight=.1)
    def loss(self,model,batch,*,detailed=False):
        reports=[];gradients=[]
        for row,case in enumerate(batch.cases):
            kin=self.config.model_key=='A_GEO/KIN_001' and case.pde=='wave'
            adapter=QueryAdapter(self.config.architecture,self.weak,chunk=self.config.query_chunk,checkpointing=True,
                                 field_only=self.config.model_key!='C_GEO/ALL_001',amp='off',cache=self.cache,
                                 kinematic_weight=.001 if kin else 0.,kinematic_queries=512,
                                 kinematic_seed=self.config.data_seed,kinematic_cursor=self.cursor)
            report=adapter.loss(model,batch.slice(row,row+1),detailed=detailed)
            if case.pde=='poisson':
                # The established weak function returns a zero placeholder. Physically absent IC is N/A.
                report.unweighted.pop('initial',None);report.weighted.pop('initial',None)
                report.total=sum(report.weighted.values())
            if detailed or self.measure_gradients:
                gg=torch.autograd.grad(report.total,tuple(model.parameters()),allow_unused=True,retain_graph=True)
                gradients.append({'pde':case.pde,'case_id':case.case_id,'unclipped_case_gradient_norm':sum(float(g.detach().float().square().sum()) for g in gg if g is not None)**.5})
            reports.append(report)
        n=len(reports);keys=set().union(*(r.unweighted for r in reports));zero=reports[0].total.new_zeros(())
        raw={k:sum((r.unweighted.get(k,zero) for r in reports),zero)/n for k in keys}
        weighted={k:sum((r.weighted.get(k,zero) for r in reports),zero)/n for k in keys}
        self.last={'per_case':[{'pde':c.pde,'geometry':c.family,'case_id':c.case_id,'active_terms':list(r.weighted),'kinematic_queries':r.diagnostics.get('kinematic',[]),
                                'unweighted':{k:float(v.detach()) for k,v in r.unweighted.items()},
                                'weighted':{k:float(v.detach()) for k,v in r.weighted.items()}} for c,r in zip(batch.cases,reports)],
                   'case_gradient_norms':gradients,'amp':'FP32 (auto resolves off; no unverified T4 BF16/FP16 path)',
                   'weighting':'equal physical cases; equal active quantities within each case; absent terms contribute no loss'}
        self.window.append(self.last)
        self.window=self.window[-4:]
        return LossReport(sum(weighted.values()),raw,weighted,{k:'established loss, equal case reduction' for k in raw},
                          {k:sum(r.counts.get(k,0) for r in reports) for k in ('cases','valid_queries','valid_observations','active_coefficients')},self.last)
