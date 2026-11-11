"""Exactly TWO extra bounded updates per new model; total Phase G cap = FIVE.

The initial LR3e-4 smoke increased fixed-case loss. This independent LR1e-6
control checks short-horizon descent, NOT convergence or successful full training.
"""
import json,time,gc
from pathlib import Path
import torch
from pdeep.geometric.data import Normalizers,physical_case,batch
from pdeep.geometric.models import create
from pdeep.geometric.physics import Adapter,WeakConfig
from pdeep.core import require

if __name__=='__main__':
    path=Path('reports/geometry_phase/models-cpu.json');result=json.loads(path.read_text());n=Normalizers.fromdict(json.loads(Path('reports/geometry_phase/normalizers.json').read_text()))
    b=batch([physical_case('train','heat',0,h=.16,dt=.02)],n,sensors=8,queries=8);weak=WeakConfig(2,2,4,256)
    for name in ('A_GEO','C_GEO'):
        require('short_descent_control' not in result['architectures'][name],'bounded control already ran; refuse to exceed update cap')
        m=create(name,n,'medium',seed=903);optimizer=torch.optim.AdamW(m.parameters(),lr=1e-6,weight_decay=1e-4);adapter=Adapter(name,weak);losses=[];start=time.perf_counter()
        for step in range(3):
            report=adapter.loss(m,b);losses.append(float(report.total.detach()))
            if step<2:
                optimizer.zero_grad(set_to_none=True);report.total.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        require(losses[-1]<losses[0],'bounded fixed-case descent control failed')
        result['architectures'][name]['short_descent_control']={'optimizer_calls':2,'total_phase_G_optimizer_calls':5,'learning_rate':1e-6,'losses':losses,'seconds':time.perf_counter()-start,'status':'short descent only, not overfit/convergence'}
        path.write_text(json.dumps(result,indent=2));print(name,losses,flush=True);del m,optimizer,report;gc.collect()
