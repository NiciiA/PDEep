"""Two explicit one-update diagnostic runs, not the requested100-update runs.

Real unchanged medium models, existing AdamW engine/losses, actual certified
CPU64 references, three PDEs, two fixed domains. Small query/grid settings bound
verification cost. Notebook budgets/optimizer settings remain unchanged.
"""
from pathlib import Path
import json,time
import torch
from pdeep.smooth.training import Config,Engine,resume
from pdeep.smooth.workflows import prepare
from pdeep.smooth.qualification import ensure_compatible_certificate
from pdeep.artifact_tensors import state_identity
from pdeep.artifact_io import read_payload
from pdeep.core import require,digest

torch.set_num_threads(1);root=Path('reports/colab_repair/bounded-verification')/str(time.time_ns());start=time.perf_counter()
certificate,path=ensure_compatible_certificate()
plan=prepare(models=('A_GEO','C_GEO'),experiment='tiny_overfit',updates=100,device='cpu',certificate_path=path)
require(all(c.updates==100 and len(c.case_pool)==6 for c in plan.configs),'requested plan budget/pool changed')
rows={}
for architecture in ('A_GEO','C_GEO'):
 print('EXPLICIT BOUNDED VERIFICATION',architecture,'one update only; original100-update plan NOT executed',flush=True)
 config=Config(architecture=architecture,preset='tiny_overfit',updates=1,training_families=('square','rectangle'),cases_per_geometry=1,capacity='medium',device='cpu',bank_queries=128,query_count=8,validation_grid=8,validation_every=1,checkpoint_every=1,run_tag='colab-repair-one-update')
 engine=Engine(config,root,certificate=certificate);before=state_identity(engine.model);reports=engine.run();require(engine.state.successful==1 and before!=state_identity(engine.model),'actual bounded update did not execute')
 checkpoint=engine.state.last_checkpoint;record,payload=read_payload(checkpoint,kind='training_checkpoint');loaded=resume(checkpoint,root/'resume-check')
 require(state_identity(loaded.model)==state_identity(engine.model),'bounded checkpoint model restoration failed')
 require(digest(payload['sampler'])==digest(loaded.sampler.state_dict(loaded.state.sampler_cursor)),'bounded sampler resume mismatch')
 rows[architecture]={'actual_successful_updates':1,'failed':engine.state.failed,'checkpoint':checkpoint,'parameters':engine.model.parameter_count,'optimizer_state_entries':len(loaded.optimizer.state),'model_resume_identical':True,'validation_case_quantity_rows':len(engine.latest_validation.architecture_metrics['per_case_quantity']),'real_loss_terms':list(reports[0].unweighted),'scope':'execution verification only; NOT overfit success or generalization evidence'}
 del loaded,engine
result={'passed':True,'actual_optimizer_updates':2,'notebook_requested_overfit_budget':100,'requested_plan_executed':False,'architectures':rows,'seconds':time.perf_counter()-start}
Path('reports/colab_repair/bounded-training.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
