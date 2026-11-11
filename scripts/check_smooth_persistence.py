"""Non-optimizing full-state roundtrip; synthetic moment fixtures are labeled.

Requires the real numerical certificate. Uses the real writer/loader/Drive-copy
transaction. Filesystem Drive simulation is NOT actual Google Drive verification.
"""
from pathlib import Path
from dataclasses import asdict
import json,time,shutil
from unittest.mock import patch
import torch,numpy as np
from pdeep.core import require,digest
from pdeep.smooth.training import Config,Engine,resume
from pdeep.smooth.data import Case
from pdeep.artifact_io import read_payload,read_manifest
from pdeep.artifact_drive import PersistencePolicy,sync_verified
from pdeep.artifact_rng import capture_rng
from pdeep.artifact_tensors import tree_identity,state_identity

torch.set_num_threads(1)
root=Path('reports/smooth_finalization/persistence-fixture')/str(time.time_ns());calls=[]
def forbidden(*a,**k):calls.append(1);raise AssertionError('optimizer steps forbidden')
for cls in (torch.optim.AdamW,torch.optim.Adam,torch.optim.SGD):cls.step=forbidden
config=Config(architecture='C_GEO',preset='smoke',capacity='verification',updates=2,device='cpu',bank_queries=128,validation_grid=8,run_tag='nonoptimizing-full-state-fixture')
engine=Engine(config,root,allow_existing_run=True)
# Manually populated test moments, explicitly not the result of a training step.
for p in engine.model.parameters():engine.optimizer.state[p]={'step':torch.tensor(0.),'exp_avg':torch.full_like(p,.0003),'exp_avg_sq':torch.full_like(p,.0002)}
engine.rng_generators={'fixture':np.random.default_rng(37)}
# Real CPU64 production labels, valid observations and fixed bank verification.
case=Case('square','wave','train',0,config.conditioning_h);batch,ids=engine.bank.batch((case,),cursor=7,count=8)
with torch.no_grad():expected=engine.model(batch.specs,batch.observations,batch.queries)
checkpoint=engine.save_checkpoint();metadata,payload=read_payload(checkpoint,kind='training_checkpoint')
restored=resume(checkpoint,root/'resumed')
require(digest(tree_identity(restored.optimizer.state_dict()))==digest(tree_identity(payload['optimizer'])),'optimizer moments failed roundtrip')
require(restored.scheduler.state_dict()==payload['scheduler'],'scheduler failed roundtrip')
require(restored.sampler.state_dict(restored.state.sampler_cursor)==engine.sampler.state_dict(engine.state.sampler_cursor),'sampler/config/order failed roundtrip')
actual_rng=capture_rng(restored.rng_generators);require(digest(tree_identity(actual_rng))==digest(tree_identity(payload['rng'])),'full CPU/RNG streams failed roundtrip')
with torch.no_grad():actual=restored.model(batch.specs,batch.observations,batch.queries)
error=float(abs(actual-expected).max());require(error==0,'checkpoint prediction not exact')
# Compare the next real loss/backward without an optimizer step.
loss_a=engine.adapter.loss(engine.model,batch);loss_b=restored.adapter.loss(restored.model,batch)
loss_a.total.backward();loss_b.total.backward()
loss_difference=float(abs(loss_a.total.detach()-loss_b.total.detach()))
gradient_difference=max(float(abs(p.grad-q.grad).max()) for p,q in zip(engine.model.parameters(),restored.model.parameters()) if p.grad is not None and q.grad is not None)
require(loss_difference==0 and gradient_difference==0,'restored next backward differs')
require(state_identity(engine.model)==state_identity(restored.model),'backward changed parameters')
# New local root regenerates the SAME labels; explicitly test cache read separately.
b2,ids2=restored.bank.batch((case,),cursor=7,count=8)
require(torch.equal(batch.observations.value,b2.observations.value) and torch.equal(batch.queries.value,b2.queries.value) and ids==ids2,'bank/order/noise regeneration mismatch')
from pdeep.smooth.bank import Bank
with patch('pdeep.smooth.bank.evaluate_pairs',side_effect=AssertionError('cache miss unexpectedly recomputed reference')):
 cached=Bank(engine.bank.root,engine.certificate,train_queries=config.bank_queries,grid_size=config.validation_grid)
 cached_batch,cached_ids=cached.batch((case,),cursor=7,count=8)
 require(torch.equal(batch.queries.value,cached_batch.queries.value) and ids==cached_ids,'verified cache reload failed')
# Exercise the full real validation/weak-diagnostic interface on the cached wave
# fixture, without generating twelve new validation trajectories or training.
from dataclasses import replace
engine.manifest=replace(engine.manifest,cases=(('wave','square','train',0),),name='nonoptimizing cached-wave interface fixture')
validation=engine.validate();require(len(validation.architecture_metrics['per_case_quantity'])==2 and np.isfinite(validation.metrics['common']['field_normalized_MSE']),'validation failed to separate wave u/v')
mount=root/'filesystem-drive-simulation';mount.mkdir(parents=True,exist_ok=True)
policy=PersistencePolicy(True,str(mount),True,'filesystem_test',drive_root=str(mount/'MyDrive/research'))
status=sync_verified(checkpoint,policy);require(status['Drive_verified'] and status['Drive_status']=='completed','verified filesystem copy failed')
remote=resume(status['Drive_manifest'],root/'from-simulated-drive');require(state_identity(remote.model)==state_identity(engine.model),'portable full-state resume failed')
failed_policy=PersistencePolicy(True,str(mount),True,'filesystem_test',drive_root=str(mount/'failed-destination'))
with patch.object(shutil,'copyfile',side_effect=OSError('simulated disconnected filesystem')):failed=sync_verified(checkpoint,failed_policy)
require(failed['Drive_status']=='failed' and read_manifest(checkpoint)['artifact_id']==metadata['artifact_id'],'failed copy damaged local checkpoint')
result={'passed':True,'optimizer_calls':len(calls),'checkpoint':str(checkpoint),'synthetic_optimizer_moments':'manually populated serialization fixture, NOT trained moments','prediction_max_discrepancy':error,'next_loss_discrepancy':loss_difference,'next_gradient_max_discrepancy':gradient_difference,'optimizer_scheduler_rng_order_counters_restored':True,'reference_bank_regenerated_identically_after_resume':True,'verified_cache_read_without_reference_evaluation':True,'real_validation_and_weak_diagnostics':True,'filesystem_sync_status':status,'failed_sync_local_preserved':True,'actual_Colab_CUDA_Drive':'NOT TESTED'}
Path('reports/smooth_finalization/persistence.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
