"""Complete variant-identified B checkpoints at successful optimizer boundaries.

Includes ALL encoders/refiner/spatial state, Adam moments, constant schedule,
stateless sampler next key/root seed, CPU RNG snapshot, priors/normalizers/vocab,
source/environment/dataset identity. No per-case spatial recurrence is persisted:
it is reconstructed/reset on every inference. No pending accumulated gradients.
"""
from dataclasses import asdict
import hashlib,json,os,tempfile
from pathlib import Path
import torch
from .architecture_b import build_architecture_b
from .b_training import BTrainingConfig
from .core import require,digest,decode
from .data import Normalizers
from .encoding import EquationVocabulary
from .runtime import resolve_device,environment_report


def tree_cpu(obj):
    if isinstance(obj,torch.Tensor):return obj.detach().to('cpu').clone()
    if isinstance(obj,dict):return {k:tree_cpu(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):return type(obj)(tree_cpu(v) for v in obj)
    return obj


def tree_identity(obj):
    if isinstance(obj,torch.Tensor):return {'shape':list(obj.shape),'dtype':str(obj.dtype),'sha256':hashlib.sha256(obj.contiguous().numpy().tobytes()).hexdigest()}
    if isinstance(obj,dict):return {str(k):tree_identity(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):return [tree_identity(v) for v in obj]
    return obj


def save_checkpoint(model,optimizer,config,directory,*,source,dataset_id,next_step,counters):
    require(model.approach==config.approach,'checkpoint config/variant mismatch')
    require(optimizer is not None and type(next_step) is int and next_step>=0,'B checkpoint requires optimizer and next sampler step')
    state=tree_cpu(model.state_dict());opt=tree_cpu(optimizer.state_dict())
    from .benchmarks import benchmark
    metadata=dict(version='B-inverse-v1',approach=model.approach,config=asdict(config),source=source,dataset_id=dataset_id,next_step=next_step,
                  counters=counters,normalizers=asdict(model.normalizers),vocabulary=list(model.vocabulary.quantities),
                  state_identity=tree_identity(state),optimizer_identity=tree_identity(opt),environment=environment_report(),
                  problem_ids={p:benchmark(p).identity for p in ('poisson','heat','wave')},physics='validated analytical T1 CPU64; differentiable transfers',
                  schedule='constant lr bounded protocol; no research scheduler',sampler='stateless B matched-cases keyed by root_seed and optimizer step',
                  geometry='geometry provider16+64 for B1 only',posterior_samples=False)
    metadata=json.loads(json.dumps(metadata,allow_nan=False));directory=Path(directory)/model.approach/digest(metadata);directory.mkdir(parents=True,exist_ok=True)
    path=directory/'checkpoint.pt';manifest=directory/'manifest.json'
    if path.exists() or manifest.exists():
        require(path.exists() and manifest.exists(),'incomplete B checkpoint')
        old=json.loads(manifest.read_text());require(old['metadata']==metadata and hashlib.sha256(path.read_bytes()).hexdigest()==old['payload_sha256'],'B checkpoint metadata/checksum mismatch');return manifest
    payload=dict(model=state,optimizer=opt,scheduler=None,scaler=None,torch_cpu_rng=torch.random.get_rng_state().clone(),
                 sampler=dict(root_seed=config.root_seed,next_step=next_step,dataset_id=dataset_id),counters=counters)
    with tempfile.NamedTemporaryFile(dir=directory,delete=False) as f:tmp=f.name
    try:torch.save(payload,tmp);os.replace(tmp,path)
    finally:
        if Path(tmp).exists():Path(tmp).unlink()
    record=dict(metadata=metadata,payload_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    with tempfile.NamedTemporaryFile(mode='w',dir=directory,delete=False) as f:json.dump(record,f,indent=2,allow_nan=False);f.write('\n');tmp=f.name
    os.replace(tmp,manifest);return manifest


def load_checkpoint(manifest,*,expected_approach,device='cpu',restore_optimizer=True,restore_rng=False):
    resolved,_=resolve_device(device,'float32');manifest=Path(manifest);record=json.loads(manifest.read_text());m=record['metadata']
    require(m['version']=='B-inverse-v1' and digest(m)==manifest.parent.name,'B checkpoint identity/version mismatch')
    require(expected_approach in {'B0','B1'} and m['approach']==expected_approach,'incompatible B checkpoint variant')
    path=manifest.parent/'checkpoint.pt';require(hashlib.sha256(path.read_bytes()).hexdigest()==record['payload_sha256'],'B checkpoint checksum mismatch')
    payload=torch.load(path,map_location='cpu',weights_only=True)
    require(tree_identity(payload['model'])==m['state_identity'] and tree_identity(payload['optimizer'])==m['optimizer_identity'],'B checkpoint tensor identity mismatch')
    config=decode(BTrainingConfig,m['config']);require(config.approach==expected_approach,'checkpoint config/variant mismatch')
    model=build_architecture_b(expected_approach,decode(Normalizers,m['normalizers']),EquationVocabulary(tuple(m['vocabulary'])),root_seed=config.root_seed,iterations=config.iterations)
    model.load_state_dict(payload['model'],strict=True);model.to(resolved).eval()
    require(all(bool(torch.isfinite(t).all()) for t in model.state_dict().values() if t.is_floating_point()),'nonfinite B checkpoint tensors')
    optimizer=None
    if restore_optimizer:
        optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
        optimizer.load_state_dict(payload['optimizer'])
    if restore_rng:torch.random.set_rng_state(payload['torch_cpu_rng'])
    return model,optimizer,config,payload['sampler'],m
