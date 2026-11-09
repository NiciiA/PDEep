"""C inference export with normalizers, initial RMS and physical-rule provenance.

This export is not an optimizer-resume checkpoint. Load checks version, identity,
checksum and strict tensor schema using torch weights_only=True.
"""
from dataclasses import asdict
import hashlib,json,os,tempfile
from pathlib import Path
import torch
from .core import require,digest,decode
from .data import Normalizers
from .encoding import EquationVocabulary
from .architecture_c import build_architecture_c
from .constraints import prior_initial_scales


def save_inference(model,directory,*,source,config,status='untrained',dataset_id=None):
    from .benchmarks import benchmark
    state={k:t.detach().cpu().clone() for k,t in model.state_dict().items()}
    hashes={k:hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest() for k,t in state.items()}
    metadata=dict(version='C-field-weak-v1',source=source,config=config,status=status,dataset_id=dataset_id,
                  normalizers=asdict(model.normalizers),vocabulary=list(model.vocabulary.quantities),
                  initial_RMS={name:prior_initial_scales(benchmark(name)) for name in ('heat','wave')},
                  quadrature=dict(spatial_order=16,time_order=8),geometry='unit_square_homogeneous_Dirichlet',
                  decoder_frequencies=[1,2,4,8],observation_frequencies=[1,2,4,8,16,32],state_hashes=hashes,
                  artifact_kind='inference only; not an optimizer-resume checkpoint')
    metadata=json.loads(json.dumps(metadata,allow_nan=False));directory=Path(directory)/digest(metadata);directory.mkdir(parents=True,exist_ok=True)
    path=directory/'weights.pt';manifest=directory/'manifest.json'
    if path.exists() or manifest.exists():
        require(path.exists() and manifest.exists(),'incomplete C inference export')
        old=json.loads(manifest.read_text());require(old['metadata']==metadata,'C export metadata mismatch')
        require(hashlib.sha256(path.read_bytes()).hexdigest()==old['weights_sha256'],'C weights checksum mismatch');return manifest
    with tempfile.NamedTemporaryFile(dir=directory,delete=False) as stream:tmp=stream.name
    try:torch.save(state,tmp);os.replace(tmp,path)
    finally:
        if Path(tmp).exists():Path(tmp).unlink()
    contents=dict(metadata=metadata,weights_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    with tempfile.NamedTemporaryFile(mode='w',dir=directory,delete=False) as stream:
        json.dump(contents,stream,indent=2,allow_nan=False);stream.write('\n');tmp=stream.name
    os.replace(tmp,manifest);return manifest


def load_inference(manifest,*,device='cpu'):
    from .runtime import resolve_device
    resolved,_=resolve_device(device,'float32');manifest=Path(manifest);record=json.loads(manifest.read_text());m=record['metadata']
    require(m['version']=='C-field-weak-v1' and digest(m)==manifest.parent.name,'C artifact version/identity mismatch')
    path=manifest.parent/'weights.pt';require(hashlib.sha256(path.read_bytes()).hexdigest()==record['weights_sha256'],'C weights checksum mismatch')
    model=build_architecture_c(decode(Normalizers,m['normalizers']),EquationVocabulary(tuple(m['vocabulary'])))
    state=torch.load(path,map_location='cpu',weights_only=True)
    require(all(bool(torch.isfinite(t).all()) for t in state.values() if t.is_floating_point()),'nonfinite C artifact state')
    require({k:hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest() for k,t in state.items()}==m['state_hashes'],'C tensor checksum mismatch')
    model.load_state_dict(state,strict=True);model.to(resolved).eval();return model,m
