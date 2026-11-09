"""Versioned A inference export; safe weights-only load, immutable/checksummed.

This is not a resumable research checkpoint. Optimizer-step-boundary training
resume needs sampler/schedule state and remains outside the Stage 4 smoke scope.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import os
import tempfile
import torch
from .architecture_a import build_architecture_a
from .core import require,decode,digest
from .data import Normalizers
from .encoding import EquationVocabulary


def save_inference(model,directory,*,source,config,status="untrained",dataset_id=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    state={name:t.detach().cpu().clone() for name,t in model.state_dict().items()}
    state_id=digest({k:{"shape":list(v.shape),"dtype":str(v.dtype),"sha256":hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest()} for k,v in state.items()})
    metadata={"version":"A-direct-v1","status":status,"normalizers":asdict(model.normalizers),
              "vocabulary":list(model.vocabulary.quantities),"source":source,"config":config,"dataset_id":dataset_id,
              "geometry":"unit_square_16coarse_64fine","boundary":"homogeneous_square_product",
              "frequencies":[1,2,4,8],"parameter_inventory":model.parameter_inventory(),"state_id":state_id,
              "artifact_kind":"inference; not a resumable optimizer checkpoint"}
    metadata=json.loads(json.dumps(metadata,allow_nan=False))
    run=digest(metadata);target=directory/run;target.mkdir(exist_ok=True)
    path=target/"weights.pt";manifest=target/"manifest.json"
    if path.exists() or manifest.exists():
        require(path.exists() and manifest.exists(),"incomplete existing inference export")
        existing=json.loads(manifest.read_text());require(existing["metadata"]==metadata,"artifact metadata mismatch")
        require(hashlib.sha256(path.read_bytes()).hexdigest()==existing["weights_sha256"],"inference weights checksum mismatch")
        return manifest
    with tempfile.NamedTemporaryFile(dir=target,delete=False) as stream:temporary=stream.name
    try:
        torch.save(state,temporary);os.replace(temporary,path)
    finally:
        if Path(temporary).exists():Path(temporary).unlink()
    content={"metadata":metadata,"weights_sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    with tempfile.NamedTemporaryFile(mode="w",dir=target,delete=False) as stream:
        json.dump(content,stream,indent=2,allow_nan=False);stream.write("\n");temporary=stream.name
    os.replace(temporary,manifest)
    return manifest


def load_inference(manifest,*,device="cpu"):
    from .runtime import resolve_device
    resolved,_=resolve_device(device,"float32")
    manifest=Path(manifest);content=json.loads(manifest.read_text());metadata=content["metadata"]
    require(metadata["version"]=="A-direct-v1" and digest(metadata)==manifest.parent.name,"A inference identity/version mismatch")
    weights=manifest.parent/"weights.pt"
    require(hashlib.sha256(weights.read_bytes()).hexdigest()==content["weights_sha256"],"A inference weights checksum mismatch")
    model=build_architecture_a(decode(Normalizers,metadata["normalizers"]),EquationVocabulary(tuple(metadata["vocabulary"])))
    state=torch.load(weights,map_location="cpu",weights_only=True)
    model.load_state_dict(state,strict=True);model.to(resolved).eval()
    require(all(bool(torch.isfinite(t).all()) for t in model.state_dict().values() if t.is_floating_point()),"nonfinite inference weights")
    return model,metadata
