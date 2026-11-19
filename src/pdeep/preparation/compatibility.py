"""Reviewed adapter-only continuation; scientific/model identities stay strict."""
import json
from pathlib import Path
from dataclasses import asdict,replace
from ..core import require
from ..artifact_io import read_manifest
from ..unified_bayes.compatibility import approved_source
from ..unified_modes.compatibility import approved_extension
from ..unified_modes.architecture import model_config
from .feasibility import source_identity

def approved_adapter(source):
    current=source_identity()
    if source==current:return True
    path=Path(__file__).with_name('readiness_transition.json')
    if not path.exists():return False
    proof=json.loads(path.read_text())
    return source==proof['before'] and current==proof['after']

def resume_configurations(configs,paths):
    if not configs or not hasattr(configs[0],'feasibility_source'):
        from ..unified_modes.compatibility import resume_configurations as original
        return original(configs,paths)
    result=[]
    for c in configs:
        if c.model_key in paths:
            r=read_manifest(paths[c.model_key]);old=r['config'];new=asdict(c)
            require(approved_adapter(old['feasibility_source']) and approved_source(old['source_id']) and approved_extension(old['extension_source'],c.capacity),'unreviewed resume source')
            require(r['model_config']==asdict(model_config(c.architecture,c.capacity)),'resume architecture differs')
            omit={'source_id','extension_source','feasibility_source'}
            require({k:v for k,v in old.items() if k not in omit}==json.loads(json.dumps({k:v for k,v in new.items() if k not in omit})),'resume settings differ; use original experiment configuration')
            c=replace(c,**{k:old[k] for k in omit})
        result.append(c)
    return tuple(result)
