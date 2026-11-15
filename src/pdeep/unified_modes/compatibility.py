"""Narrow, reviewed medium-only transition; never expands checkpoint tensors."""
import json
from dataclasses import asdict,replace
from pathlib import Path
from .config import extension_identity
from ..core import require
from ..artifact_io import read_manifest
from ..unified_bayes.compatibility import approved_source
from .architecture import model_config

def approved_extension(source,capacity):
    current=extension_identity()
    if source==current:return True
    path=Path(__file__).with_name('large_transition.json')
    if not path.exists():return False
    proof=json.loads(path.read_text())
    return capacity in ('medium','verification') and source==proof['before'] and current==proof['after']

def resume_configurations(configs,paths):
    result=[]
    for c in configs:
        if c.model_key in paths:
            r=read_manifest(paths[c.model_key]);old=r['config'];new=asdict(c)
            require(approved_extension(old['extension_source'],c.capacity) and approved_source(old['source_id']),'unreviewed resume source')
            require(r['model_config']==asdict(model_config(c.architecture,c.capacity)),'resume architecture differs')
            omit={'source_id','extension_source'}
            require({k:v for k,v in old.items() if k not in omit}==json.loads(json.dumps({k:v for k,v in new.items() if k not in omit})),'resume settings differ; use the original experiment configuration')
            c=replace(c,source_id=old['source_id'],extension_source=old['extension_source'])
        result.append(c)
    return tuple(result)
