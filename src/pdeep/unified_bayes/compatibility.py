"""Reviewed observability-only source transition; all other resume checks stay strict."""
from pathlib import Path
import json
from dataclasses import asdict,replace
from ..core import require
from ..artifact_io import read_manifest


def approved_source(source):
    from ..unified.config import implementation_identity
    current=implementation_identity()
    if source==current:return True
    proof=json.loads(Path(__file__).with_name('observability_transition.json').read_text())
    if source==proof['before'] and current==proof['after']:return True
    extra=Path(__file__).with_name('large_metadata_transition.json')
    if not extra.exists():return False
    transition=json.loads(extra.read_text())
    return current==transition['after'] and (source==transition['before'] or
        source==proof['before'] and proof['after']==transition['before'])


def resume_configurations(configs,paths):
    """Retain original run/config identity only across the reviewed source transition.

    All scientific settings, device, seeds, budgets and experiment ID must match.
    This never edits saved manifests or silently changes an experiment setting.
    """
    result=[]
    for c in configs:
        if c.model_key in paths:
            r=read_manifest(paths[c.model_key]);old=r['config'];new=asdict(c)
            require(approved_source(r['implementation_identity']) and approved_source(old['source_id']),'unreviewed source revision')
            require({k:v for k,v in old.items() if k!='source_id'}==json.loads(json.dumps({k:v for k,v in new.items() if k!='source_id'})), 'resume settings differ; keep original experiment configuration')
            c=replace(c,source_id=old['source_id'])
        result.append(c)
    return tuple(result)
