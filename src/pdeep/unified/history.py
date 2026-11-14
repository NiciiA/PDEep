"""Small immutable incremental JSONL shards; exact known artifact paths only."""
from pathlib import Path
import json
from ..artifact_io import write_artifact,read_payload
from ..core import require
from .config import SCHEMA
from .persistence import sync


def archive(engine):
    previous=engine.history_refs[-1]['through_update'] if engine.history_refs else 0
    rows={}
    for name in ('steps.jsonl','case_losses.jsonl','validation.jsonl'):
        path=engine.run_dir/name;selected=[]
        if path.exists():
            with path.open() as stream:
                for line in stream:
                    row=json.loads(line);update=row.get('update',row.get('successful',0))
                    if previous<update<=engine.state.successful:selected.append(row)
        rows[name]=selected
    if not any(rows.values()):return
    path=write_artifact(engine.run_dir/'history',f'history-{engine.state.successful:08d}',
        {'artifact_type':'diagnostic_bundle','architecture_id':engine.config.architecture,'run_id':engine.config.run_id,
         'scientific_schema':SCHEMA,'from_update_exclusive':previous,'through_update':engine.state.successful},
        {'history':rows},required=('history',),idempotent=True)
    status=sync(path,engine.persistence_policy)
    engine.history_refs.append({'from_update_exclusive':previous,'through_update':engine.state.successful,'local_manifest':str(path),**status})


def read(refs):
    out={name:[] for name in ('steps.jsonl','case_losses.jsonl','validation.jsonl')}
    for ref in refs:
        local=Path(ref['local_manifest']);source=local if local.is_file() else ref.get('Drive_manifest')
        require(source is not None,'missing explicit persisted history shard')
        _,p=read_payload(source,kind='diagnostic_bundle',required=('history',))
        for name,rows in p['history'].items():out[name].extend(rows)
    return out
