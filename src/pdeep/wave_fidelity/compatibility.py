"""Explicit immutable migration for checkpoints unaffected by the area-format fix.

This is NOT a loader bypass. All old geometric source files except the known
mesher fix must match. Every saved numerical mesh must regenerate identically.
A checkpoint made with the broken fine-mesh option is rejected. Payloads/model
weights/optimizer/RNG are copied byte-for-byte as tensor trees; originals survive.
"""
from pathlib import Path
from ..core import require
from ..artifact_io import read_payload,write_artifact
from ..geometric.training import geometry_identity
from ..geometric.data import physical_case

OLD_DOMAIN='5739ef76399e4d43941d29be3670d63f0df72820926ff27c01ca7ebc70eb9a4b'
FIXED_DOMAIN='bf7d8f8604f0a5999be4174c451212b5dbefea1951070ccda5366b5ccacea4a8'
OLD_TRAINING='65c71ee147edeff648992a7738f123d27eb4a31c1e18ffb22954eb3ae5f33e4c'
GATED_TRAINING='1f526f61f5dd192c3d9222b4f1c288648e5ab265cd92ae9198592b7c970a6954'


def migrate_unaffected_checkpoint(path,destination):
    record,payload=read_payload(path,kind='training_checkpoint')
    require(record.get('scientific_schema')=='geometry-inverse-v1','only explicit geometry checkpoint migration supported')
    old=record['geometry_source'];new=geometry_identity();changed={k for k in set(old)|set(new) if old.get(k)!=new.get(k)}
    # The later portable-input repair adds only a verified array loader. It is
    # admitted by an exact source manifest AND the unchanged-mesh checks below.
    portable_ok=False;representation_cache_ok=False
    proof=Path(__file__).resolve().parents[3]/'certifications/G1_smooth_v1/portable/execution.json'
    if proof.exists():
        import json
        from ..core import digest
        r=json.loads(proof.read_text())
        intact=r.get('identity')==digest({k:v for k,v in r.items() if k!='identity'})
        expected=r.get('repaired_geometry_sources',{})
        normalized=dict(new)
        if intact and old.get('representation.py')==expected.get('representation.py'):
            from ..learning_diagnostics.execution_compatibility import approved_workflow_predecessor
            representation_cache_ok=approved_workflow_predecessor(r['repaired_implementation'])
            if representation_cache_ok:normalized['representation.py']=expected['representation.py']
        portable_ok=intact and normalized==expected
    allowed={'domain.py','training.py','portable.py'} if portable_ok else {'domain.py','training.py'}
    if representation_cache_ok:allowed.add('representation.py')
    require(bool(changed) and changed<=allowed and old['domain.py'] in {OLD_DOMAIN,FIXED_DOMAIN} and (new['domain.py']==FIXED_DOMAIN or portable_ok) and
            old['training.py'] in {OLD_TRAINING,GATED_TRAINING} and new['training.py']==GATED_TRAINING,
            'migration requires only the reviewed mesher correction and research-launch gate; all scientific model/data code must match')
    config=record['config'];verified=[]
    for item in record['meshes']:
        case=physical_case(item['split'],item['pde'],item['case_index'],h=config['reference_h'],dt=config['reference_dt'])
        require(case.problem.mesh.mesh_id==item['mesh_id'] and case.case_id==item['physical_case_id'],'checkpoint numerical mesh/physical case changed; cannot claim equivalent continuation')
        require(case.manifest['prior']==item['prior'],'checkpoint prior changed')
        verified.append(item['mesh_id'])
    exclude={'artifact_id','schema_version','created_UTC','files','verification','content_identity','payload_tree_identity'}
    metadata={k:v for k,v in record.items() if k not in exclude}
    metadata.update(geometry_source=new,compatibility_migration={'version':'G3-reviewed-mesh-format-v1','original_artifact':record['artifact_id'],
                    'original_manifest':str(Path(path).resolve()),'old_domain_sha256':old['domain.py'],'fixed_domain_sha256':new['domain.py'],
                    'old_training_sha256':old['training.py'],'gated_training_sha256':GATED_TRAINING,
                    'training_difference':'run() rejects uncertified research budgets; update arithmetic unchanged',
                    'representation_cache_compatibility':representation_cache_ok,'representation_source':new.get('representation.py'),'verified_unchanged_meshes':verified,'weights_optimizer_RNG_changed':False})
    migrated=write_artifact(destination,None,metadata,payload,required=tuple(payload),idempotent=True)
    copied,_=read_payload(migrated,kind='training_checkpoint')
    require(copied['payload_tree_identity']==record['payload_tree_identity'],'migration changed payload state')
    return migrated
