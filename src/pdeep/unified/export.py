"""Explicit device-portable inference export; never mistaken for exact resume."""
from dataclasses import asdict
from pathlib import Path
from ..artifact_io import read_payload,write_artifact
from ..core import require,digest
from ..geometric.models import create,ModelConfig
from .config import SCHEMA,implementation_identity


def export_inference(checkpoint,dataset,root,*,selection='best'):
    require(selection in ('best','latest'),'unknown inference selection')
    r,p=read_payload(checkpoint,kind='training_checkpoint',required=('model','best_validation_model'))
    require(r.get('scientific_schema')==SCHEMA and r['dataset_identity']==dataset.config.identity,'not a compatible unified checkpoint')
    require(r['implementation_identity']==implementation_identity(),'source changed; review inference compatibility explicitly')
    weights=p['best_validation_model'] if selection=='best' else p['model'];require(weights is not None,'no validation best weights')
    metadata={'artifact_type':'inference_export','architecture_id':r['architecture_id'],'run_id':r['run_id'],'scientific_schema':SCHEMA,
              'parent_checkpoint':r['artifact_id'],'selection':selection,'dataset_identity':dataset.config.identity,'normalizer_identity':dataset.norms.identity,
              'implementation_identity':implementation_identity(),'model_config':r['model_config'],'source':r['source'],'neural_seed':r['config']['root_seed']}
    return write_artifact(root,f"inference-{r['artifact_id'][:20]}-{selection}",metadata,{'model':weights,'normalizers':dataset.norms.asdict()},required=('model','normalizers'),idempotent=True)


def load_inference(path,dataset,*,device='cpu'):
    r,p=read_payload(path,kind='inference_export',required=('model','normalizers'))
    require(r.get('scientific_schema')==SCHEMA and r['dataset_identity']==dataset.config.identity,'inference dataset mismatch')
    require(r['implementation_identity']==implementation_identity() and p['normalizers']['identity']==dataset.norms.identity,'inference source/normalization mismatch')
    cfg=ModelConfig(**r['model_config']);require(cfg==ModelConfig.preset(cfg.architecture,cfg.capacity),'architecture differs')
    model=create(cfg.architecture,dataset.norms,cfg.capacity,seed=r['neural_seed'],device=device);model.load_state_dict(p['model'],strict=True);model.eval();return model
