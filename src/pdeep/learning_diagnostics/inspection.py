"""Read-only trained-weight inspection; never treats mixed checkpoints as fresh runs."""
import json
from pathlib import Path
from ..core import require
from ..artifact_io import read_payload,atomic_json
from ..geometric.models import GeometryModel,ModelConfig
from ..smooth.certification import authorize,normalizers
from ..smooth.training import select_device


def load_model(path,certificate,*,device='auto'):
    meta,p=read_payload(path,required=('model','model_config','normalizers'))
    require(meta.get('benchmark')=='G1_smooth_v1','select a smooth geometry checkpoint')
    require(meta['artifact_type'] in {'training_checkpoint','inference_export'},'not an inference-capable artifact')
    from ..smooth.compatibility import approved_predecessor
    implementation=meta.get('implementation_identity',meta.get('config',{}).get('source_id'))
    require(approved_predecessor(implementation),'checkpoint/export architecture/loss implementation is incompatible')
    authorize(certificate)
    require(p['normalizers']['identity']==certificate['normalizers']['identity'],'trained normalization incompatible with selected numerical certificate')
    m=GeometryModel(ModelConfig(**p['model_config']),normalizers(certificate)).to(select_device(device));m.load_state_dict(p['model'],strict=True);m.eval()
    return m,meta


def inspect_mixed_checkpoint(path,certificate,root,*,device='auto',grid_size=32,gradients=False):
    """Optional Colab cell: diagnose actual old six-case pool, no optimizer created."""
    from .data import prepare_data
    from .metrics import reconstruction,representation,gradient_terms
    from .sensitivity import sensitivity
    from ..geometric.physics import Adapter,WeakConfig
    m,meta=load_model(path,certificate,device=device);c=meta['config']
    require(c['cases_per_geometry']==1 and c['preset']=='tiny_overfit','this inspector targets the reported six-case tiny-overfit, not implicit new datasets')
    result={'checkpoint':str(path),'successful_updates':meta.get('successful_update'),'actual_config':c,'cases':[],'optimizer_updates':0}
    for pde in c['pdes']:
        for family in c['training_families']:
            data=prepare_data(Path(root)/'learning_diagnostics'/'inspection_data',certificate,experiment='observation_sensitivity',seed=0,pde=pde,family=family,grid_size=grid_size,conditioning_h=c['conditioning_h'],count=c['query_count'])
            r={'case_id':data.case.case_id,'pde':pde,'family':family,'observations':'original case and observation_set=0, verified generator; independent diagnostic grid','metrics':reconstruction(m,data),'representation':representation(m,data),'sensitivity':sensitivity(m,data)}
            if gradients:r['loss_gradients']=gradient_terms(m,data.batch(next(m.parameters()).device),Adapter(m.config.architecture,WeakConfig(c['spatial_order'],c['time_order'],c['weak_tests'],c['physics_chunk'])))
            result['cases'].append(r)
    dest=Path(root)/'learning_diagnostics'/'trained_checkpoint_inspection'/meta['artifact_id']/'report.json';atomic_json(dest,result)
    return result,dest
