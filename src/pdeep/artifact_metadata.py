"""Stable scientific provenance, explicit constructor schemas and compatibility.

Inspection is JSON-only. Constructors come from the installed registry, never
from an executable class/path in a manifest. This module imports no trainer.
"""
from dataclasses import asdict
from pathlib import Path
import json
import torch
from .core import digest,decode,require
from .data import Normalizers, CaseDataset, VERSION, SPLITS, SENSOR_COUNTS
from .reference import canonical_spec
from .encoding import EquationVocabulary
from .runtime import environment_report
from .diagnostics import source_identity
from .geometry import BoxGeometry
from .artifact_tensors import state_schema,tree_identity

ROOT=Path(__file__).resolve().parents[2]
FAMILIES=('poisson','heat','wave')
SCIENTIFIC_FILES=('docs/scientific_contract_v1.md',)+tuple('src/pdeep/'+name+'.py' for name in (
    'core','spec','graph','benchmarks','batches','reference','data','diagnostics','encoding','geometry','memory',
    'representations','architecture_a','architecture_c','architecture_b','constraints','b_physics','bayes',
    'a_training','c_training','b_training','train_registry','train_data','train_types','training','evaluation_plan','evaluation_metrics'))


def scientific_identity():
    from .artifact_io import sha256
    return {name:sha256(ROOT/name) for name in SCIENTIFIC_FILES}


def architecture_configuration(architecture):
    require(architecture in {'A','C','B0','B1','D'},'exact A/C/B0/B1/D architecture required')
    return {'version':'canonical-model-v1','capacity':'canonical_v1','benchmark':'unit_square_modal_v1','inverse_track':'T1',
            'hidden_width':0 if architecture=='D' else 96,'typed_message_layers':0 if architecture=='D' else 2,
            'coefficient_slots':18,'active_dimensions':{'poisson':9,'heat':9,'wave':18},
            'coefficient_order':'a lexicographic (1,1)..(3,3), then b for wave; inactive zero padding',
            'decoder':{'hidden_layers':4,'width':128,'smooth_activation':'SiLU'} if architecture in {'A','C'} else None,
            'refinement_steps':3 if architecture.startswith('B') else 0,
            'geometry':asdict(BoxGeometry()),'boundary':'homogeneous unit-square Dirichlet; product factor for A/C',
            'spatial_anchors':{'coarse':16,'fine':64} if architecture in {'A','B1'} else None,
            'A_retrieval':{'kind':'all-anchor normalized Gaussian','coarse_length':.30,'fine_length':.15} if architecture=='A' else None,
            'C_constraints':{'spatial_Gauss_order':16,'spatial_check_order':32,'time_Gauss_order':8,'sine_tests':9} if architecture=='C' else None,
            'physical_operator':'direct analytical CPU float64 T1; no temporal interpolation' if architecture in {'B0','B1','D'} else None}


def objective_configuration(architecture):
    values={'A':{'field':1.,'xi':.1},'C':{'field':1.,'xi':.1,'weak':.1,'initial':.1},'B0':{'xi_final':1.},'B1':{'xi_final':1.},'D':{}}
    return {'version':'scientific-objectives-v1','implementation':{'A':'a_training.objective','C':'c_training.objective','B0':'b_training.coefficient_loss','B1':'b_training.coefficient_loss','D':'bayes.ExactGaussianD'}[architecture],
            'coefficients':values[architecture],'best_validation_metric':'common.field_normalized_MSE (fixed validation query-risk, not raw training loss)'}


def model_inventory(model):
    if model is None:return {'modules':{},'state_schema':{},'parameter_count':0,'model_state_identity':None}
    groups={}
    for name,module in model.named_children():
        keys=[k for k in model.state_dict() if k.startswith(name+'.')]
        groups[name]={'class':type(module).__module__+'.'+type(module).__name__,
                      'state_keys':keys,'parameters':sum(p.numel() for p in module.parameters())}
    return {'modules':groups,'state_schema':state_schema(model.state_dict()),
            'parameter_count':sum(p.numel() for p in model.parameters()),'model_state_identity':digest(tree_identity(model.state_dict()))}


def make_metadata(kind,architecture,normalizers,*,model=None,run_id,config=None,successful=0,stage='stage7C',parents=(),device=None):
    require(kind in {'training_checkpoint','inference_export','diagnostic_bundle'},'artifact type required')
    norm=normalizers;specs=tuple(canonical_spec(p) for p in FAMILIES)
    vocabulary=model.vocabulary if model is not None else EquationVocabulary.from_specs(specs)
    geometry=getattr(model,'geometry_provider',None)
    if geometry is None and getattr(model,'spatial',None) is not None:geometry=model.spatial.geometry_provider
    if geometry is not None:require(geometry==BoxGeometry(),'artifact constructor currently supports only the canonical geometry configuration')
    source=source_identity(ROOT);dataset=CaseDataset(norm.root_seed)
    policy={'version':'observations-v1','default_counts':SENSOR_COUNTS,'positions':'Uniform[.05,.95]^2',
            'dynamic_time_mixture':{'zero':.1,'early':.3,'late':.6},'wave':'balanced u/v',
            'noise':'independent Gaussian; eta logUniform[.02,.08], sigma=eta*training_quantity_RMS',
            'streams':'separate keyed position/time/quantity/eta/noise; case split preserved'}
    resolved=asdict(config) if config is not None else None
    save_device=device or (str(next(model.parameters()).device) if model is not None else 'cpu')
    return {'artifact_type':kind,'run_id':run_id,'architecture_id':architecture,
            'model_class':None if model is None else type(model).__module__+'.'+type(model).__name__,
            'architecture_configuration':architecture_configuration(architecture),'component_inventory':model_inventory(model),
            'scientific_contract':{'version':'1.0','sha256':source['files']['docs/scientific_contract_v1.md']},
            'source':source,'scientific_source_files':scientific_identity(),'runtime':environment_report(),
            'save_device':save_device,'save_precision':'float64' if architecture=='D' else 'float32 neural; explicit CPU64 physical/reference calculations',
            'problem_specs':{s.name:json.loads(s.to_json()) for s in specs},'problem_identities':{s.name:s.identity for s in specs},
            'equation_vocabulary':{'version':'typed-representation-v1','identity':vocabulary.identity,'quantities':list(vocabulary.quantities)},
            'geometry_and_boundaries':{s.name:{'geometry':asdict(s.geometry),'boundaries':[asdict(v) for v in s.boundaries]} for s in specs},
            'priors':{s.name:[asdict(b) for b in s.parameter_blocks] for s in specs},
            'normalization':{'identity':norm.identity,'values':asdict(norm),'version':norm.version},
            'data_generation':{'version':VERSION,'dataset_identity':dataset.identity,'root_seed':dataset.root_seed,
                               'splits':[{'name':n,'count_per_PDE':count,'global_ID_offset':offset,
                                          'identity':digest((dataset.identity,n,count,offset))} for n,count,offset in SPLITS]},
            'observation_policy':{**policy,'identity':digest(policy)},'objective':objective_configuration(architecture),
            'training_config':resolved,'config_identity':None if config is None else config.identity,
            'optimizer_configuration':None if config is None else {'family':'AdamW','lr':config.learning_rate,'weight_decay':config.weight_decay,'grad_clip_norm':config.grad_clip_norm,'AMP':False,'scaler':None},
            'scheduler_configuration':None if config is None else {'kind':config.scheduler,'updates':config.updates,'warmup_updates':config.warmup_updates,'minimum_lr_fraction':config.minimum_lr_fraction},
            'training_stage':stage,'successful_updates':successful,
            'evaluation_protocol':{'version':'matched-evaluation-v1','track':'T1','default_split':'validation','test_requires_explicit_final_authorization':True},
            'dependencies':{'package':'pdeep-research','runtime_solvers':{s.name:s.solver_key for s in specs},
                            'parent_checkpoints':list(parents),'self_contained_payload':True,'external_checkpoint_required_to_load':False}}


def validate_metadata(record, *, model=None):
    required={'architecture_id','architecture_configuration','scientific_contract','scientific_source_files','objective',
              'normalization','equation_vocabulary','problem_specs','problem_identities','priors','geometry_and_boundaries',
              'dependencies','data_generation','observation_policy','component_inventory','model_class'}
    require(required<=record.keys(),'artifact required scientific/normalization/model metadata missing: '+str(sorted(required-record.keys())))
    arch=record['architecture_id']
    require(record['architecture_configuration']==json.loads(json.dumps(architecture_configuration(arch))), 'model dimensions/track/geometry/physical parameterization mismatch')
    require(record['scientific_contract']=={'version':'1.0','sha256':__import__('hashlib').sha256((ROOT/'docs/scientific_contract_v1.md').read_bytes()).hexdigest()},'scientific contract mismatch')
    require(record['scientific_source_files']==scientific_identity(),'scientific source/solver/objective incompatibility; explicit versioned migration required')
    require(record['objective']==objective_configuration(arch),'objective/loss coefficient mismatch')
    require('normalization' in record and 'values' in record['normalization'],'normalization metadata missing')
    norm=decode(Normalizers,record['normalization']['values']);require(norm.identity==record['normalization']['identity'],'normalization identity mismatch')
    vocab=EquationVocabulary(tuple(record['equation_vocabulary']['quantities']))
    require(record['equation_vocabulary']['version']=='typed-representation-v1' and vocab.identity==record['equation_vocabulary']['identity'],'equation vocabulary version/identity incompatible')
    for p in FAMILIES:
        from .spec import ProblemSpec
        spec=ProblemSpec.from_json(json.dumps(record['problem_specs'][p]))
        require(spec==canonical_spec(p) and spec.identity==record['problem_identities'][p],'unsupported physical operator/parameterization/specification')
        require(record['priors'][p]==json.loads(json.dumps([asdict(b) for b in spec.parameter_blocks])),'prior definitions changed')
        require(record['geometry_and_boundaries'][p]==json.loads(json.dumps({'geometry':asdict(spec.geometry),'boundaries':[asdict(b) for b in spec.boundaries]})),'geometry/boundary mismatch')
        require(record['dependencies']['runtime_solvers'][p]==spec.solver_key,'required physical solver support missing')
    require(record['data_generation']['version']==VERSION and record['data_generation']['dataset_identity']==CaseDataset(norm.root_seed).identity,'data-generation identity mismatch')
    policy={k:v for k,v in record['observation_policy'].items() if k!='identity'}
    require(digest(policy)==record['observation_policy']['identity'],'observation policy identity mismatch')
    if model is not None:
        require(record['model_class']==type(model).__module__+'.'+type(model).__name__,'model class incompatible with installed constructor')
        require(record['component_inventory']['state_schema']==state_schema(model.state_dict()),'required model components/dimensions missing or incompatible')
        require(vocab.identity==model.vocabulary.identity,'equation vocabulary incompatible with constructed model')
    return norm,vocab
