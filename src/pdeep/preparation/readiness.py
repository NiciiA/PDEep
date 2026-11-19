"""Complete declared-stream reference audit BEFORE any optimizer exists.

Case-major traversal bounds IO/RAM. Receipts are content-bound to the exact
dataset snapshot, schedule, sampler and adapter implementation. Warm reuse still
reverifies file fingerprints/SHA; no test payload is interpreted or used.
"""
from pathlib import Path
from collections import defaultdict
import time,json
from ..core import require,digest
from ..artifact_io import atomic_json,sha256
from ..smooth.data import Case
from ..unified.config import SCHEMA as DATA_SCHEMA
from ..unified.data import BalancedSampler
from ..unified.streams import coordinates_id
from .provider import FeasibilityProvider
from .coordinates import prepared_design
from .loading import verified_folder,MissingPreparedDraw
from .feasibility import source_identity

def ensure_ready(configs,dataset,root,*,progress=None,force=False):
    progress=progress or getattr(dataset,'progress',lambda *a,**k:None);start=time.perf_counter()
    require(configs and dataset.index.get('feasibility'),'a prepared feasibility dataset/configuration is required')
    require(all(c.dataset_identity==dataset.config.identity and c.updates==dataset.config.stream_updates and c.query_count==dataset.config.stream_queries and c.validation_query_count==dataset.config.stream_validation_queries and c.data_seed==dataset.config.seed for c in configs),'training/validation schedule differs from prepared dataset')
    # Strict local verification also protects reuse of a previous successful audit.
    _,index=verified_folder(dataset.path,{},progress)
    implementation=digest({p.name:sha256(p) for p in (Path(__file__),Path(__file__).with_name('validation.py'),Path(__file__).with_name('coordinates.py'))})
    signature={'schema':'G1-complete-training-readiness-v1','dataset_manifest':index['manifest_identity'],'dataset':dataset.config.identity,'mapping':index['feasibility']['mapping_identity'],'adapter':source_identity(),'audit_implementation':implementation,'updates':dataset.config.stream_updates,'training_queries':dataset.config.stream_queries,'validation_queries':dataset.config.stream_validation_queries,'validation_protocol':'all validation cases; original full=False views; fixed64 sensors; shared query/noise across models/checkpoints'}
    identity=digest(signature);folder=Path(root)/'training-readiness'/dataset.config.identity;path=folder/(identity+'.json')
    if path.exists() and not force:
        r=json.loads(path.read_text());require(r['identity']==digest(r['signature']) and r['signature']==signature and r.get('content_digest')==digest({k:v for k,v in r.items() if k!='content_digest'}),'readiness receipt identity mismatch')
        if r['status']=='ready':
            progress('Complete training readiness reused',checked_designs=r['checked_designs'],missing=0,seconds=time.perf_counter()-start,receipt=str(path));return r
    provider=FeasibilityProvider(dataset);require(len(provider.case_ids)==dataset.config.stream_updates,'recorded schedule horizon differs')
    rows={tuple(v['key']):v['case_id'] for v in dataset.index['cases']};byid={v['case_id']:v for v in dataset.index['cases']};events=defaultdict(list)
    sampler=BalancedSampler(type('Seed',(),{'data_seed':dataset.config.seed})(),dataset,provider=provider)
    for cursor in range(dataset.config.stream_updates):
        expected=tuple(rows[tuple(k)] for k in sampler.keys(cursor));require(expected==tuple(provider.case_ids[cursor]),'training case order differs from verified mapping')
        for slot,cid in enumerate(expected):
            a,b,n=map(int,provider.mapping[cursor,slot]);events[cid].append((a,b,n))
    selected=[v for v in index['cases'] if v['key'][2] in ('train','validation')];missing=[];case_reports=[];checked=0;training=validation=0
    missing_root=folder/(identity+'-missing');missing_root.mkdir(parents=True,exist_ok=True)
    for ordinal,row in enumerate(selected):
        p,g,s,i=row['key'];case=Case(g,p,s,i,dataset.config.conditioning_h);recipes=[]
        if s=='train':
            for a,b,n in events[case.case_id]:
                recipes.extend(((True,n,(DATA_SCHEMA,'encounter',dataset.config.seed,a,b)),(False,dataset.config.stream_queries,(DATA_SCHEMA,'supervised',dataset.config.seed,a,b))))
        else:
            recipes=[(False,dataset.config.stream_validation_queries,('unified-validation',dataset.config.seed)),(True,64,(DATA_SCHEMA,'fixed-evaluation',dataset.config.seed))]
        keys=[]
        for observation,n,sid in recipes:
            try:
                design=prepared_design(dataset,case,n,observation=observation,set_id=sid);key=coordinates_id(*design[:3]);keys.append((observation,n,sid,key))
            except MissingPreparedDraw as error:
                request=missing_root/(digest((case.case_id,observation,n,sid,error.request['coordinates_id']))+'.json');atomic_json(request,error.request)
                missing.append({'case_id':case.case_id,'split':s,'role':'observation' if observation else 'query','points':n,'coordinates_id':error.request['coordinates_id'],'request_file':request.relative_to(folder).as_posix(),'sha256':sha256(request)})
            checked+=1
            if s=='train':training+=1
            else:validation+=1
        case_reports.append({'case_id':case.case_id,'split':s,'PDE':p,'geometry':g,'required_designs':len(recipes),'verified_design_digest':digest(keys)})
        elapsed=time.perf_counter()-start;rate=(ordinal+1)/max(elapsed,1e-9)
        progress('Complete training reference audit',checked_cases=ordinal+1,total_cases=len(selected),checked_designs=checked,missing=len(missing),split=s,task=p+'/'+g,seconds=elapsed,cases_per_second=rate,ETA_seconds=(len(selected)-ordinal-1)/rate)
    body={'identity':identity,'signature':signature,'status':'ready' if not missing else 'missing','checked_designs':checked,'training_designs':training,'validation_designs':validation,'case_reports':case_reports,'missing':missing,'seconds':time.perf_counter()-start,'optimizer_updates':0,'reference_solves':0,'test_payloads_consumed':0,'preflight_references':'subset of declared training draws; smaller diagnostics slice paired full draws','checkpoint_references':'no independent spatial labels; checkpoint/Drive storage verified separately'}
    body['content_digest']=digest(body);atomic_json(path,body)
    require(not missing,f'Complete readiness found ALL {len(missing)} missing training/validation designs. Audit: {path}. Fail before optimizer construction; explicitly prepare the batch offline, then rerun this audit.')
    return body
