"""EXPLICIT offline repair of one exact missing draw; never called by loading.

Original cases remain immutable. A new verified supplemental artifact, containing
both exact coordinates and CPU64 labels, is indexed under the same physical case.
Generation requires the original qualified producer runtime and numerical QC.
"""
import json,fcntl
from pathlib import Path
import numpy as np
import torch
from ..core import require,digest
from ..artifact_io import write_artifact
from ..smooth.data import Case
from ..smooth.certification import authorize,normalizers
from ..unified.config import DatasetConfig
from ..unified.streams import coordinates_id
from ..unified.dataset import quality_check_full
from .portable import read_index,verify_folder,commit,add_artifact
from .numerics import FastReference

def prepare_label(manifest,request,certificate):
    path,index=verify_folder(manifest);r=json.loads(Path(request).read_text());config=DatasetConfig(**index['config'])
    require(r['schema']=='G1-exact-missing-label-request-v1' and r['dataset_identity']==config.identity and r['producer_certificate']==index['certificate_id'],'missing-label request belongs to a different dataset/producer')
    require(certificate['certification_id']==index['certificate_id'],'offline supplement requires original producer runtime/certificate; existing references are never relabeled')
    authorize(certificate,pdes=config.pdes,families=config.families,normalizer_identity=index['normalizer_identity'])
    pde,family,split,i=r['case_key'];c=Case(family,pde,split,i,config.conditioning_h)
    require(split=='train' and c.case_id==r['case_id'] and c.case_id in index['records'],'optimizer-preflight repair requires an existing TRAIN case')
    x=np.asarray(r['position'],np.float64);t=np.asarray(r['physical_time'],np.float64);q=np.asarray(r['quantity_id'],np.int64);key=coordinates_id(x,t,q)
    require(key==r['coordinates_id'] and len(x)==r['points'],'exact requested coordinate checksum differs')
    with (path.parent/'.preparation.lock').open('a+') as guard:
        try:fcntl.flock(guard.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('dataset preparation coordinator is active; do not repair concurrently') from None
        path,index=read_index(path)
        if key in index.get('supplemental_labels',{}).get(c.case_id,{}):return {'status':'already_prepared','reference_solves':0}
        full=FastReference(c);qc=quality_check_full(c,full,normalizers(certificate),config,finer=False);require(qc['passed'],'offline supplement numerical QC failed')
        y=full.evaluate(x,t,q);require(y.dtype==np.float64 and np.isfinite(y).all(),'supplement needs finite CPU64 labels')
        root=path.parent/'reference_cache/supplements';stem=digest((config.identity,c.case_id,key,certificate['certification_id']))
        record=write_artifact(root,stem,{'artifact_type':'diagnostic_bundle','architecture_id':'REFERENCE_ONLY','run_id':'G1-explicit-exact-label-repair','dataset_identity':config.identity,'case_id':c.case_id,'coordinates_id':key,'certificate_id':certificate['certification_id'],'execution_source':config.reference_execution_source,'normalizer_identity':index['normalizer_identity'],'physics':certificate['source_revision']},{'position':torch.from_numpy(x),'physical_time':torch.from_numpy(t),'quantity_id':torch.from_numpy(q),'labels':torch.from_numpy(y),'qc':qc},required=('position','physical_time','quantity_id','labels','qc'),idempotent=True)
        add_artifact(index,path.parent,record);index.setdefault('supplemental_labels',{}).setdefault(c.case_id,{})[key]=record.relative_to(path.parent).as_posix();commit(path,index)
        return {'status':'prepared','artifact':str(record),'coordinates_id':key,'points':len(x),'physical_cases_solved':1,'original_case_artifacts_modified':False}
