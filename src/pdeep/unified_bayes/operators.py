"""Full-reference, geometry-specific public C3 sensitivities; no hidden case draws.

Prepare ALL requested sensor/query designs together. Existing Reference.dynamic
propagates32/64 RHS simultaneously, reusing its factorization and continuous C1
output. Cache evaluated unit responses, NOT dense per-basis space-time trajectories.
New arbitrary continuous designs can be added explicitly; no solve per observation.
"""
from dataclasses import dataclass
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
import hashlib,time
import numpy as np
import torch
from ..core import digest,require
from ..artifact_io import read_payload,read_manifest,write_artifact,sha256
from ..smooth.definition import PDES,GEOMETRIES,POLICY,COEFFICIENTS,geometry,prior_version,physics_identity
from ..smooth.reference import reference
from ..smooth.certification import authorize,normalizers
from ..wave_fidelity.fields import centers,SmoothField

SCHEMA='G1-public-C3-full-reference-sensitivities-v1'
ROLES={'poisson':('f',),'heat':('u0',),'wave':('u0','v0')}


def prior_specification(pde,family):
    require(pde in PDES and family in GEOMETRIES,'unsupported G1 PDE/geometry')
    c,r=centers(geometry(family,.08),relative_radius=.18,count=32);roles=ROLES[pde];d=32*len(roles)
    body={'schema':'public-generative-G1-C3-Gaussian-v1','pde':pde,'family':family,'geometry_id':geometry(family,.08).geometry_id,
          'prior_version':prior_version(pde),'coefficient_order':[(role,j) for role in roles for j in range(32)],
          'mean':[0.]*d,'covariance':'identity','coefficient_correlations':'independent; wave u0/v0 blocks independent',
          'physical_basis':'max(0,1-||x-c_j||²/r²)^4 / sqrt32','centers':c.tolist(),'radius':float(r),'amplitude':1.,
          'field_correlation':'overlapping spatial basis; Gaussian coefficients are NOT correlated',
          'truncation_or_nonlinear_transform':'none; physical finite32 basis is the actual benchmark, not an added cutoff',
          'source_of_prior':'public frozen simulator specification; no training/validation/test moments fitted'}
    return {**body,'dimension':d,'identity':digest(body)}


@dataclass(frozen=True)
class Coordinates:
    position: np.ndarray
    physical_time: np.ndarray | None
    quantity: np.ndarray
    def validate(self,pde,family):
        x=np.asarray(self.position,dtype=np.float64);q=np.asarray(self.quantity)
        require(x.ndim==2 and x.shape[1]==2 and q.shape==(len(x),) and np.isfinite(x).all(),'invalid physical coordinates/quantities')
        require(np.isin(q,[0,1] if pde=='wave' else [0]).all(),'quantity not supported for this PDE')
        require((self.physical_time is None)==(pde=='poisson'),'Poisson physical time must be ABSENT; dynamics require time, including zero')
        if self.physical_time is not None:
            t=np.asarray(self.physical_time);require(t.shape==(len(x),) and np.isfinite(t).all() and ((0<=t)&(t<=1)).all(),'invalid continuous time')
        geometry(family,.08).locate(x)
        return self
    @property
    def time_storage(self):return np.zeros(len(self.position)) if self.physical_time is None else np.asarray(self.physical_time,dtype=np.float64)
    @property
    def identity(self):
        h=hashlib.sha256();h.update(b'absent' if self.physical_time is None else b'physical-time-present')
        for a,dtype in ((self.position,np.float64),(self.time_storage,np.float64),(self.quantity,np.int64)):
            a=np.ascontiguousarray(a,dtype=dtype);h.update(str(a.shape).encode());h.update(a.tobytes())
        return h.hexdigest()
    def slice(self,s):return Coordinates(self.position[s],None if self.physical_time is None else self.physical_time[s],self.quantity[s])
    @classmethod
    def design(cls,design):
        x,t,q,present=design;return cls(np.asarray(x,dtype=np.float64),np.asarray(t,dtype=np.float64) if present else None,np.asarray(q,dtype=np.int64))


def _basis_fields(spec):
    c=np.asarray(spec['centers']);r=spec['radius'];roles=ROLES[spec['pde']];d=spec['dimension'];out={role:[] for role in roles}
    for k in range(d):
        for block,role in enumerate(roles):
            z=np.zeros(32)
            if k//32==block:z[k%32]=1.
            out[role].append(SmoothField(c,r,z,spec['geometry_id']))
    return out


class ForwardOperators:
    def __init__(self,root,certificate,*,progress=None,persist=None,max_loaded_bytes=256*1024**2,_reference_factory=None,_control_identity=None):
        self.root=Path(root);self.certificate=certificate;self.norms=normalizers(certificate);self.progress=progress or (lambda *a,**k:None);self.persist=persist
        require((_reference_factory is None)==(_control_identity is None),'numerical control factory must have a separate explicit identity')
        self.factory=_reference_factory or (lambda family:reference(family,.02,4));self.control=_control_identity
        self.physics=physics_identity();self.specifications={};self.loaded=OrderedDict();self.max_loaded_bytes=max_loaded_bytes;self.statistics={'forward_batches':0,'design_hits':0,'designs_prepared':0,'preprocessing_seconds':0.}
    def spec(self,pde,family):
        if (pde,family) in self.specifications:return self.specifications[(pde,family)]
        prior=prior_specification(pde,family)
        if self.control is None:
            authorize(self.certificate,pdes=(pde,),families=(family,),normalizer_identity=self.norms.identity)
            from ..smooth.resolution import resolve_reference
            _,resolved,_=resolve_reference(family,h=.02,degree=4);resolved=resolved.asdict()
        else:resolved={'scope':'NONPRODUCTION numerical control','identity':self.control}
        body={'schema':SCHEMA,'prior_identity':prior['identity'],'pde':pde,'family':family,'coefficient':COEFFICIENTS[pde],
              'boundary':'tagged homogeneous Dirichlet','known_dynamic_source':0.,'affine_offset':'identically zero for this benchmark',
              'policy':POLICY,'resolved_reference':resolved,'physics':self.physics,'certificate_id':self.certificate['certification_id'],
              'implementation':sha256(Path(__file__)),'control':self.control}
        result={**body,'identity':digest(body),'prior':prior};self.specifications[(pde,family)]=result;return result
    def path(self,pde,family,coordinates):return self.root/(digest((self.spec(pde,family)['identity'],coordinates.identity))+'.json')
    def _forward(self,r,pde,fields,coords):
        x,t,q=coords.position,coords.time_storage,coords.quantity
        if pde=='poisson':U,diag=r.poisson(fields['f'],x,8);return U.T,diag
        U,V,diag=r.dynamic(pde,fields['u0'],fields.get('v0'),x,t,.002)
        return (np.where(q[None]==1,V,U) if V is not None else U).T,diag
    def prepare(self,pde,family,designs,*,verify_superposition=True):
        """Manual CPU64 preparation, one full batched evolution per task/new design set."""
        start=time.perf_counter();require(designs,'at least one design needed');spec=self.spec(pde,family)
        if self.control is None:authorize(self.certificate,pdes=(pde,),families=(family,),normalizer_identity=self.norms.identity)
        missing={}
        for name,c in designs.items():
            c.validate(pde,family);path=self.path(pde,family,c)
            if path.exists():self.get(pde,family,c);self.statistics['design_hits']+=1
            else:missing[c.identity]=c
        if not missing:return {'cached':True,'new_designs':0,'operator_identity':spec['identity']}
        self.progress('Preparing references',pde=pde,geometry=family,basis_dimension=spec['prior']['dimension'],new_designs=len(missing),operation='full batched unit responses',cached_forward_operators=len(self.loaded))
        unique=list(missing.values());lengths=[len(c.position) for c in unique]
        require(all(n>0 for n in lengths),'empty physical designs are handled by prior-only algebra, not a PDE solve')
        joined=Coordinates(np.concatenate([c.position for c in unique]),None if pde=='poisson' else np.concatenate([c.physical_time for c in unique]),np.concatenate([c.quantity for c in unique]))
        r=self.factory(family);fields=_basis_fields(spec['prior']);H,diagnostics=self._forward(r,pde,fields,joined);self.statistics['forward_batches']+=1
        require(H.shape==(sum(lengths),spec['prior']['dimension']) and H.dtype==np.float64 and np.isfinite(H).all(),'invalid numerical sensitivities')
        verification={'performed':False}
        if verify_superposition:
            # Independent combined-field solve, reserved diagnostic coefficients;
            # never the hidden coefficients of an evaluation physical case.
            z=np.random.default_rng(7193).standard_normal(spec['prior']['dimension']);take=np.unique(np.linspace(0,len(H)-1,min(128,len(H)),dtype=int));check=joined.slice(take)
            combined={}
            for block,role in enumerate(ROLES[pde]):combined[role]=[SmoothField(np.asarray(spec['prior']['centers']),spec['prior']['radius'],z[block*32:(block+1)*32],spec['prior']['geometry_id'])]
            independent,_=self._forward(r,pde,combined,check);error=H[take]@z-independent[:,0];scale=np.array([self.norms.scale(pde,int(q)) for q in check.quantity])
            maximum=float(np.max(abs(error)));normalized=float(np.max(abs(error)/scale));require(normalized<1e-8,'full-reference superposition failed')
            verification={'performed':True,'maximum_physical_error':maximum,'maximum_training_RMS_error':normalized,'scope':'independent full combined-field solve at up to128 points; not continuum certification'}
        seconds=time.perf_counter()-start;offset=0;paths=[]
        for c,n in zip(unique,lengths):
            path=self.path(pde,family,c);payload={'H':torch.from_numpy(H[offset:offset+n].copy()),'offset':torch.zeros(n,dtype=torch.float64),
                'position':torch.from_numpy(c.position.copy()),'physical_time':None if c.physical_time is None else torch.from_numpy(c.physical_time.copy()),'quantity':torch.from_numpy(c.quantity.copy())}
            metadata={'artifact_type':'diagnostic_bundle','architecture_id':'G1_BAYES_OPERATOR','run_id':'G1-full-reference-operators','scientific_schema':SCHEMA,
                'operator_spec':spec,'coordinate_identity':c.identity,'numerical_diagnostics':diagnostics,'superposition':verification,
                'preprocessing_batch_identity':digest((spec['identity'],list(missing))),'batch_preprocessing_seconds':seconds,'new_designs_in_batch':len(unique),
                'exactness':'exact Gaussian posterior for this FULL discretized P4/CN/Gauss4-C1 forward map; no response compression' if self.control is None else 'NONPRODUCTION numerical control '+str(self.control)}
            write_artifact(path.parent,path.stem,metadata,payload,required=tuple(payload),idempotent=True);paths.append(str(path));offset+=n
            if self.persist:self.persist(path)
        total_seconds=time.perf_counter()-start
        batch=digest((spec['identity'],list(missing)))
        ledger={'batch_identity':batch,'operator_identity':spec['identity'],'numerical_preprocessing_seconds':seconds,'end_to_end_preparation_seconds':total_seconds,'artifact_write_and_sync_seconds':total_seconds-seconds,'new_designs':len(unique),'scope':'includes mesh/assembly, batched solves, superposition, response artifact writes and their verified sync; excludes this small ledger write'}
        ledger_path=write_artifact(self.root/'preparation_batches',batch,{'artifact_type':'diagnostic_bundle','architecture_id':'G1_BAYES_PREPARATION','run_id':'G1-full-reference-operators'},ledger,required=('batch_identity',),idempotent=True)
        if self.persist:self.persist(ledger_path)
        self.statistics['designs_prepared']+=len(unique);self.statistics['preprocessing_seconds']+=seconds
        return {'cached':False,'new_designs':len(unique),'seconds':seconds,'end_to_end_seconds':total_seconds,'cost_ledger':str(ledger_path),'verification':verification,'paths':paths,'operator_identity':spec['identity']}
    def preparation_cost(self,batch):
        _,ledger=read_payload(self.root/'preparation_batches'/(batch+'.json'),kind='diagnostic_bundle',architecture='G1_BAYES_PREPARATION');require(ledger['batch_identity']==batch,'preparation ledger mismatch');return ledger
    def get(self,pde,family,c):
        key=(pde,family,c.identity)
        if key not in self.loaded:
            path=self.path(pde,family,c);require(path.is_file(),'Full-reference Bayesian design missing. Explicitly prepare selected task designs; no T1 sine or approximate-cache fallback.')
            r,p=read_payload(path,kind='diagnostic_bundle',architecture='G1_BAYES_OPERATOR',required=('H','offset'))
            require(r['scientific_schema']==SCHEMA and r['operator_spec']['identity']==self.spec(pde,family)['identity'] and r['coordinate_identity']==c.identity,'operator/source/geometry identity mismatch')
            H,b=p['H'].numpy(),p['offset'].numpy()
            require(H.dtype==b.dtype==np.float64 and H.shape==(len(c.position),r['operator_spec']['prior']['dimension']) and b.shape==(len(c.position),) and np.isfinite(H).all() and np.isfinite(b).all(),'invalid cached forward tensors')
            actual=Coordinates(p['position'].numpy(),None if p['physical_time'] is None else p['physical_time'].numpy(),p['quantity'].numpy());require(actual.identity==c.identity,'serialized operator coordinate mismatch')
            self.loaded[key]=(H,b,r)
            while len(self.loaded)>1 and sum(a.nbytes+b.nbytes for a,b,_ in self.loaded.values())>self.max_loaded_bytes:self.loaded.popitem(last=False)
        self.loaded.move_to_end(key)
        return self.loaded[key]
