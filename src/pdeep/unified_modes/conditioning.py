"""Public, sanitized C3 field information. Mode labels are never neural features."""
from dataclasses import dataclass,replace
import numpy as np
import torch
from ..core import require,digest
from ..unified_bayes.operators import prior_specification,ROLES

SCHEMA='G1-explicit-C3-field-information-v1'
ROLES_ALL=('f','u0','v0')
MODES=('FORWARD','INVERSE','HYBRID')

@dataclass(frozen=True)
class FieldCondition:
    pde:str
    family:str
    quantity:np.ndarray
    component:np.ndarray
    centers:np.ndarray
    radius:np.ndarray
    values:np.ndarray
    applicable:np.ndarray
    available:np.ndarray
    valid:np.ndarray
    geometry_identity:str
    representation:str=SCHEMA
    def __post_init__(self):
        require(self.representation==SCHEMA,'only exact public G1 C3 basis coefficients supported; arbitrary samples/functions are not complete known fields')
        spec=prior_specification(self.pde,self.family);require(self.geometry_identity==spec['geometry_id'],'field geometry identity differs')
        for key in ('available','applicable','valid'):require(np.asarray(getattr(self,key)).dtype==np.bool_,'explicit boolean availability/applicability/valid masks required')
        require(np.issubdtype(self.quantity.dtype,np.integer) and np.isin(self.quantity,[0,1,2]).all() and np.issubdtype(self.component.dtype,np.integer),'typed integer field quantity/component IDs required')
        n=len(self.quantity)
        for key in ('quantity','component','radius','values','applicable','available','valid'):require(np.asarray(getattr(self,key)).shape==(n,),'field token shape mismatch: '+key)
        require(self.centers.shape==(n,2),'centers must retain physical coordinates')
        require(not np.any(self.available&~self.applicable),'not-applicable inputs cannot be available')
        require(np.isfinite(self.centers).all() and np.isfinite(self.radius).all() and (self.radius>0).all(),'invalid physical basis metadata')
        safe=np.where(self.valid&self.applicable&self.available,self.values,0.).astype(np.float64)
        require(np.isfinite(safe).all(),'known values must be finite exact coefficients')
        # Scrub BEFORE encoding, hashing or logging. NaN*0 is not sanitization.
        for key in ('quantity','component','centers','radius','applicable','available','valid'):
            a=np.array(getattr(self,key),copy=True);a.flags.writeable=False;object.__setattr__(self,key,a)
        safe.flags.writeable=False;object.__setattr__(self,'values',safe)
        for role in ROLES[self.pde]:
            idx=self.valid&(self.quantity==ROLES_ALL.index(role));require(set(self.component[idx])==set(range(32)) and idx.sum()==32,'each applicable role needs the complete public 32-component basis metadata')
            require(np.array_equal(self.centers[idx],np.asarray(spec['centers'])[self.component[idx]]) and np.all(self.radius[idx]==spec['radius']),'only the frozen public C3 basis is supported; incompatible center/radius metadata')
        require(np.array_equal(self.applicable,np.isin(self.quantity,[ROLES_ALL.index(r) for r in ROLES[self.pde]])&self.valid),'PDE/role applicability mismatch')
    @property
    def known_fraction(self):return float(self.available[self.valid&self.applicable].mean())
    @property
    def complete(self):return self.known_fraction==1.
    @property
    def status(self):return np.where(self.applicable,np.where(self.available,2,1),0) # N/A, unknown, known
    @property
    def identity(self):return digest((SCHEMA,self.pde,self.family,self.geometry_identity,[(k,np.asarray(getattr(self,k)).tolist()) for k in ('quantity','component','centers','radius','values','applicable','available','valid')]))
    def permute(self,order):return replace(self,**{k:getattr(self,k)[order] for k in ('quantity','component','centers','radius','values','applicable','available','valid')})
    def coefficients(self):
        vals=[];known=[]
        for role in ROLES[self.pde]:
            ix=np.flatnonzero(self.valid&(self.quantity==ROLES_ALL.index(role)));ix=ix[np.argsort(self.component[ix])];vals.extend(self.values[ix]);known.extend(self.available[ix])
        return np.array(vals),np.array(known,bool)
    @classmethod
    def from_coefficients(cls,pde,family,values,available):
        spec=prior_specification(pde,family);d=spec['dimension'];values=np.asarray(values);raw_mask=np.asarray(available);require(np.isin(raw_mask,[0,1]).all(),'exact knowledge requires boolean availability, not uncertain probabilities');available=raw_mask.astype(bool)
        require(values.shape==available.shape==(d,),'coefficient vector/mask dimension mismatch')
        q=np.repeat(np.arange(3),32);app=np.isin(q,[ROLES_ALL.index(r) for r in ROLES[pde]]);v=np.zeros(96);m=np.zeros(96,bool)
        for i,role in enumerate(ROLES[pde]):idx=q==ROLES_ALL.index(role);m[idx]=available[32*i:32*(i+1)];v[idx]=np.where(m[idx],values[32*i:32*(i+1)],0.)
        return cls(pde,family,q,np.tile(np.arange(32),3),np.tile(spec['centers'],(3,1)),np.full(96,spec['radius']),v,app,m,np.ones(96,bool),spec['geometry_id'])
    @classmethod
    def from_case(cls,c,available):
        # Dataset view construction only: only disclosed indices leave this function.
        a=np.concatenate([c.fields[r].coefficients for r in ROLES[c.pde]])
        return cls.from_coefficients(c.pde,c.family,a,available)

@dataclass(frozen=True)
class Problem:
    base:object
    field_condition:FieldCondition
    def __getattr__(self,name):return getattr(self.base,name)
    @property
    def identity(self):return self.base.identity # Typed operator is unchanged; no case-specific graph cache.

@dataclass(frozen=True)
class CaseView:
    base:object
    condition:FieldCondition
    mode:str
    conditioning_h:float|None=None
    def __post_init__(self):
        require(self.mode in MODES,'unknown reporting mode')
        if self.conditioning_h is None:object.__setattr__(self,'conditioning_h',self.base.conditioning_h)
        elif self.conditioning_h!=self.base.conditioning_h:object.__setattr__(self,'base',replace(self.base,conditioning_h=self.conditioning_h))
    def __getattr__(self,name):return getattr(self.base,name)
    @property
    def problem(self):return Problem(self.base.problem,self.condition)


def mask_for(pde,mode,rng,*,fraction=None,pattern=None):
    d=64 if pde=='wave' else 32
    if mode=='FORWARD':return np.ones(d,bool),'all'
    if mode=='INVERSE':return np.zeros(d,bool),'none'
    fraction=float(rng.choice((.25,.5,.75))) if fraction is None else fraction
    pattern=pattern or str(rng.choice(('random','group','u0','v0') if pde=='wave' else ('random','group')))
    mask=np.zeros(d,bool);n=int(round(d*fraction))
    if pattern in ('u0','v0'):
        require(pde=='wave','whole-field masks apply to wave only');mask[:32] = pattern=='u0';mask[32:] = pattern=='v0'
    elif pattern=='group':mask[:n]=True
    else:mask[rng.choice(d,n,replace=False)]=True
    return mask,pattern
