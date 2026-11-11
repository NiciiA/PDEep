"""Canonical resolved reference specification shared by checks and evaluation."""
from dataclasses import dataclass,asdict
from .definition import POLICY,geometry,GEOMETRIES

@dataclass(frozen=True)
class ResolvedReference:
    family: str
    h: float
    degree: int
    dt: float
    source_quadrature: int
    boundary_policy: str
    geometry_id: str
    polygon_id: str
    mesh_id: str
    aspect: float
    mesher_options: str
    def asdict(self):return asdict(self)

def resolve_reference(family,*,h=None,degree=None,policy=None):
    p=POLICY if policy is None else policy;h=p['reference_h'] if h is None else h;degree=p['reference_degree'] if degree is None else degree
    m=geometry(family,float(h))
    from ..geometric.portable import mesh_key,index
    catalog=index();origin='certified artifact' if catalog is not None and mesh_key(m.shape,h) in catalog['meshes'] else 'newly generated; numerical qualification required'
    spec=ResolvedReference(family,float(h),int(degree),float(p['reference_dt']),int(p['source_quadrature']),p['boundary_policy'],m.geometry_id,m.shape.polygon_id,m.mesh_id,float(GEOMETRIES[family]['aspect']),f'pq28a{h*h/2:.16f}Q')
    return m,spec,origin

def report_selection(record,families,*,progress=print,policy=None):
    rows=[]
    for family in families:
        m,selected,origin=resolve_reference(family,policy=policy);expected=record.get('resolved_meshes',{}).get(family)
        if expected is None:
            old=record['mesh_records'][family]['production'];_,baseline,_=resolve_reference(family,policy=record['reference_policy']);expected={**baseline.asdict(),**{k:old[k] for k in ('h','degree','geometry_id','polygon_id','mesh_id')}}
        differences={k:{'selected':v,'certified':expected.get(k)} for k,v in selected.asdict().items() if v!=expected.get(k)}
        row={'family':family,'selected':selected.asdict(),'certified':expected,'differing_fields':differences,'origin':origin};rows.append(row)
        progress(f"Reference {family}: selected={m.mesh_id}; certified={expected['mesh_id']}; h={selected.h}, P{selected.degree}, dt={selected.dt}; {origin}; differing fields={list(differences)}")
    return rows
