"""Verified byte-exact numerical inputs, not replacement identity strings.

The shipped arrays ARE the meshes originally used for certification. Loading
recomputes geometry/mesh identities and validates topology, normals and quality.
Unsupported shapes/resolutions continue through the original mesher. Ideal curved
shape parameters + facet/tag counts select artifacts; requested ring coordinates
must also agree within roundoff. No physical geometry is silently substituted.
"""
from functools import lru_cache
from pathlib import Path
import hashlib,json
from ..core import digest,require

ROOT=Path(__file__).resolve().parents[3]/'certifications/G1_smooth_v1/portable'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def mesh_key(shape,h):
    return digest({'geometry':shape.geometry_id,'rings':[len(shape.outer),*[len(r) for r in shape.holes]],'tags':list(shape.tags),'h':float(h)})

@lru_cache(maxsize=1)
def index():
    path=ROOT/'index.json'
    if not path.exists():return None
    record=json.loads(path.read_text());require(record['identity']==digest({k:v for k,v in record.items() if k!='identity'}),'portable numerical-input index checksum mismatch')
    return record

@lru_cache(maxsize=48)
def _load(name,checksum):
    from .domain import GeometryDomain
    import numpy as np
    path=ROOT/name;require(Path(name).name==name and sha(path)==checksum,'certified mesh artifact file checksum mismatch')
    mesh=GeometryDomain.load(path)
    for v in (mesh.vertices,mesh.cells,mesh.boundary_edges,mesh.boundary_tags,mesh.region_tags):v.setflags(write=False)
    return mesh

def stored_mesh(shape,h):
    import numpy as np
    catalog=index()
    if catalog is None:return None
    item=catalog['meshes'].get(mesh_key(shape,h))
    if item is None:return None
    mesh=_load(item['file'],item['sha256'])
    require(mesh.mesh_id==item['mesh_id'] and mesh.geometry_id==shape.geometry_id,'certified mesh payload identity mismatch')
    for requested,frozen in zip((shape.outer,*shape.holes),(mesh.shape.outer,*mesh.shape.holes)):
        require(np.asarray(requested).shape==np.asarray(frozen).shape and np.allclose(requested,frozen,atol=32*np.finfo(float).eps,rtol=0),'requested represented polygon differs from certified polygon beyond roundoff')
    require(tuple(shape.tags)==tuple(mesh.shape.tags),'requested boundary tags differ from certified mesh')
    return mesh

def stored_basis(mesh,relative_radius,count):
    import numpy as np
    catalog=index()
    if catalog is None or relative_radius!=.18 or count!=32:return None
    item=catalog['families'].get(mesh.shape.family)
    if item is None or item['canonical_mesh_id']!=mesh.mesh_id:return None
    path=ROOT/item['basis_file'];require(sha(path)==item['basis_sha256'],'C3 basis artifact checksum mismatch')
    with np.load(path,allow_pickle=False) as data:centers=data['centers'].copy();radius=float(data['radius'])
    require(centers.shape==(32,2) and np.isfinite(centers).all() and radius>0,'invalid canonical C3 basis')
    return centers,radius
