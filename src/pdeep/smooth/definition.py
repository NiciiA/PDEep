"""Frozen field laws and physical identities; no neural or optimizer imports.

Poisson/heat extension was explicitly authorized in the finalization request.
The existing wave sampler is called unchanged, including its original RNG version.
"""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path
import numpy as np
from ..core import digest,require
from ..geometric.domain import make_shape,mesh_shape
from ..wave_fidelity.fields import SmoothField,centers,pair,VERSION as C3_ID
from ..wave_fidelity.study import FAMILIES

BENCHMARK_ID='G1_smooth_v1'
GENERATOR_VERSION='G1-smooth-generator-1'
EXTENSION_VERSION='G1-C3-authorized-source-and-heat-extension-1'
GEOMETRY_VERSION='GeometryDomain-v1'
SOLVER_VERSION='consistent-P4-Gauss4-wave-CN-heat-elliptic-1'
BOUNDARY_VERSION='tagged-homogeneous-Dirichlet-1'
OBSERVATION_VERSION='area-uniform-independent-heteroscedastic-1'
NORMALIZATION_VERSION='training-only-frozen-C3-extension-RMS-1'
PDES=('poisson','heat','wave')
SEEN=('square','rectangle','lshape','disk')
HELD_OUT=('ellipse','hole','annulus','polygon')
SPLIT_COUNTS={'train':12000,'validation':1024,'test':1024,'geometry_held_out':1024}
TIMES=(0.,.0073,.02,.05,.1,.237,.5,.75,1.)
# Use exact repository identifiers/parameters, not inferred geometry categories.
GEOMETRIES={family:{'family':family,'aspect':aspect} for family,aspect,_,_ in FAMILIES}
COEFFICIENTS={'poisson':1.,'heat':.02,'wave':.49}
POLICY={'reference_h':.02,'reference_degree':4,'reference_dt':.002,'source_quadrature':8,
        'boundary_policy':BOUNDARY_VERSION,'observation_policy':OBSERVATION_VERSION,
        'relative_radius':.18,'basis_count':32,'physical_time_interval':[0.,1.],
        'wave_time_method':'Gauss4-C1-Hermite-v=du_dt','heat_time_method':'Crank-Nicolson-C1-Hermite',
        'spatial_interpolation':'conforming Lagrange degree4; represented polygon fixed64 facets'}


def random_stream(*key):return np.random.default_rng(int(digest((GENERATOR_VERSION,key))[:16],16))


@lru_cache(maxsize=32)
def geometry(family,h=.08):
    require(family in GEOMETRIES,'unknown smooth geometry identifier')
    return mesh_shape(make_shape(family,aspect=GEOMETRIES[family]['aspect']),h)


@dataclass(frozen=True)
class ExtendedField(SmoothField):
    role: str='f'
    @property
    def identity(self):return digest((EXTENSION_VERSION,self.role,super().identity))
    @property
    def specification(self):return {**super().specification,'version':EXTENSION_VERSION,'role':self.role,'field_identity':self.identity}


def physical_fields(family,pde,split,index):
    require(pde in PDES,'unsupported smooth PDE')
    mesh=geometry(family,.08)
    if pde=='wave':
        u,v=pair(mesh,(split,int(index)),relative_radius=.18,count=32)
        return {'u0':u,'v0':v} # EXACT existing C3 call; version/coefficients unchanged.
    role='f' if pde=='poisson' else 'u0';c,r=centers(mesh,relative_radius=.18,count=32)
    z=random_stream(EXTENSION_VERSION,mesh.geometry_id,pde,split,int(index),role).standard_normal(32)
    return {role:ExtendedField(c,r,z,mesh.geometry_id,1.,role)}


def physical_id(family,pde,split,index):
    require(family in GEOMETRIES and pde in PDES and split in {*SPLIT_COUNTS,'calibration'},'invalid smooth physical identity')
    require(type(index) is int and index>=0 and (split=='calibration' or index<SPLIT_COUNTS[split]),'invalid physical case index')
    return digest((BENCHMARK_ID,GENERATOR_VERSION,geometry(family).geometry_id,pde,split,index,prior_version(pde)))


def prior_version(pde):return C3_ID if pde=='wave' else EXTENSION_VERSION


def split_manifest():
    body={'benchmark_id':BENCHMARK_ID,'generator_version':GENERATOR_VERSION,'counts_per_PDE':SPLIT_COUNTS,
          'train_geometry_instances':[geometry(g).geometry_id for g in SEEN],
          'validation_test_geometry_instances':[geometry(g).geometry_id for g in SEEN],
          'geometry_held_out_instances':[geometry(g).geometry_id for g in HELD_OUT],
          'case_key':'benchmark/PDE/physical split/index/geometry/prior; independent of propagation mesh',
          'field_rng':'unchanged C3 wave pair(split,index); separately keyed authorized Poisson/heat extension',
          'mesh_transfer':'same physical ID and split; never called geometry generalization',
          'instance_transfer':'e.g. rectangle1.4 is a new instance, not currently certified',
          'family_topology_transfer':'ellipse/hole/annulus/polygon; distinct from ID validation',
          'calibration':'independent reserved calibration keys; no validation/test/held-out geometry calibration'}
    return {**body,'identity':digest(body)}


def physics_identity():
    import scipy,triangle,shapely
    root=Path(__file__).resolve().parents[3]
    names=('src/pdeep/geometric/domain.py','src/pdeep/geometric/quadrature.py','src/pdeep/wave_fidelity/fields.py','src/pdeep/wave_fidelity/reference.py',
           'src/pdeep/smooth/definition.py','src/pdeep/smooth/reference.py','src/pdeep/smooth/data.py')
    files={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names}
    versions={'numpy':np.__version__,'scipy':scipy.__version__,'triangle':triangle.__version__,'shapely':shapely.__version__}
    return {'files':files,'dependencies':versions,'scientific_sha':digest((files,versions))}


def manifest(certification=None):
    from ..diagnostics import source_identity
    root=Path(__file__).resolve().parents[3]
    return {'benchmark_id':BENCHMARK_ID,'generator_version':GENERATOR_VERSION,'field_prior_version':{p:prior_version(p) for p in PDES},
            'legacy_mapping':{C3_ID:{'canonical_benchmark':BENCHMARK_ID,'scope':'wave prior only; old Poisson/heat were undefined'},'G1':'original canonical-P1 prior; unchanged and blocked'},
            'geometry_schema_version':GEOMETRY_VERSION,'FEM_solver_version':SOLVER_VERSION,'boundary_condition_version':BOUNDARY_VERSION,
            'observation_policy_version':OBSERVATION_VERSION,'normalization_version':NORMALIZATION_VERSION,'source_revision':physics_identity(),
            'git_revision':source_identity(root)['git_revision'],'certification_id':None if certification is None else certification['certification_id'],
            'certification_status':'missing' if certification is None else certification['pass_fail_status'],'reference_policy':POLICY,'splits':split_manifest()}
