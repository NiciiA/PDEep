"""Tagged conforming triangular geometry; true shape and polygon mesh are distinct.

SDF is positive inside the represented polygon. It is Lipschitz, not globally C1.
Triangle/Shapely are optional CPU preprocessing dependencies, never neural solvers.
"""
from dataclasses import dataclass
from functools import cached_property
import json,math
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components,dijkstra
from ..core import require,digest


def dependencies():
    try:
        import shapely,triangle
    except ImportError as e:raise RuntimeError('Geometry preprocessing requires pip install -e ".[geometry]" (binary Triangle + Shapely).') from e
    return shapely,triangle


@dataclass(frozen=True)
class Shape:
    family: str
    outer: tuple
    holes: tuple=()
    tags: tuple=()
    seed: int=0
    intended: tuple=()

    @cached_property
    def polygon(self):
        shapely,_=dependencies();p=shapely.Polygon(self.outer,self.holes)
        require(p.is_valid and p.area>0 and p.geom_type=='Polygon','invalid/self-intersecting/disconnected polygon')
        require(p.exterior.is_ccw and all(not r.is_ccw for r in p.interiors),'outer ring must be CCW and holes CW')
        require(len(self.tags)==sum(len(r) for r in (self.outer,*self.holes)) and min(self.tags)>0,'complete positive boundary tags required')
        return p

    @property
    def geometry_id(self):
        # Physical geometry is independent of sampling seed, ring starting
        # vertex, boundary labels, mesh resolution and the family display name.
        def canonical(r):
            r=tuple(r);return min(r[i:]+r[:i] for i in range(len(r)))
        return digest({'schema':'GeometryDomain-v1','shape':self.intended or
                       (canonical(self.outer),tuple(sorted(canonical(r) for r in self.holes)))})
    @property
    def geometry_error(self):
        """Polygonal approximation budget, distinct from FEM discretization."""
        if not self.intended:return {'exact_polygon':True,'relative_area_error':0.,'max_boundary_chord_error_bound':0.}
        kind,*v=self.intended
        if kind in {'disk','ellipse'}:area=np.pi*v[2]*v[3];radius=max(v[2:]);n=len(self.outer)
        elif kind=='annulus':area=np.pi*(v[2]**2-v[3]**2);radius=v[2];n=len(self.outer)
        else:area=1-np.pi*v[2]**2;radius=v[2];n=len(self.holes[0])
        return {'exact_polygon':False,'relative_area_error':abs(self.polygon.area-area)/area,
                'max_boundary_chord_error_bound':radius*(1-np.cos(np.pi/n))}
    @property
    def polygon_id(self):return digest((self.outer,self.holes,self.tags))
    @property
    def topology_id(self):return digest(('connected_planar_domain',len(self.holes)))


def polygon_shape(outer,holes=(),*,tags=None,seed=0,family='polygon',intended=()):
    shapely,_=dependencies()
    raw=shapely.Polygon(outer,holes)
    require(raw.is_valid and raw.area>0,'invalid polygon; no automatic repair')
    # Canonical ring orientation preserves supplied segment tags by endpoint pair.
    if tags is not None:
        require(len(tags)==sum(len(r) for r in (outer,*holes)),'boundary tag count mismatch')
        mapping={};offset=0
        for ring in (outer,*holes):
            for i,a in enumerate(ring):mapping[frozenset((tuple(a),tuple(ring[(i+1)%len(ring)])))]=int(tags[offset+i])
            offset+=len(ring)
    from shapely.geometry.polygon import orient
    p=orient(raw,sign=1.);rings=[tuple(tuple(float(v) for v in x) for x in list(p.exterior.coords)[:-1])]
    rings.extend(tuple(tuple(float(v) for v in x) for x in list(r.coords)[:-1]) for r in p.interiors)
    values=[]
    for k,ring in enumerate(rings):
        values.extend(mapping[frozenset((a,ring[(i+1)%len(ring)]))] if tags is not None else k+1 for i,a in enumerate(ring))
    return Shape(family,rings[0],tuple(rings[1:]),tuple(values),seed,tuple(intended))


def make_shape(family='square',*,seed=0,aspect=1.5,segments=64):
    require(segments>=12,'curved boundaries need at least 12 facets')
    theta=np.arange(segments)*2*np.pi/segments
    circle=lambda radius,c=(.5,.5),a=1.:tuple(zip(c[0]+radius*a*np.cos(theta),c[1]+radius*np.sin(theta)))
    if family=='square':return polygon_shape(((0,0),(1,0),(1,1),(0,1)),tags=(1,2,3,4),family=family,seed=seed)
    if family=='rectangle':
        require(.2<=aspect<=5,'declared aspect range [.2,5]')
        return polygon_shape(((0,0),(aspect,0),(aspect,1),(0,1)),tags=(1,2,3,4),family=family,seed=seed)
    if family=='lshape':return polygon_shape(((0,0),(1,0),(1,.5),(.5,.5),(.5,1),(0,1)),tags=(1,2,3,4,5,6),family=family,seed=seed)
    if family in {'disk','ellipse'}:
        a=1. if family=='disk' else aspect
        return polygon_shape(circle(.5,a=a),family=family,seed=seed,intended=(family,.5,.5,.5*a,.5))
    if family=='hole':return polygon_shape(((0,0),(1,0),(1,1),(0,1)),(circle(.18),),family=family,seed=seed,intended=('square_minus_disk',.5,.5,.18))
    if family=='annulus':return polygon_shape(circle(.5),(circle(.2),),family=family,seed=seed,intended=('annulus',.5,.5,.5,.2))
    if family=='polygon':return polygon_shape(((0,0),(1.1,0),(.9,.7),(.5,1),(-.1,.6)),family=family,seed=seed)
    raise ValueError('unsupported geometry family')


@dataclass
class GeometryDomain:
    shape: Shape
    vertices: np.ndarray
    cells: np.ndarray
    boundary_edges: np.ndarray
    boundary_tags: np.ndarray
    region_tags: np.ndarray
    provenance: dict

    @cached_property
    def cell_areas(self):
        t=self.vertices[self.cells];a=t[:,1]-t[:,0];b=t[:,2]-t[:,0];return (a[:,0]*b[:,1]-a[:,1]*b[:,0])/2
    @cached_property
    def edge_index(self):
        edges=np.concatenate([self.cells[:,[0,1]],self.cells[:,[1,2]],self.cells[:,[2,0]]]);return np.unique(np.sort(edges,axis=1),axis=0).T
    @property
    def edge_lengths(self):return np.linalg.norm(self.vertices[self.edge_index[0]]-self.vertices[self.edge_index[1]],axis=1)
    @property
    def boundary_lengths(self):return np.linalg.norm(np.diff(self.vertices[self.boundary_edges],axis=1)[:,0],axis=1)
    @property
    def boundary_normals(self):
        d=np.diff(self.vertices[self.boundary_edges],axis=1)[:,0];return np.stack((d[:,1],-d[:,0]),axis=1)/self.boundary_lengths[:,None]
    @property
    def mesh_id(self):return digest({'schema':'GeometryDomain-v1','geometry':self.shape.geometry_id,'polygon':self.shape.polygon_id,'v':self.vertices.tolist(),'t':self.cells.tolist(),'boundary':self.boundary_edges.tolist(),'tags':self.boundary_tags.tolist(),'regions':self.region_tags.tolist()})
    @property
    def geometry_id(self):return self.shape.geometry_id
    @property
    def extent(self):
        a,b,c,d=self.shape.polygon.bounds;return ((a,c),(b,d))
    @cached_property
    def graph(self):
        a,b=self.edge_index;v=self.edge_lengths;return coo_matrix((np.r_[v,v],(np.r_[a,b],np.r_[b,a])),shape=(len(self.vertices),)*2).tocsr()
    @cached_property
    def _tree(self):
        shapely,_=dependencies();return shapely.STRtree(shapely.polygons(self.vertices[self.cells]))
    def locate(self,position,*,reject=True,coordinate_tolerance=None):
        shapely,_=dependencies();p=np.asarray(position,dtype=float).reshape(-1,2);pairs=self._tree.query(shapely.points(p),predicate='covered_by')
        ids=np.full(len(p),-1,dtype=int)
        # Min triangle ID is deterministic for shared-edge hits; P1 value agrees.
        np.minimum.at(ids:=np.full(len(p),len(self.cells),dtype=int),pairs[0],pairs[1]);ids[ids==len(self.cells)]=-1
        # Floating point evaluation of edge quadrature can fall a few ulps off
        # a boundary. Admit only roundoff-distance hits, never physical exterior.
        missing=np.flatnonzero(ids<0)
        if len(missing):
            nearest=self._tree.nearest(shapely.points(p[missing]));distance=shapely.distance(self._tree.geometries[nearest],shapely.points(p[missing]))
            tol=64*np.finfo(float).eps*max(1.,np.abs(self.vertices).max()) if coordinate_tolerance is None else coordinate_tolerance
            keep=distance<=tol;ids[missing[keep]]=nearest[keep]
        if reject:require(np.all(ids>=0),'query is outside represented domain or inside a hole')
        cells=self.cells[np.maximum(ids,0)];t=self.vertices[cells]
        A=np.stack((t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=-1);local=np.linalg.solve(A,(p-t[:,0])[...,None])[...,0]
        weights=np.column_stack((1-local.sum(1),local));weights[ids<0]=0
        return ids,weights
    def evaluate(self,nodal,position):
        ids,w=self.locate(position);value=np.asarray(nodal)
        return np.einsum('qi,...qi->...q',w,value[...,self.cells[ids]])
    def sample(self,count,rng,*,strategy='uniform'):
        require(count>=0 and strategy in {'uniform','coverage','clustered','boundary','partial'},'invalid sensor sampling strategy')
        if count==0:return np.zeros((0,2))
        if strategy=='coverage':
            candidates=self.sample(max(16*count,256),rng);chosen=[int(rng.integers(len(candidates)))];dist=np.full(len(candidates),np.inf)
            for _ in range(1,count):
                dist=np.minimum(dist,np.sum((candidates-candidates[chosen[-1]])**2,axis=1));chosen.append(int(dist.argmax()))
            return candidates[chosen]
        if strategy=='uniform':
            chosen=rng.choice(len(self.cells),size=count,p=self.cell_areas/self.cell_areas.sum());uv=rng.random((count,2));uv[:,0]=np.sqrt(uv[:,0])
            w=np.column_stack((1-uv[:,0],uv[:,0]*(1-uv[:,1]),uv[:,0]*uv[:,1]));return np.einsum('qi,qij->qj',w,self.vertices[self.cells[chosen]])
        center=self.sample(1,rng)[0];out=[];lo=np.array([b[0] for b in self.extent]);hi=np.array([b[1] for b in self.extent])
        while len(out)<count:
            trial=self.sample(max(32,count),rng);_,w=self.locate(trial)
            if strategy=='clustered':keep=np.linalg.norm(trial-center,axis=1)<.2*np.linalg.norm(hi-lo)
            elif strategy=='partial':keep=trial[:,0]<(lo[0]+hi[0])/2
            else:keep=self.features(trial)['distance']<.1*np.linalg.norm(hi-lo)
            out.extend(trial[keep].tolist())
        return np.array(out[:count])
    def features(self,position):
        shapely,_=dependencies();p=np.asarray(position).reshape(-1,2);a=self.vertices[self.boundary_edges[:,0]];d=self.vertices[self.boundary_edges[:,1]]-a
        t=np.clip(np.einsum('qej,ej->qe',p[:,None]-a,d)/np.sum(d*d,axis=1),0,1)
        nearest=a[None]+t[...,None]*d;dist=np.linalg.norm(p[:,None]-nearest,axis=-1);edge=dist.argmin(1);distance=dist[np.arange(len(p)),edge]
        inside=shapely.covers(self.shape.polygon,shapely.points(p));ids,_=self.locate(p,reject=False)
        ties=np.abs(dist-distance[:,None])<=64*np.finfo(float).eps*max(1.,np.abs(self.vertices).max())
        normal=ties.astype(float)@self.boundary_normals/np.maximum(ties.sum(1,keepdims=True),1)
        tag=np.where(ties,self.boundary_tags[None],np.iinfo(np.int64).max).min(1)
        return {'distance':distance,'signed_distance':np.where(inside,distance,-distance),'inside':inside,
                'boundary_tag':tag,'normal':normal,'normal_defined':ties.sum(1)==1,
                'local_h':np.sqrt(self.cell_areas[np.maximum(ids,0)]),'region':self.region_tags[np.maximum(ids,0)]}
    def quality(self):
        tri=self.vertices[self.cells];length=np.stack([np.linalg.norm(tri[:,(i+1)%3]-tri[:,i],axis=1) for i in range(3)],axis=1)
        angles=[]
        for i in range(3):
            a,b,c=length[:,i],length[:,(i+1)%3],length[:,(i+2)%3];angles.append(np.arccos(np.clip((b*b+c*c-a*a)/(2*b*c),-1,1)))
        return {'h_max':float(length.max()),'h_mean':float(self.edge_lengths.mean()),'min_angle_degrees':float(np.min(angles)*180/np.pi),
                'max_aspect_ratio':float((length.max(1)**2/(2*self.cell_areas)).max()),'vertices':len(self.vertices),'cells':len(self.cells),'boundary_edges':len(self.boundary_edges)}
    def validate(self):
        shapely,_=dependencies();require(self.vertices.ndim==2 and self.vertices.shape[1]==2 and np.isfinite(self.vertices).all(),'invalid mesh coordinates')
        require(self.cells.ndim==2 and self.cells.shape[1]==3 and self.cells.min()>=0 and self.cells.max()<len(self.vertices),'invalid triangle indices')
        require(np.issubdtype(self.cells.dtype,np.integer) and np.issubdtype(self.boundary_edges.dtype,np.integer),'indices must be integers')
        require(self.boundary_edges.ndim==2 and self.boundary_edges.shape[1]==2 and self.boundary_edges.min()>=0 and self.boundary_edges.max()<len(self.vertices),'invalid boundary indices')
        length=np.ptp(self.vertices,axis=0).max();tol=128*np.finfo(float).eps*length
        require(np.all(self.cell_areas>np.finfo(float).eps*length**2),'degenerate or incorrectly oriented triangles')
        require(len(np.unique(self.cells))==len(self.vertices),'isolated vertices')
        require(self.region_tags.shape==(len(self.cells),) and self.boundary_tags.shape==(len(self.boundary_edges),),'mesh tag dimensions mismatch')
        tri=shapely.polygons(self.vertices[self.cells]);area=float(self.cell_areas.sum())
        require(np.all(shapely.covered_by(tri,self.shape.polygon.buffer(tol))),'triangle crosses exterior wall or hole')
        union=shapely.union_all(tri)
        require(abs(area-self.shape.polygon.area)<1e-10*area and union.symmetric_difference(self.shape.polygon).area<1e-10*area,'mesh coverage/overlap mismatch')
        all_edges=np.concatenate([self.cells[:,[0,1]],self.cells[:,[1,2]],self.cells[:,[2,0]]]);keys,cnt=np.unique(np.sort(all_edges,axis=1),axis=0,return_counts=True)
        require(np.all(cnt<=2) and set(map(tuple,keys[cnt==1]))==set(map(tuple,np.sort(self.boundary_edges,axis=1))),'boundary topology mismatch')
        require(len(self.vertices)-len(keys)+len(self.cells)==1-len(self.shape.holes),'Euler/topology mismatch')
        require(connected_components(self.graph,directed=False)[0]==1,'disconnected mesh')
        mid=self.vertices[self.boundary_edges].mean(1);eps=1e-7*self.quality()['h_mean']
        require(np.all(shapely.covers(self.shape.polygon,shapely.points(mid-eps*self.boundary_normals))) and not np.any(shapely.covers(self.shape.polygon,shapely.points(mid+eps*self.boundary_normals))),'boundary normals do not point out of domain')
        q=self.quality();require(q['min_angle_degrees']>5 and q['max_aspect_ratio']<50,'unsupported mesh element quality')
        require(np.all(self.boundary_tags>0),'missing boundary marker')
        return self
    def save(self,path):
        from dataclasses import asdict
        np.savez_compressed(path,shape=json.dumps(asdict(self.shape)),vertices=self.vertices,cells=self.cells,boundary_edges=self.boundary_edges,boundary_tags=self.boundary_tags,region_tags=self.region_tags,provenance=json.dumps(self.provenance),mesh_id=self.mesh_id)
    @classmethod
    def load(cls,path):
        a=np.load(path,allow_pickle=False);s=json.loads(str(a['shape']));shape=Shape(s['family'],tuple(map(tuple,s['outer'])),tuple(tuple(map(tuple,h)) for h in s['holes']),tuple(s['tags']),s['seed'],tuple(s['intended']))
        mesh=cls(shape,*(a[k] for k in ('vertices','cells','boundary_edges','boundary_tags','region_tags')),json.loads(str(a['provenance']))).validate()
        require(mesh.mesh_id==str(a['mesh_id']),'mesh identity mismatch');return mesh


def mesh_shape(shape,h=.1,*,portable=True):
    if portable:
        from .portable import stored_mesh
        frozen=stored_mesh(shape,h)
        if frozen is not None:return frozen
    require(h>0,'positive target mesh length required');shapely,triangle=dependencies();poly=shape.polygon
    vertices=[];segments=[];holes=[]
    for ring in (shape.outer,*shape.holes):
        start=len(vertices);vertices.extend(ring);segments.extend((start+i,start+(i+1)%len(ring)) for i in range(len(ring)))
    for ring in shape.holes:holes.append(tuple(shapely.Polygon(ring).representative_point().coords)[0])
    data={'vertices':np.asarray(vertices),'segments':np.asarray(segments),'segment_markers':np.asarray(shape.tags)[:,None]}
    if holes:data['holes']=np.asarray(holes)
    # Triangle's command switch parser expects positional decimal numbers.
    # Scientific notation (e.g. a5e-05) is parsed as OTHER switches and can
    # silently produce a coarse mesh. Keep all digits without an exponent.
    target_area=h*h/2
    require(np.isfinite(target_area) and target_area>0,'unrepresentable target mesh area')
    area_text=np.format_float_positional(target_area,unique=True,trim='-')
    result=triangle.triangulate(data,f'pq28a{area_text}Q')
    v=result['vertices'].astype(np.float64);t=result['triangles'].astype(np.int64)
    oriented=np.concatenate((t[:,[0,1]],t[:,[1,2]],t[:,[2,0]]));keys,cnt=np.unique(np.sort(oriented,axis=1),axis=0,return_counts=True);boundary={tuple(k) for k in keys[cnt==1]}
    edges=np.array([e for e in oriented if tuple(sorted(e)) in boundary]);marker={tuple(sorted(e)):int(m) for e,m in zip(result['segments'],result['segment_markers'].ravel())}
    tags=np.array([marker[tuple(sorted(e))] for e in edges],dtype=np.int64)
    mesh=GeometryDomain(shape,v,t,edges,tags,np.zeros(len(t),dtype=np.int64),{'backend':'triangle','version':triangle.__version__,'target_h':h,'seed':shape.seed,'coordinate_system':'physical Cartesian','polygon_id':shape.polygon_id}).validate()
    require(mesh.cell_areas.max()<=target_area*(1+1e-10),'mesher failed requested maximum-area constraint; reference refinement invalid')
    return mesh
