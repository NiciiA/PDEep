"""CPU64 numerical certification ONLY. No model/optimizer imports or updates.

Four fixed Gaussian physical cases/PDE/geometry; all cases/resolutions share
continuous fields, coordinates and physical times. This is sampled numerical
certification, not a uniform bound over the unbounded Gaussian prior.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
from pathlib import Path
import argparse,json,time,datetime,gc
import numpy as np
from pdeep.core import digest
from pdeep.smooth.definition import *
from pdeep.smooth.data import calibrate,Normalizers
from pdeep.smooth.reference import Reference
from pdeep.smooth.certification import save_record,THRESHOLD,additional_sources
from pdeep.geometric.quadrature import boundary

OUT=Path('reports/smooth_finalization');OUT.mkdir(parents=True,exist_ok=True)

def write(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--families',nargs='*',default=list(GEOMETRIES));args=parser.parse_args()
    p=OUT/'normalizers.json'
    if p.exists():norms=Normalizers.fromdict(json.loads(p.read_text()))
    else:norms=calibrate(lambda s:print(s,flush=True));write(p,norms.asdict())
    for family in args.families:
        result=OUT/(family+'.json')
        if result.exists():print('existing '+family,flush=True);continue
        start=time.perf_counter();print('START '+family,flush=True)
        split='validation' if family in SEEN else 'geometry_held_out'
        fields={p:[physical_fields(family,p,split,i) for i in range(4)] for p in PDES}
        mesh=geometry(family,.08);x=mesh.sample(128,random_stream('certification',family,'field-coordinates'))
        sensors=mesh.sample(128,random_stream('certification',family,'sensor-coordinates'))
        sensor_t=random_stream('certification',family,'continuous-sensor-times').random(128);sensor_t[:8]=0.;sensor_t[8:24]*=.1
        xx=np.concatenate([np.tile(x,(len(TIMES),1)),sensors]);tt=np.concatenate([np.repeat(TIMES,len(x)),sensor_t]);cut=len(x)*len(TIMES)
        runs={};meshes={};boundaries=[]
        # Different degree/mesh comparator independently limits spatial error.
        for tag,h,degree,dt in [('h04',.04,4,.002),('production',.02,4,.002),('temporal_coarse',.02,4,.004),('temporal_fine',.02,4,.001),('independent',.01,3,.001)]:
            print(f'{family} {tag}',flush=True)
            r=Reference(family,h,degree);meshes[tag]=r.mesh_report();runs[tag]={}
            b=boundary(r.space.mesh,4);B=r.space.sample_operator(b.position)
            # Every constrained nodal DOF is absent from interpolation operator.
            trace=float(abs(B).max()) if B.nnz else 0.
            boundaries.append({'tag':tag,'max_Dirichlet_trace_operator':trace,'normal_unit_error':float(np.max(abs(np.linalg.norm(b.normal,axis=1)-1))),
                               'boundary_tags':sorted(map(int,np.unique(b.boundary_tag))),'outside_points':int(np.sum(~r.space.mesh.features(xx)['inside']))})
            for pde in PDES:
                fs=fields[pde]
                if pde=='poisson':
                    if tag.startswith('temporal'):continue
                    u,diag=r.poisson([f['f'] for f in fs],xx);runs[tag][pde]={'u':u,'diagnostic':diag}
                else:
                    u,v,diag=r.dynamic(pde,[f['u0'] for f in fs],[f['v0'] for f in fs] if pde=='wave' else None,xx,tt,dt)
                    runs[tag][pde]={'u':u,'diagnostic':diag}
                    if v is not None:runs[tag][pde]['v']=v
                print(pde+' '+str(runs[tag][pde]['diagnostic']),flush=True)
            del r;gc.collect()
        rows=[];diagnostics={tag:{p:run['diagnostic'] for p,run in ps.items()} for tag,ps in runs.items()}
        for pde in PDES:
            for q in ('u','v') if pde=='wave' else ('u',):
                for left,right,label in [('h04','production','mesh04_to02'),('production','independent','production_vs_independent'),('production','temporal_fine','dt002_to001'),('temporal_coarse','production','dt004_to002')]:
                    if pde=='poisson' and label.startswith('dt'):continue
                    a=runs[left][pde][q];b=runs[right][pde][q];d=(a-b)/norms.scale(pde,q)
                    for i in range(4):
                        row={'pde':pde,'quantity':q,'family':family,'index':i,'case_id':physical_id(family,pde,split,i),'comparison':label,
                             'field_normalized_RMSE':float(np.sqrt(np.mean(d[i,:cut]**2))),'sensor_normalized_RMSE':float(np.sqrt(np.mean(d[i,cut:]**2))),
                             'maximum_normalized_absolute_error':float(abs(d[i]).max()),'time_normalized_RMSE':{str(t):float(np.sqrt(np.mean(d[i,j*len(x):(j+1)*len(x)]**2))) for j,t in enumerate(TIMES)}}
                        rows.append(row)
        # t=0 compatibility checked against the SAME continuous field.
        initial=[]
        for pde in ('heat','wave'):
            for q,role in [('u','u0')]+([('v','v0')] if pde=='wave' else []):
                for i in range(4):
                    d=(runs['production'][pde][q][i,:len(x)]-fields[pde][i][role](x))/norms.scale(pde,q)
                    initial.append({'pde':pde,'quantity':q,'index':i,'normalized_RMSE':float(np.sqrt(np.mean(d*d)))})
        payload={'family':family,'cases_per_PDE':4,'source_revision':physics_identity(),'reference_policy':POLICY,'normalizer_identity':norms.identity,'meshes':meshes,'boundaries':boundaries,'initial':initial,'rows':rows,'diagnostics':diagnostics,'seconds':time.perf_counter()-start,
                 'physical_fields':{p:[{k:v.identity for k,v in f.items()} for f in fs] for p,fs in fields.items()},'coordinate_identity':digest((xx.tolist(),tt.tolist()))}
        # Small representative solution arrays for scientific plots, not training.
        np.savez_compressed(OUT/(family+'-values.npz'),x=x,times=TIMES,**{f'{tag}_{p}_{q}':run[q][0,:cut].reshape(len(TIMES),len(x)) for tag,ps in runs.items() for p,run in ps.items() for q in ('u','v') if q in run})
        write(result,payload);print('DONE '+family+' '+str(payload['seconds']),flush=True)
    families=[json.loads((OUT/(g+'.json')).read_text()) for g in GEOMETRIES if (OUT/(g+'.json')).exists()]
    if len(families)!=8:print('partial evidence; no certificate published',flush=True);return
    require(all(f['source_revision']==physics_identity() and f['normalizer_identity']==norms.identity for f in families),'stale evidence; rerun changed numerical suite')
    analytical=json.loads((OUT/'analytical-controls.json').read_text());require(analytical['passed'] and analytical['source_revision']==physics_identity(),'analytical square control missing/stale')
    controls=json.loads((OUT/'independent-controls.json').read_text());require(controls['passed'] and controls['source_revision']==physics_identity(),'independent controls missing/stale')
    capabilities=[]
    for f in families:
        for pde in PDES:
            selected=[r for r in f['rows'] if r['pde']==pde and r['comparison'] in {'production_vs_independent','dt002_to001'}]
            expected_quantities={'u','v'} if pde=='wave' else {'u'}
            require(all({r['index'] for r in selected if r['quantity']==q and r['comparison']=='production_vs_independent'}==set(range(4)) for q in expected_quantities),'incomplete representative case evidence')
            errors=[r[k] for r in selected for k in ('field_normalized_RMSE','sensor_normalized_RMSE')]
            errors += [v for r in selected for v in r['time_normalized_RMSE'].values()]
            errors += [r['normalized_RMSE'] for r in f['initial'] if r['pde']==pde]
            bc=max(b['max_Dirichlet_trace_operator'] for b in f['boundaries']);outside=max(b['outside_points'] for b in f['boundaries'])
            capabilities.append({'pde':pde,'family':f['family'],'physical_cases':4,'maximum_sampled_normalized_RMSE':max(errors),'median_sampled_normalized_RMSE':float(np.median(errors)),'q95_sampled_normalized_RMSE':float(np.quantile(errors,.95)),
                                 'passed':max(errors)<=THRESHOLD and bc<1e-8 and outside==0,'boundary_trace_operator':bc,'outside_points':outside})
    body={'certification_schema':'G1_smooth-numerical-certification-v1','benchmark_id':BENCHMARK_ID,'certified_PDE_families':sorted({c['pde'] for c in capabilities if c['passed']}),'certified_geometry_families':sorted({c['family'] for c in capabilities if c['passed']}),'certified_mesh_resolutions':[.02],
          'certified_field_prior':{p:prior_version(p) for p in PDES},'certified_boundary_policies':[BOUNDARY_VERSION],'certified_observation_policy':OBSERVATION_VERSION,
          'mesh_records':{f['family']:f['meshes'] for f in families},'independent_controls':json.loads((OUT/'independent-controls.json').read_text()),
          'analytical_square_controls':analytical,'boundary_controls':json.loads((OUT/'boundary-controls.json').read_text()),'reference_policy':POLICY,'normalizers':norms.asdict(),'capabilities':capabilities,'numerical_error_summary':{'threshold':THRESHOLD,'maximum':max(c['maximum_sampled_normalized_RMSE'] for c in capabilities)},
          'validation_case_ids':[physical_id(f['family'],p,'validation' if f['family'] in SEEN else 'geometry_held_out',i) for f in families for p in PDES for i in range(4)],
          'sampled_error_tables':{f['family']:f['rows'] for f in families},'initial_compatibility':{f['family']:f['initial'] for f in families},'reference_diagnostics':{f['family']:f['diagnostics'] for f in families},'boundary_evidence':{f['family']:f['boundaries'] for f in families},'physical_field_identities':{f['family']:f['physical_fields'] for f in families},'known_coefficients':dict(COEFFICIENTS),'reference_solver_identity':SOLVER_VERSION,'source_revision':physics_identity(),'additional_scientific_sources':additional_sources(),'verification_timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'pass_fail_status':'passed' if all(c['passed'] for c in capabilities) else 'partial/failed','splits':split_manifest(),
          'limits':'96 fixed physical cases;128 area points x9 times plus128 continuous-time sensors/case. Sampled convergence evidence, not a uniform bound over Gaussian tails; D0 fixed polygon instances only. Curved-domain polygon approximation is separate.'}
    path=save_record(body);write(path.parent/'default.json',{'record':path.name});write(OUT/'certification-summary.json',body)
    print('CERTIFICATE '+str(path)+' '+body['pass_fail_status'],flush=True)

if __name__=='__main__':main()
