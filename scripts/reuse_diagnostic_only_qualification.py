"""Reuse completed numerical checks after a diagnostic-only policy-report fix.

Proves that every AST node outside report_selection is identical. Never rewrites
old certificates, mesh IDs, numerical measurements or checkpoint artifacts.
"""
from pathlib import Path
import ast,copy,hashlib,json
from pdeep.core import digest,require
from pdeep.smooth.certification import save_record
from pdeep.smooth.qualification import CACHE
p=Path('src/pdeep/smooth/resolution.py');current=p.read_text()
new="old=record['mesh_records'][family]['production'];_,baseline,_=resolve_reference(family,policy=record['reference_policy']);expected={**baseline.asdict(),**{k:old[k] for k in ('h','degree','geometry_id','polygon_id','mesh_id')}}"
old="old=record['mesh_records'][family]['production'];expected={**selected.asdict(),**{k:old[k] for k in ('h','degree','geometry_id','polygon_id','mesh_id')}}"
require(new in current,'unexpected diagnostic source; no automatic reuse')
previous=current.replace(new,old)
def numerical_ast(source):
 tree=ast.parse(source);tree.body=[n for n in tree.body if not(isinstance(n,ast.FunctionDef) and n.name=='report_selection')];return ast.dump(tree,include_attributes=False)
require(numerical_ast(current)==numerical_ast(previous),'numerical resolution code changed; rerun qualification')
oldsha=hashlib.sha256(previous.encode()).hexdigest();newsha=hashlib.sha256(current.encode()).hexdigest()
for identifier in ('59dcbd688ad875de7bcdcab268e2725dac08fec4551b30d09d50bcc3509ee4e9','1de974851000036bc940eecd15b74877c17013281b15a0778c39cf0d69d404ed'):
 r=json.loads((CACHE/(identifier+'.json')).read_text());require(r['runtime_identity']['resolution_sources'][str(p)]==oldsha,'unexpected predecessor source')
 body={k:v for k,v in r.items() if k!='certification_id'};body['runtime_identity']['resolution_sources'][str(p)]=newsha
 body['non_numerical_revision']={'parent_qualification_id':identifier,'old_diagnostic_file_sha256':oldsha,'new_diagnostic_file_sha256':newsha,'unchanged_numerical_AST_sha256':hashlib.sha256(numerical_ast(current).encode()).hexdigest(),'reason':'Only reporting of old-certificate dt differences changed. Numerical resolution, actual mesh arrays, physical fields, PDE kernels and measured comparisons are unchanged.'}
 path=save_record(body,CACHE);alias=CACHE/(digest((r['parent_certification_id'],body['runtime_identity']))+'.json');alias.write_text(json.dumps({'record':path.name})+'\n')
 shipped=Path('certifications/G1_smooth_v1/qualified');shipped.mkdir(exist_ok=True);target=shipped/path.name
 if target.exists():require(target.read_bytes()==path.read_bytes(),'immutable shipped qualification collision')
 else:target.write_bytes(path.read_bytes())
 print(path)
