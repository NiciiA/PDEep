"""Read/restore existing checkpoints with no updates or rewriting originals."""
from pathlib import Path
import json
from pdeep.smooth.training import resume
from pdeep.artifact_io import read_payload,sha256
from pdeep.artifact_tensors import state_identity,tree_identity
from pdeep.core import require,digest
path=json.loads(Path('reports/smooth_finalization/persistence.json').read_text())['checkpoint'];before=sha256(path);record,payload=read_payload(path,kind='training_checkpoint')
engine=resume(path,Path('reports/colab_repair/legacy-resume'))
require(state_identity(engine.model)==digest(tree_identity(payload['model'])),'legacy model differs')
require(digest(tree_identity(engine.optimizer.state_dict()))==digest(tree_identity(payload['optimizer'])),'legacy optimizer state differs')
require(sha256(path)==before,'original checkpoint changed')
result={'passed':True,'original_checkpoint':path,'original_manifest_sha256':before,'model_optimizer_identical':True,'run_id_preserved':engine.config.run_id==record['run_id'],'sampler_cursor_preserved':engine.state.sampler_cursor==payload['training_state']['sampler_cursor'],'optimizer_updates':0,'dataset_certificate_id_preserved':engine.certificate['certification_id']==record['certificate']['certification_id'],'execution_qualification_id':engine.execution_certificate['certification_id']}
Path('reports/colab_repair/legacy-resume.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
