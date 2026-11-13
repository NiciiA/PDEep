# READ ONLY: uses APIs already present in the running Phase G version.
# Paste this into a notebook cell (it queues if that kernel is busy), OR run:
# python scripts/inspect_phase_g_study.py /content/.../suites/<actual-suite-id>
# No training, model construction, reload, worker launch, or artifact writes.
from pathlib import Path
import json
from pdeep.learning_diagnostics.suite import _snapshot

try:
    p = C_WAVE_STUDY_PLAN
    directory = Path(p.root) / 'learning_diagnostics' / 'suites' / p.suite_id
except NameError:
    import sys
    if len(sys.argv) != 2:
        raise SystemExit('Provide the actual study directory, or paste this code in the notebook with C_WAVE_STUDY_PLAN defined.')
    directory = Path(sys.argv[1]).expanduser()

pointer = json.loads((directory / 'latest.json').read_text())
manifest = pointer['manifest']
if not Path(manifest).is_file():
    manifest = pointer.get('Drive_manifest')
s, _ = _snapshot(manifest)  # Verified, read-only saved suite state.
recipe = next(r for r in s['preview']['experiments'] if r['experiment'] == 'c_physics_ablation')
order = recipe['variants']  # Explicit list preserves order; JSON dict keys are sorted.
entry = s['entries']['c_physics_ablation']
budget = s['preview']['settings']['updates']
first_unfinished = next((v for v in order if entry['runs'].get(v, {}).get('status') != 'completed'), None)
print('Selected execution order:', ' -> '.join(order))
print('Snapshot:', manifest, '| Suite state:', s['status'])
print(f"{'Variant':16} {'State':27} {'Logged/planned':17} Checkpointed")
for variant in order:
    r = entry['runs'].get(variant, {})
    saved = r.get('last_successful_update', 0)
    successful = max(saved, r.get('observed_update', 0))
    checkpoint = r.get('checkpoint')
    if checkpoint:
        log = Path(checkpoint).parent.parent / 'steps.jsonl'
        if log.is_file():
            with log.open('rb') as stream:
                stream.seek(0, 2)
                stream.seek(max(0, stream.tell() - 262144))
                lines = stream.read().splitlines()
            for line in reversed(lines):
                try:
                    row = json.loads(line)
                except (ValueError, UnicodeError):
                    continue  # Ignore an incomplete in-progress last write.
                if row.get('run_id') == r.get('run_id') and row.get('status') == 'successful':
                    successful = max(successful, row['successful'])
                    break
    status = r.get('status', 'pending')
    if variant == first_unfinished and s.get('active_experiment') == 'c_physics_ablation':
        if s['status'] == 'running':
            status = 'finalizing (inferred)' if successful >= budget else 'running (inferred)'
        elif s['status'] in {'failed', 'interrupted'}:
            status = s['status'] + '; resume required'
    print(f"{variant:16} {status:27} {str(successful)+'/'+str(budget):17} {saved}")
print('Automatic next variant: YES after successful completion, unless failure/interruption stops the suite.')
print('Counts are saved/logged observations, not live process liveness. Checkpointed counts alone are resumable.')
