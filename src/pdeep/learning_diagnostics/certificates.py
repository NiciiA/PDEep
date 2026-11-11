"""Read-only numerical capability resolution. Never launches qualification.

Integrity (body identity AND original-byte checksum) is distinct from numerical
compatibility. Copying a Linux certificate to a Mac does not qualify the Mac.
"""
from dataclasses import dataclass, field
from pathlib import Path
import hashlib, json, os, tempfile
from ..core import require, digest
from ..smooth.definition import BENCHMARK_ID, PDES, SEEN, POLICY, BOUNDARY_VERSION, prior_version, SOLVER_VERSION, split_manifest
from ..smooth.certification import ROOT, load_record, authorize, normalizers

SUPPLIED_ID = '524b9aa89bb0baa3606cc47657d6ee31c95ef2d8a212e9c96b6a4ba43d33c711'
ORIGINAL_SHA = {SUPPLIED_ID: 'f1a245a503f366191b9fb74815963d2ba093126ecbcbfccf2996237b0bb5e2da'}
LOCAL_CACHE = Path(os.environ.get('PDEEP_OUTPUT_DIR', 'artifacts')) / 'certifications' / BENCHMARK_ID


def differences(expected, actual, prefix=''):
    if isinstance(expected, dict) and isinstance(actual, dict):
        return [r for k in sorted(set(expected) | set(actual)) for r in differences(expected.get(k), actual.get(k), f'{prefix}.{k}'.strip('.'))]
    return [] if expected == actual else [dict(field=prefix, certified=expected, current=actual)]


def certificate_file(path):
    """Accept the existing one-hop registry pointer, preserving full JSON bytes."""
    path=Path(path); header=json.loads(path.read_bytes())
    if 'record' in header:
        require(set(header)=={'record'} and Path(header['record']).name==header['record'],'unsafe certification pointer')
        target=path.parent/header['record'];require(target!=path,'cyclic certification pointer')
        require('record' not in json.loads(target.read_bytes()),'nested certification pointer is unsupported')
        return target
    return path


def inspect_file(path):
    path = certificate_file(path); raw = path.read_bytes(); record = json.loads(raw)
    require(record.get('certification_id') == digest({k:v for k,v in record.items() if k != 'certification_id'}), 'certification body hash mismatch')
    checksum = hashlib.sha256(raw).hexdigest()
    expected = ORIGINAL_SHA.get(record['certification_id'])
    require(expected is None or checksum == expected, 'original certificate file SHA256 mismatch (distinct from certification ID)')
    # Existing schema/independent-control checks, without environment acceptance.
    load_record(path, current=False)
    require(record.get('pass_fail_status') == 'passed', 'certificate did not pass')
    return record, checksum


def copy_certificate(source, destination):
    """One-time import of FULL original bytes, atomically verified; no deletion."""
    source, destination = certificate_file(source), Path(destination)
    record, checksum = inspect_file(source)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / (record['certification_id'] + '.json')
    if target.exists():
        require(hashlib.sha256(target.read_bytes()).hexdigest() == checksum, 'immutable certificate destination collision')
    else:
        fd, tmp = tempfile.mkstemp(prefix='.certificate-', dir=destination)
        try:
            with os.fdopen(fd, 'wb') as stream: stream.write(source.read_bytes())
            require(inspect_file(tmp)[1] == checksum, 'certificate copy verification failed')
            os.replace(tmp, target)
        finally:
            if Path(tmp).exists(): Path(tmp).unlink()
    return target


@dataclass
class Resolution:
    status: str
    location: str | None = None
    record: dict | None = field(default=None, repr=False)
    mismatches: list = field(default_factory=list)
    checksum: str | None = None
    @property
    def verified(self): return self.status == 'VERIFIED'
    def require(self):
        require(self.verified, f'Certification {self.status}: {self.mismatches}. Place the complete issued JSON in local/Drive certifications/{BENCHMARK_ID}; no qualification is launched automatically.')
        return self.record
    def summary(self):
        r = self.record or {}
        return {'Benchmark': r.get('benchmark_id', BENCHMARK_ID), 'Certification': self.status,
                'Certificate ID': r.get('certification_id'), 'Certified PDEs': r.get('certified_PDE_families'),
                'Certified geometries': r.get('certified_geometry_families'), 'Certified mesh resolution': r.get('certified_mesh_resolutions'),
                'Source compatibility': 'matching' if self.verified else self.mismatches, 'Location': self.location,
                'Raw file SHA256': self.checksum, 'Qualification rerun required': 'No' if self.verified else 'Undetermined: locate the existing issued artifact first' if self.status=='MISSING' else 'Yes, or restore the certified runtime/settings',
                'Qualification executed': False}


def resolve_certification(benchmark_id=BENCHMARK_ID, requested_capabilities=None, configured_path=None, drive_root=None, *, cache_dir=None):
    caps = {} if requested_capabilities is None else requested_capabilities
    require(isinstance(caps,dict),'requested_capabilities must be a dictionary')
    require(set(caps)<={'certification_id','pdes','families','mesh_resolution','boundary','observation_policy','prior','normalization_id','solver'},'unknown requested certification capability')
    require(benchmark_id == BENCHMARK_ID, 'Original rough G1 remains blocked')
    cache = LOCAL_CACHE if cache_dir is None else Path(cache_dir)
    identifier = caps.get('certification_id', SUPPLIED_ID)
    paths = [Path(configured_path)] if configured_path is not None else []
    if configured_path is None and identifier is not None:
        paths += [cache / (identifier+'.json')]
        if drive_root: paths += [Path(drive_root)/'certifications'/benchmark_id/(identifier+'.json')]
        paths += [ROOT/(identifier+'.json')]
        # Explicit identity=None permits existing runtime-qualified registry records.
    if configured_path is None and identifier is None: paths = list(cache.glob('*.json')) + (list((Path(drive_root)/'certifications'/benchmark_id).glob('*.json')) if drive_root else []) + list((ROOT/'qualified').glob('*.json'))
    found = []
    for path in paths:
        if not path.is_file(): continue
        record = None; checksum = None
        try:
            record, checksum = inspect_file(path)
            copy_certificate(path, cache)
            require(identifier is None or record['certification_id'] == identifier, 'selected certification ID mismatch')
            require(record['benchmark_id'] == benchmark_id, 'certificate benchmark mismatch')
            require(record['certified_field_prior']=={p:prior_version(p) for p in PDES},'certified field-prior identity mismatch')
            require(record['reference_solver_identity']==SOLVER_VERSION,'numerical solver identity mismatch')
            require(record['splits']==split_manifest(),'data split identity mismatch')
            normalizers(record)  # independently recompute the normalization identity
            from ..smooth.qualification import runtime_identity
            mismatches = differences(record.get('runtime_identity'), runtime_identity(), 'runtime_identity') if 'qualification' in record else []
            for key, actual in (('normalization_id', record['normalizers']['identity']), ('prior', record['certified_field_prior']), ('solver', record['reference_solver_identity'])):
                if key in caps: mismatches += differences(caps[key], actual, key)
            require(caps.get('mesh_resolution', .02) in record['certified_mesh_resolutions'], 'uncertified mesh resolution')
            require(caps.get('observation_policy', POLICY['observation_policy']) == record['certified_observation_policy'], 'uncertified observation policy')
            if mismatches:
                found.append(Resolution('INCOMPATIBLE',str(path),record,mismatches,checksum))
                if configured_path is not None or identifier is not None:return found[-1]
                continue
            authorize(record, benchmark=benchmark_id, pdes=tuple(caps.get('pdes', PDES)), families=tuple(caps.get('families', SEEN)), normalizer_identity=caps.get('normalization_id'), boundary=caps.get('boundary', BOUNDARY_VERSION))
            cached = copy_certificate(path, cache)
            return Resolution('VERIFIED', str(cached), record, [], checksum)
        except (ValueError, KeyError, OSError, TypeError) as error:
            found.append(Resolution('INCOMPATIBLE', str(path), record, [{'error':str(error)}], checksum))
            # An explicitly selected artifact must never silently fall through.
            if configured_path is not None or identifier is not None: return found[-1]
    return found[0] if found else Resolution('MISSING', mismatches=[{'required_file':str(cache/(str(identifier)+'.json')), 'instruction':'Copy the complete supplied JSON from Drive or project registry. No automatic recertification.'}])
