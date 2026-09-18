"""Record final provenance and verify preservation of the earlier experiments."""
import ast,hashlib,importlib.metadata as metadata,json,re
from datetime import datetime,timezone
from pathlib import Path
import gymnasium
ROOT=Path(__file__).resolve().parents[1];PARENT=ROOT.parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
report={'utc':datetime.now(timezone.utc).isoformat()}
original=json.loads((PARENT/'records/result_hashes.json').read_text())
for path,digest in original.items():assert sha(PARENT/path)==digest,path
report['original_push_t_artifact_hashes_verified']=len(original)
previous=json.loads((PARENT/'records/followup_result_hashes.json').read_text())
changed=[path for path,digest in previous.items() if sha(PARENT/path)!=digest]
assert set(changed)<=set(['README.md','RESULTS.md','TASK_SELECTION.md']),changed
report['prior_followup_hashes_unchanged']=len(previous)-len(changed)
report['prior_documentation_updated_to_link_round3']=changed
for p in [*ROOT.glob('src/*.py'),*ROOT.glob('scripts/*.py')]:ast.parse(p.read_text(),filename=str(p))
report['python_sources_parse']=True
missing=[]
for doc in [*ROOT.glob('*.md'),PARENT/'README.md',PARENT/'TASK_SELECTION.md']:
    for target in re.findall(r'\]\(([^)]+)\)',doc.read_text()):
        if target.startswith(('http:','https:','#')):continue
        if not (doc.parent/target.split('#')[0]).exists():missing.append((str(doc),target))
assert not missing,missing
report['local_document_links_valid']=True
gymroot=Path(gymnasium.__file__).parent
environment={'distributions':{},'environment_files':{}}
for package in ['gymnasium','mujoco','stable-baselines3','torch','diffusers']:
    dist=metadata.metadata(package)
    environment['distributions'][package]={'version':metadata.version(package),'license':dist.get('License-Expression',dist.get('License','Not specified'))}
for name in ['walker2d_v5.py','half_cheetah_v5.py','assets/walker2d.xml','assets/half_cheetah.xml']:
    path=gymroot/'envs/mujoco'/name
    environment['environment_files']['gymnasium/envs/mujoco/'+name]=sha(path)
(ROOT/'records/environment.json').write_text(json.dumps(environment,indent=2))
for name in ['audit','adapter_checks','reproduction_audit']:
    assert (ROOT/f'results/{name}.json').exists()
assert json.loads((ROOT/'results/audit.json').read_text())['passed']
assert json.loads((ROOT/'results/adapter_checks.json').read_text())['passed']
assert json.loads((ROOT/'results/reproduction_audit.json').read_text())['all_arrays_exact']
report['completed_audits_passed']=True
(ROOT/'records/delivery_checks.json').write_text(json.dumps(report,indent=2))
excluded={'.venv','__pycache__'}
manifest={str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*'))
          if p.is_file() and not(set(p.relative_to(ROOT).parts)&excluded) and p!=ROOT/'records/delivery_hashes.json'}
(ROOT/'records/delivery_hashes.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({**report,'recorded_files':len(manifest)},indent=2))
