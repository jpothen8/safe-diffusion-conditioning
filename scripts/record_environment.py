"""Record host, dependency licenses, source revision and asset hashes."""
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def command(args):
    result=subprocess.run(["rtk","proxy",*args],capture_output=True,text=True)
    return {"command":args,"returncode":result.returncode,"stdout":result.stdout,"stderr":result.stderr}

host={"utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"platform":platform.platform(),
      "python":platform.python_version(),"commands":[command(args) for args in
        [["uname","-a"],["lscpu"],["free","-b"],["df","-h",str(ROOT)],
         ["nvidia-smi"],["nvidia-smi","--query-gpu=name,driver_version,memory.total,compute_cap","--format=csv"]]]}
(ROOT/"records/host.json").write_text(json.dumps(host,indent=2))
packages=[]
for dist in sorted(importlib.metadata.distributions(),key=lambda d:d.metadata.get("Name","").lower()):
    licenses=[]
    for file in dist.files or []:
        if any(token in str(file).lower() for token in ["license","copying"]):
            path=Path(dist.locate_file(file))
            if path.is_file() and path.stat().st_size<200000:
                licenses.append({"path":str(file),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    packages.append({"name":dist.metadata.get("Name"),"version":dist.version,
                     "license":dist.metadata.get("License"),"license_expression":dist.metadata.get("License-Expression"),
                     "classifiers":dist.metadata.get_all("Classifier",[]),
                     "project_urls":dist.metadata.get_all("Project-URL",[]),"license_files":licenses})
(ROOT/"records/dependency_licenses.json").write_text(json.dumps(packages,indent=2))
repo=ROOT/"vendor/diffusion_policy"
revision=subprocess.check_output(["rtk","proxy","git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
source_files={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(repo.rglob("*")) if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts}
(ROOT/"records/vendor_hashes.json").write_text(json.dumps(source_files,indent=2))
checkpoint=ROOT/"assets/pusht_transformer.ckpt"
manifest={"retrieved_utc":host["utc"],"source":{"url":"https://github.com/real-stanford/diffusion_policy",
           "revision":revision,"license":"MIT","license_file":"vendor/diffusion_policy/LICENSE",
           "file_hashes":"records/vendor_hashes.json"},
          "checkpoint":{"url":"https://diffusion-policy.cs.columbia.edu/data/experiments/low_dim/pusht/diffusion_policy_transformer/train_0/checkpoints/epoch%3D0850-test_mean_score%3D0.967.ckpt",
          "path":str(checkpoint.relative_to(ROOT)),"bytes":checkpoint.stat().st_size,
          "sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
          "revision":"Author release: train_0, epoch 0850; no server version identifier; pinned by SHA256",
          "license":"No checkpoint-specific license found in the release listing. Source repository is MIT; no separate weights grant is assumed."},
          "candidate_resources":"records/candidate_sources.json",
          "extra_sources":"records/extra_sources.json",
          "dependency_versions":"requirements.lock.txt","dependency_licenses":"records/dependency_licenses.json"}
(ROOT/"records/asset_manifest.json").write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
