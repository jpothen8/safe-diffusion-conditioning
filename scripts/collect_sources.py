"""Acquire candidate metadata and record immutable upstream revisions."""
import hashlib
import json
import pathlib
import subprocess
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]

def fetch(url, path):
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"url": url, "path": str(path.relative_to(ROOT)),
            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}

records = []
for repo in ["sb3/td3-Pendulum-v1", "sb3/sac-Walker2d-v3", "sb3/sac-Humanoid-v3"]:
    name = repo.split("/")[1]
    entry = fetch(f"https://huggingface.co/api/models/{repo}?blobs=true", ROOT / "records" / f"{name}_api.json")
    metadata = json.loads((ROOT / entry["path"]).read_text())
    revision = metadata["sha"]
    entry["revision"] = revision
    records.append(entry)
    for filename in ["README.md", "config.yml", "results.json", "args.yml"]:
        records.append(fetch(f"https://huggingface.co/{repo}/resolve/{revision}/{filename}", ROOT / "records" / name / filename))
    for sibling in metadata["siblings"]:
        if sibling["rfilename"] == f"{name}.zip":
            url = f"https://huggingface.co/{repo}/resolve/{revision}/{name}.zip"
            request = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(request, timeout=60) as response:
                records.append({"url": url, "http_status": response.status,
                                "revision": revision, "file_metadata": sibling,
                                "license": metadata.get("cardData", {}).get("license", "not specified")})

for name, url in {
    "humanoid_docs.html": "https://gymnasium.farama.org/environments/mujoco/humanoid/",
    "walker_docs.html": "https://gymnasium.farama.org/environments/mujoco/walker2d/",
    "pendulum_docs.html": "https://gymnasium.farama.org/environments/classic_control/pendulum/",
    "pytorch_blackwell.html": "https://pytorch.org/blog/pytorch-2-7/",
    "nvidia_compute.html": "https://developer.nvidia.com/cuda/gpus",
}.items():
    records.append(fetch(url, ROOT / "records" / name))
(ROOT / "records" / "candidate_sources.json").write_text(json.dumps(records, indent=2))
print(json.dumps(records, indent=2))
