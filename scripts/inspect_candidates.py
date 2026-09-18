import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
records=[]
def fetch(url,path):
    with urllib.request.urlopen(url,timeout=60) as response:
        content=response.read()
    dest=ROOT/"records"/path
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_bytes(content)
    records.append({"url":url,"path":str(dest.relative_to(ROOT)),"sha256":hashlib.sha256(content).hexdigest()})

for name in ["td3-Pendulum-v1","sac-Walker2d-v3","sac-Humanoid-v3"]:
    metadata=json.loads((ROOT/"records"/f"{name}_api.json").read_text())
    rev=metadata["sha"]
    for file in ["data","system_info.txt","_stable_baselines3_version"]:
        fetch(f"https://huggingface.co/sb3/{name}/resolve/{rev}/{name}/{file}",f"{name}/{file}")

for filename in ["humanoid_v3.py","walker2d_v3.py","assets/humanoid.xml","assets/walker2d.xml"]:
    fetch(f"https://raw.githubusercontent.com/openai/gym/v0.21.0/gym/envs/mujoco/{filename}",f"gym_v0.21.0/{filename}")
fetch("https://raw.githubusercontent.com/openai/gym/v0.21.0/gym/envs/classic_control/pendulum.py","gym_v0.21.0/pendulum.py")
fetch("https://minari.farama.org/main/datasets/mujoco/walker2d/expert-v0/","minari_walker_expert.html")
fetch("https://minari.farama.org/main/datasets/mujoco/humanoid/expert-v0/","minari_humanoid_expert.html")
url="https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"
with urllib.request.urlopen(urllib.request.Request(url,method="HEAD"),timeout=60) as response:
    records.append({"url":url,"status":response.status,"bytes":response.headers.get("Content-Length"),
                    "downloaded":False,"license":"No separate dataset license located in the inspected release listing."})
(ROOT/"records/extra_sources.json").write_text(json.dumps(records,indent=2))
print(json.dumps(records,indent=2))
