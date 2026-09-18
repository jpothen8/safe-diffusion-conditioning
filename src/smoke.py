import argparse
import json
import time
import numpy as np
import torch
import pymunk
import core

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cuda")
parser.add_argument("--episodes", type=int, default=8)
args = parser.parse_args()
start = time.perf_counter()
policy = core.load_policy(args.device)
envs, histories = zip(*(core.make_env(100000 + i) for i in range(args.episodes)))
histories = list(histories)
returns = [[] for _ in envs]
done = [False for _ in envs]
all_actions = [[] for _ in envs]
for chunk in range(38):
    active = [i for i in range(len(envs)) if not done[i]]
    if not active:
        break
    actions = core.sample(policy, [histories[i] for i in active], 11000 + chunk)
    for i, aa in zip(active, actions):
        for action in aa:
            if done[i] or len(returns[i]) >= 300:
                break
            reward, done[i], _ = core.step(envs[i], histories[i], action)
            returns[i].append(reward)
            all_actions[i].append(action.tolist())
    print(json.dumps({"chunk": chunk, "successes": sum(done), "active": len(active)}), flush=True)
records = [{"seed": 100000 + i, "max_reward": max(rr), "sum_reward": sum(rr),
            "steps": len(rr), "success": done[i]} for i, rr in enumerate(returns)]
result = {"episodes": records, "mean_max_reward": float(np.mean([r["max_reward"] for r in records])),
          "success_rate": float(np.mean(done)), "seconds": time.perf_counter() - start,
          "torch": torch.__version__, "cuda": torch.version.cuda, "pymunk": pymunk.version,
          "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
          "capability": torch.cuda.get_device_capability() if args.device == "cuda" else None,
          "parameters": sum(p.numel() for p in policy.parameters())}
(core.ROOT / "results/smoke.json").write_text(json.dumps(result, indent=2))
(core.ROOT / "results/smoke_actions.json").write_text(json.dumps(all_actions))
print(json.dumps(result, indent=2))
