"""Compare consolidated task reruns with all saved original numerical arrays."""

import numpy as np
from safety_conditioning.artifacts import PROJECT, read_json, write_json

report = {"passed": True, "tasks": {}}
for task, repeated in [
    ("HalfCheetah", "halfcheetah"),
    ("Walker2d", "walker2d"),
    ("pendulum", "pendulum"),
]:
    original = PROJECT / f"results/{task}_comparison"
    rerun = PROJECT / f"results/reproduction_{repeated}"
    count = 0
    arrays = 0
    for path in sorted(original.glob("*.npz")):
        with np.load(path) as old, np.load(rerun / path.name) as new:
            assert set(old.files) <= set(new.files), path.name
            for name in old.files:
                assert np.array_equal(old[name], new[name]), (task, path.name, name)
                arrays += 1
        count += 1
    if (original / "snapshots.json").exists():
        assert read_json(original / "snapshots.json") == read_json(
            rerun / "snapshots.json"
        )
    if (original / "initial_states.npy").exists():
        assert np.array_equal(
            np.load(original / "initial_states.npy"),
            np.load(rerun / "initial_states.npy"),
        )
    report["tasks"][task] = {
        "npz_files": count,
        "arrays": arrays,
        "all_original_arrays_bitwise_equal": True,
        "initial_states_or_full_snapshots_identical": True,
    }
report["pusht_native_smoke"] = read_json(PROJECT / "records/pusht_unified_smoke.json")
write_json(PROJECT / "records/unified_reproduction.json", report)
print(report, flush=True)
