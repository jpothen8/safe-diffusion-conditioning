"""Verify the new delivery and preservation of every historical artifact."""
import ast
import json
import re
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from safety_conditioning.artifacts import PROJECT, read_json, sha256, verify_lock, write_json


def main():
    report = {"utc": datetime.now(timezone.utc).isoformat(), "historical_manifests": {}}
    for name, base, allowed in [
        ("records/result_hashes.json", PROJECT, set()),
        ("records/followup_result_hashes.json", PROJECT, {"README.md", "RESULTS.md", "TASK_SELECTION.md"}),
        ("round3/records/delivery_hashes.json", PROJECT / "round3", set()),
    ]:
        manifest = read_json(PROJECT / name)
        changed = [path for path, digest in manifest.items() if sha256(base / path) != digest]
        assert set(changed) <= allowed, (name, changed)
        report["historical_manifests"][name] = {"checked": len(manifest), "updated_navigation_docs": changed}
    output = PROJECT / "results/halfcheetah_replication"
    report["study_locked_inputs"] = verify_lock(output / "lock.json")
    assert read_json(output / "audit.json")["passed"]
    mismatch = PROJECT / "results/safe_set_mismatch"
    report["mismatch_locked_inputs"] = verify_lock(mismatch / "lock.json")
    assert read_json(mismatch / "audit.json")["passed"]
    # Preserve the previous delivery record and verify that its trained models,
    # data, outputs and scientific study implementation were not altered.
    previous = read_json(PROJECT / "docs/history/2026-09-17_consolidation_hashes.json")
    allowed = {"README.md", "RESULTS.md", "docs/PAPER_INTEGRATION.md", "docs/ARTIFACTS.md",
               "safety_conditioning/cli.py", "scripts/verify_consolidation.py"}
    changes = [name for name, digest in previous.items() if sha256(PROJECT / name) != digest]
    assert all(name in allowed or name.startswith("docs/paper_preview/") for name in changes), changes
    report["previous_delivery"] = {"checked": len(previous), "updated_navigation_cli_and_paper_files": changes,
                                   "scientific_study_code_and_results_unchanged": True}
    assert read_json(PROJECT / "records/consolidation_checks.json")["passed"]
    assert read_json(PROJECT / "results/collection_reproduction/audit.json")["complete_episode_archive_byte_identical"]
    original = PROJECT / "round3/results/HalfCheetah_comparison"
    copied = PROJECT / "results/cli_reproduction/round3/results/HalfCheetah_comparison"
    files = list(original.glob("*.npz"))
    for path in files:
        with np.load(path) as first, np.load(copied / path.name) as second:
            assert first.files == second.files
            assert all(np.array_equal(first[key], second[key]) for key in first.files), path.name
    report["historical_cli_reproduction_npz_files_exact"] = len(files)
    assert read_json(original / "snapshots.json") == read_json(copied / "snapshots.json")
    write_json(PROJECT / "results/cli_reproduction/audit.json", {"all_arrays_exact": True, "files": len(files), "snapshots_exact": True})
    modules = list((PROJECT / "safety_conditioning").glob("*.py"))
    for path in modules:
        ast.parse(path.read_text(), filename=str(path))
    report["package_modules_parse"] = len(modules)
    documents = [PROJECT / "README.md", PROJECT / "RESULTS.md", PROJECT / "TASK_SELECTION.md",
                 *list((PROJECT / "docs").rglob("*.md")), output / "RESULTS.md", mismatch / "RESULTS.md"]
    for document in documents:
        for target in re.findall(r"\]\(([^)]+)\)", document.read_text()):
            if target.startswith(("http:", "https:", "#")):
                continue
            assert (document.parent / target.split("#")[0]).exists(), (document, target)
    report["document_links_valid"] = True
    log = (PROJECT / "docs/paper_preview/experiments.log").read_text()
    assert "Output written on" in log and "Overfull" not in log and "undefined" not in log
    report["paper_preview_compiled_without_overfull_or_undefined_items"] = True
    report["passed"] = True
    write_json(PROJECT / "records/consolidation_audit.json", report)
    paths = [*modules, PROJECT / "pyproject.toml", PROJECT / "run", PROJECT / "README.md", PROJECT / "RESULTS.md",
             PROJECT / "configs/halfcheetah_replication.json", PROJECT / "scripts/setup_project.sh", Path(__file__).resolve(),
             *[p for p in (PROJECT / "docs").rglob("*") if p.is_file()],
             *[p for p in output.rglob("*") if p.is_file()],
             *[p for p in mismatch.rglob("*") if p.is_file()]]
    write_json(PROJECT / "records/consolidation_hashes.json", {str(p.relative_to(PROJECT)): sha256(p) for p in sorted(set(paths))})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
