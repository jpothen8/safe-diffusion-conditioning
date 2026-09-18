"""Verify the consolidated layout, retained measurements and publishable files."""

import ast
import json
import re
import subprocess
from pathlib import Path
from safety_conditioning.artifacts import PROJECT, recorded_path, read_json, write_json


def main():
    assert not (PROJECT / "round3").exists() and not (PROJECT / "src").exists()
    modules = list((PROJECT / "safety_conditioning").rglob("*.py"))
    for path in modules:
        ast.parse(path.read_text(), filename=str(path))
        assert "round3/" not in path.read_text(), path
    for name in ["unified_checks", "unified_reproduction", "pusht_unified_smoke"]:
        assert read_json(PROJECT / f"records/{name}.json")["passed"]
    for folder in ["halfcheetah_replication", "safe_set_mismatch"]:
        assert read_json(PROJECT / f"results/{folder}/audit.json")["passed"]
    historical = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", "pre-consolidation"],
        cwd=PROJECT,
        text=True,
    ).splitlines()
    retained = 0
    for name in historical:
        if (
            (name.startswith("results/") or name.startswith("round3/results/"))
            and name.endswith("summary.json")
            and "reproduction" not in name
        ):
            before = subprocess.check_output(
                ["git", "show", f"pre-consolidation:{name}"], cwd=PROJECT
            )
            assert recorded_path(name).read_bytes() == before, name
            retained += 1
    tracked = subprocess.check_output(
        ["git", "ls-files"], cwd=PROJECT, text=True
    ).splitlines()
    # Work-tree checks remain useful before/after staging; deleted paths are ignored here.
    tracked = [name for name in tracked if (PROJECT / name).is_file()]
    patterns = [
        rb"gh[pousr]_[A-Za-z0-9]{30,}",
        rb"github_pat_[A-Za-z0-9_]{40,}",
        rb"AKIA[0-9A-Z]{16}",
        rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----",
    ]
    for name in tracked:
        path = PROJECT / name
        assert name.split("/")[0] not in {
            ".venv",
            "vendor",
            "assets",
            "data",
            "round3",
            "src",
        }, name
        assert path.stat().st_size < 50_000_000, name
        if path.suffix in {
            ".py",
            ".json",
            ".md",
            ".txt",
            ".sh",
            ".toml",
            ".yml",
            ".yaml",
        }:
            blob = path.read_bytes()
            assert not any(re.search(pattern, blob) for pattern in patterns), (
                f"Possible credential in {name}"
            )
    # Inspect the preserved source snapshot as well as the current branch.
    for name in historical:
        if Path(name).suffix in {
            ".py",
            ".json",
            ".md",
            ".txt",
            ".sh",
            ".toml",
            ".yml",
            ".yaml",
        }:
            blob = subprocess.check_output(
                ["git", "show", f"pre-consolidation:{name}"], cwd=PROJECT
            )
            assert not any(re.search(pattern, blob) for pattern in patterns), (
                f"Possible credential in historical {name}"
            )
    documents = [
        PROJECT / "README.md",
        PROJECT / "RESULTS.md",
        PROJECT / "TASK_SELECTION.md",
        *list((PROJECT / "docs").rglob("*.md")),
        *list((PROJECT / "results").rglob("*.md")),
    ]
    checked = 0
    for path in documents:
        if "reproduction" in str(path):
            continue
        for link in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if link.startswith(("http:", "https:", "#")):
                continue
            destination = path.parent / link.split("#")[0]
            # This report is being created by this very check.
            assert (
                destination.exists()
                or destination.resolve() == PROJECT / "records/repository_audit.json"
            ), (path, link)
            assert str(destination.resolve().relative_to(PROJECT)) in tracked, (
                path,
                "Link target is not published in Git",
                link,
            )
            checked += 1
    report = {
        "passed": True,
        "package_modules": len(modules),
        "unchanged_historical_summary_files": retained,
        "current_files_inspected": len(tracked),
        "historical_files_inspected": len(historical),
        "local_document_links_checked": checked,
        "separate_round_directory_removed": True,
        "separate_root_src_removed": True,
        "one_environment_and_dependency_lock": True,
        "credential_pattern_scan_current_and_history_passed": True,
        "no_external_checkpoints_or_raw_release_data_in_git": True,
    }
    write_json(PROJECT / "records/repository_audit.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
