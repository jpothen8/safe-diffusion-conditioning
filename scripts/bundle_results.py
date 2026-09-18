"""Package generated research arrays/models; never include external checkpoints."""

import argparse
import tarfile
from pathlib import Path
from safety_conditioning.artifacts import PROJECT, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    files = []
    for folder in [PROJECT / "data", PROJECT / "results"]:
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix in {".npz", ".npy", ".pt"}:
                if not any(
                    "reproduction" in part for part in path.relative_to(PROJECT).parts
                ):
                    files.append(path)
    files += list((PROJECT / "assets").glob("*diffusion*.pt"))
    files = sorted(set(files))
    hashes = {str(path.relative_to(PROJECT)): sha256(path) for path in files}
    with tarfile.open(args.output, "w:gz", compresslevel=1) as archive:
        for path in files:
            info = archive.gettarinfo(path, arcname=str(path.relative_to(PROJECT)))
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with path.open("rb") as stream:
                archive.addfile(info, stream)
    manifest = {
        "filename": args.output.name,
        "url": "https://github.com/jpothen8/safe-diffusion-conditioning/releases/download/v0.2.0/"
        + args.output.name,
        "sha256": sha256(args.output),
        "bytes": args.output.stat().st_size,
        "files": hashes,
        "contents": "Generated episodes, trained diffusion models and simulation outputs. No external checkpoint archives.",
    }
    write_json(PROJECT / "records/release_artifact.json", manifest)
    print(
        {"files": len(files), "bytes": manifest["bytes"], "sha256": manifest["sha256"]},
        flush=True,
    )


if __name__ == "__main__":
    main()
