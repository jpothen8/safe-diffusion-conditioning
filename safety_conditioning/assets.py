"""Download pinned public assets; do not replace mismatches or distribute weights."""

import subprocess
import tarfile
import urllib.request
from .artifacts import PROJECT, read_json, sha256


def fetch_assets(task="locomotion"):
    for item in read_json(PROJECT / "records/assets.json")["assets"]:
        if task != "all" and item["task_group"] != task:
            continue
        destination = PROJECT / item["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            temp = destination.with_suffix(destination.suffix + ".partial")
            with (
                urllib.request.urlopen(item["url"], timeout=90) as response,
                temp.open("wb") as stream,
            ):
                while block := response.read(1 << 20):
                    stream.write(block)
            if sha256(temp) != item["sha256"]:
                raise ValueError(
                    f"Downloaded asset has an unexpected hash: {item['path']}"
                )
            temp.rename(destination)
        if sha256(destination) != item["sha256"]:
            raise ValueError(f"Existing asset has an unexpected hash: {item['path']}")
        print("Verified", item["path"], flush=True)
    if task in ("all", "pusht"):
        source = read_json(PROJECT / "records/assets.json")["pusht_source"]
        destination = PROJECT / "vendor/diffusion_policy"
        if not destination.exists():
            destination.parent.mkdir(exist_ok=True)
            subprocess.run(
                ["git", "clone", source["url"], str(destination)], check=True
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(destination),
                    "checkout",
                    "--detach",
                    source["revision"],
                ],
                check=True,
            )
        actual = subprocess.check_output(
            ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != source["revision"]:
            raise ValueError(
                "Vendored Push-T source revision differs from the pinned release."
            )


def fetch_results():
    """Restore only manifest-listed generated artifacts from the research release."""
    manifest = read_json(PROJECT / "records/release_artifact.json")
    missing = []
    for name, digest in manifest["files"].items():
        path = PROJECT / name
        if path.exists():
            if sha256(path) != digest:
                raise ValueError(
                    f"Refusing to overwrite a changed research artifact: {name}"
                )
        else:
            missing.append(name)
    if not missing:
        print("All generated research artifacts already match the archive manifest.")
        return
    if manifest.get("publication_status") == "prepared_not_published":
        raise FileNotFoundError(
            "The optional raw-data/model archive has not been published. "
            "Use the collection and training commands in docs/REPRODUCTION.md."
        )
    cache = PROJECT / ".cache" / manifest["filename"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        with (
            urllib.request.urlopen(manifest["url"], timeout=120) as response,
            cache.open("wb") as stream,
        ):
            while block := response.read(1 << 20):
                stream.write(block)
    if sha256(cache) != manifest["sha256"]:
        raise ValueError("Research release archive failed its SHA256 check.")
    with tarfile.open(cache) as archive:
        for member in archive:
            if not member.isfile() or member.name not in manifest["files"]:
                raise ValueError(f"Unexpected archive entry: {member.name}")
            destination = (PROJECT / member.name).resolve()
            destination.relative_to(PROJECT)
            if destination.exists():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with (
                archive.extractfile(member) as source,
                destination.open("wb") as output,
            ):
                while block := source.read(1 << 20):
                    output.write(block)
            if sha256(destination) != manifest["files"][member.name]:
                raise ValueError(
                    f"Restored artifact failed its hash check: {member.name}"
                )
    if any(not (PROJECT / name).exists() for name in missing):
        raise ValueError("Research release archive was incomplete.")
    print(f"Restored {len(missing)} generated artifacts; all hashes verified.")
