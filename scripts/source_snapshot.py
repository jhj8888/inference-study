"""Record or verify SHA-256 hashes of the selected local reading-entry files."""
import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def collect():
    catalog = load_json(REPO / "sources/catalog.json")
    overrides_path = REPO / "sources/local-paths.json"
    overrides = load_json(overrides_path) if overrides_path.exists() else {}
    projects = []
    for project in catalog["projects"]:
        root = Path(overrides.get(project["id"], REPO / project["relative_root"])).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Source root missing: {project['id']}: {root}")
        commit = None
        if (root / ".git").exists():
            commit = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
            ).strip()
        entries = {}
        for name in project["files"]:
            path = root / name
            content = path.read_bytes()
            entries[name] = {
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
            }
        projects.append({
            "id": project["id"],
            "upstream_commit": commit,
            "files": entries,
        })
    return {"schema_version": 1, "scope": catalog["scope"], "projects": projects}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture", help="Write a new baseline; never overwrite.")
    capture.add_argument("--output", required=True, type=Path)
    capture.add_argument("--observed-on", required=True, type=date.fromisoformat)
    verify = commands.add_parser("verify", help="Compare current files with a baseline.")
    verify.add_argument(
        "--baseline", type=Path,
        default=REPO / "sources/baseline-2026-10-07-w03-guides.json"
    )
    args = parser.parse_args()
    try:
        actual = collect()
        if args.command == "capture":
            actual["observed_on"] = args.observed_on.isoformat()
            with args.output.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(actual, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            count = sum(len(p["files"]) for p in actual["projects"])
            print(f"Recorded {len(actual['projects'])} projects, {count} selected files.")
            return 0
        expected = load_json(args.baseline)
        expected.pop("observed_on", None)
        if expected != actual:
            print("SOURCE BASELINE MISMATCH: selected files, Git commit, or catalog changed.")
            print("Review the changes before capturing a separately named baseline.")
            before = {p["id"]: p for p in expected["projects"]}
            after = {p["id"]: p for p in actual["projects"]}
            for key in sorted(before.keys() | after.keys()):
                if before.get(key) != after.get(key):
                    print(f"  changed project: {key}")
            return 1
        print("PASS: all selected source files match the baseline.")
        print("Scope: selected entries only; this does not verify the whole source tree.")
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
