"""Package current precommit code with file hashes; no results or historical bundle.

The output is a new runnable snapshot, not a reconstruction of a private run.
Only allowlisted code, tests, and dependency files are included.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (
    "run_precommit.py", "run_precommit_v2.py", "run_precommit_v21.py",
    "run_precommit_v22.py", "summarize_precommit.py", "analyze_precommit_v22.py",
    "audit_precommit_v22_candidates.py", "audit_precommit_v22_carry.py",
    "colab_precommit_v22_suite.py", "package_precommit_code.py",
    "run_precommit_v24.py", "analyze_precommit_v24.py", "colab_precommit_v24_suite.py",
    "run_precommit_v25.py", "analyze_precommit_v25.py", "colab_precommit_v25_suite.py",
    "run_precommit_v26.py", "analyze_precommit_v26.py", "colab_precommit_v26_suite.py",
)


def build(root):
    paths = sorted([*root.glob("src/**/*.py"), *root.glob("tests/test_precommit*.py"),
                    root / "pyproject.toml", root / "requirements/precommit.txt",
                    *(root / "scripts" / name for name in SCRIPTS)])
    payload = {p.relative_to(root).as_posix(): p.read_bytes() for p in paths}
    payload["BUNDLE_MANIFEST.json"] = json.dumps(
        {name: hashlib.sha256(data).hexdigest() for name, data in payload.items()},
        sort_keys=True, indent=2).encode("utf-8")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(payload.items()):
            item = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            item.external_attr = 0o644 << 16
            archive.writestr(item, data)
    return buf.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    data = build(ROOT)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("xb") as f:
        f.write(data)
    print(json.dumps(dict(path=str(args.out), bytes=len(data),
                          sha256=hashlib.sha256(data).hexdigest())))


if __name__ == "__main__":
    main()
