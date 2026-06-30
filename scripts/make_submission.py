"""Create a flat submission.zip containing exactly results.json at the root.

Usage:
    python scripts/make_submission.py --results results.json --out submission.zip
"""
import argparse
import zipfile
from pathlib import Path

import _bootstrap  # noqa: F401


def main() -> None:
    ap = argparse.ArgumentParser(description="Zip results.json (flat).")
    ap.add_argument("--results", default="results.json")
    ap.add_argument("--out", default="submission.zip")
    args = ap.parse_args()

    results = Path(args.results)
    if results.name != "results.json":
        raise SystemExit("Refusing: the file inside the zip MUST be named 'results.json'.")
    if not results.exists():
        raise SystemExit(f"Not found: {results}")

    out = Path(args.out)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        # arcname='results.json' guarantees a flat archive (no subfolders).
        zf.write(results, arcname="results.json")

    # Verify the archive is flat.
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
    assert names == ["results.json"], f"Zip not flat: {names}"
    print(f"[make_submission] created {out} -> contains {names}")


if __name__ == "__main__":
    main()
