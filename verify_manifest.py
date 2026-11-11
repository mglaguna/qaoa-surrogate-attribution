"""Check every file of the repository against MANIFEST_SHA256.json.

Usage (from the repository root): python verify_manifest.py
"""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / "MANIFEST_SHA256.json").read_text(encoding="utf-8"))["files"]
bad, missing = [], []
for rel, digest in manifest.items():
    p = root / rel
    if not p.is_file():
        missing.append(rel)
    elif hashlib.sha256(p.read_bytes()).hexdigest() != digest:
        bad.append(rel)
print(f"{len(manifest)} files listed; {len(missing)} missing; {len(bad)} with a different SHA-256")
for rel in missing + bad:
    print("  ", rel)
raise SystemExit(1 if bad or missing else 0)
