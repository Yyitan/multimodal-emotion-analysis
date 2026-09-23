#!/usr/bin/env python3
import argparse, hashlib
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description='Verify read-only source files against the recorded SHA-256 manifest.')
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--root', type=Path, default=Path('/'))
    args = p.parse_args()
    missing, changed, checked = [], [], 0
    for line in args.manifest.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        expected, raw_path = line.split(maxsplit=1)
        path = Path(raw_path.lstrip('*'))
        if not path.is_absolute():
            path = args.root / path
        if not path.is_file():
            missing.append(str(path)); continue
        checked += 1
        actual = digest(path)
        if actual != expected:
            changed.append((str(path), expected, actual))
    print(f'checked={checked} missing={len(missing)} changed={len(changed)}')
    for x in missing: print(f'MISSING\t{x}')
    for path, exp, got in changed: print(f'CHANGED\t{path}\t{exp}\t{got}')
    return 1 if missing or changed else 0

if __name__ == '__main__':
    raise SystemExit(main())
