"""Capture or verify an immutable, locally recoverable source archive (stdlib only)."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / 'sources/vllm-local-2026-10-07.lock.json'
ARCHIVE = ROOT / 'artifacts/raw/vllm-local-2026-10-07.zip'
SOURCE = ROOT.parent / 'vllm-main'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def inventory(source):
    entries = sorted(source.rglob('*'))
    if any(p.is_symlink() for p in entries):
        raise ValueError('Symlink found: resolve snapshot policy before capture.')
    return [p for p in entries if p.is_file()], [p for p in entries if p.is_dir()]


def capture(source, archive, lock):
    if archive.exists() or lock.exists():
        raise FileExistsError('Archive or lock already exists; never overwrite a snapshot.')
    files, directories = inventory(source)
    archive.parent.mkdir(parents=True, exist_ok=True)
    records = {}
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for path in directories:
            z.writestr(path.relative_to(source).as_posix() + '/', b'')
        for path in files:
            name = path.relative_to(source).as_posix()
            data = path.read_bytes()
            records[name] = {'sha256': sha(data), 'bytes': len(data)}
            z.writestr(name, data)
    result = {
        'schema_version': 1, 'observed_on': '2026-10-07',
        'identity': 'vllm-local-2026-10-07', 'upstream_commit': None,
        'provenance': 'User-provided extracted directory ../vllm-main; upstream acquisition time and commit unknown.',
        'scope': 'All regular files and directories under source root, no exclusions. No symlinks at capture.',
        'archive_relative_to_repo': archive.relative_to(ROOT).as_posix(),
        'archive_sha256': sha(archive.read_bytes()),
        'file_count': len(records), 'total_bytes': sum(r['bytes'] for r in records.values()),
        'directories': [p.relative_to(source).as_posix() for p in directories],
        'files': records,
    }
    with lock.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    verify(source, archive, lock)


def verify(source, archive, lock):
    record = json.loads(lock.read_text(encoding='utf-8'))
    assert sha(archive.read_bytes()) == record['archive_sha256'], 'Archive SHA mismatch'
    files, dirs = inventory(source)
    assert {p.relative_to(source).as_posix() for p in files} == set(record['files']), 'Source file inventory changed'
    assert {p.relative_to(source).as_posix() for p in dirs} == set(record['directories']), 'Source directory inventory changed'
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist()) == set(record['files']) | {p+'/' for p in record['directories']}
        for name, evidence in record['files'].items():
            archived = z.read(name)
            current = (source / name).read_bytes()
            assert sha(archived) == evidence['sha256'] == sha(current), name
            assert len(archived) == evidence['bytes'] == len(current), name
    print(f"PASS: {record['file_count']} files match both complete archive and local tree.")
    print('archive_sha256=' + record['archive_sha256'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['capture', 'verify'])
    p.add_argument('--source', type=Path, default=SOURCE)
    p.add_argument('--archive', type=Path, default=ARCHIVE)
    p.add_argument('--lock', type=Path, default=LOCK)
    args = p.parse_args()
    (capture if args.mode == 'capture' else verify)(args.source.resolve(), args.archive.resolve(), args.lock.resolve())
