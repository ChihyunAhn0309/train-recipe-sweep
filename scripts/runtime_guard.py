"""Local single-host execution guards; not a distributed GPU scheduler."""
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import tempfile
import time


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2)
    fd, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(content + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def verify_assets(manifest, root):
    """Verify paths the loader will consume, never historical local_path fields."""
    root = Path(root).resolve()
    if not isinstance(manifest, dict) or not manifest.get('assets'):
        raise ValueError('A nonempty asset manifest is required')
    verified = {}
    for entry in manifest['assets']:
        kind, filename = entry['kind'], entry['filename']
        if not isinstance(kind, str) or not kind or any(c in kind for c in '/\\:') or kind in ('.', '..'):
            raise ValueError('Invalid asset kind')
        if not isinstance(filename, str) or '\\' in filename or ':' in filename:
            raise ValueError('Invalid asset filename')
        relative = PurePosixPath(filename)
        if relative.is_absolute() or '..' in relative.parts or not relative.parts:
            raise ValueError('Asset path escapes its root')
        path = (root / kind / Path(*relative.parts)).resolve()
        try:
            path.relative_to(root / kind)
        except ValueError:
            raise ValueError('Asset symlink/path escapes its kind directory') from None
        key = (kind, relative.as_posix())
        if key in verified or path in verified.values():
            raise ValueError('Duplicate asset path')
        expected = entry['sha256']
        if not isinstance(expected, str) or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
            raise ValueError('Invalid SHA-256')
        if not path.is_file() or ('bytes' in entry and path.stat().st_size != entry['bytes']) or sha256(path) != expected:
            raise ValueError(f'Asset integrity failure: {kind}/{filename}')
        verified[key] = path
    return verified


class BudgetExceeded(RuntimeError):
    pass


def _read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate state key: {key}')
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f'Non-finite state value: {value}')

    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique,
                      parse_constant=invalid_constant)


def _validate_ledger(ledger):
    if not isinstance(ledger, dict) or set(ledger) != {'attempts'}:
        raise ValueError('Invalid wall ledger object; recover verified accounting before restart')
    attempts = ledger['attempts']
    if not isinstance(attempts, list) or not attempts:
        raise ValueError('Bound study requires a nonempty wall ledger')
    numeric = {'started_unix', 'cap_seconds', 'reserved_seconds', 'elapsed_seconds'}
    statuses = {'active', 'finished', 'failed', 'budget_exhausted', 'crash_reserved_time_estimate'}
    spent = 0.0
    for index, row in enumerate(attempts):
        if not isinstance(row, dict) or set(row) != numeric | {'status'}:
            raise ValueError('Invalid wall ledger attempt schema')
        for key in numeric:
            value = row[key]
            try:
                valid = (not isinstance(value, bool) and isinstance(value, (int, float))
                         and math.isfinite(value) and value >= 0)
            except OverflowError:
                valid = False
            if not valid or (key == 'cap_seconds' and value == 0):
                raise ValueError(f'Invalid nonnegative finite ledger amount: {key}')
        if not isinstance(row['status'], str) or row['status'] not in statuses:
            raise ValueError('Unknown wall ledger status')
        if row['status'] == 'active' and index != len(attempts) - 1:
            raise ValueError('Only the final attempt can remain active')
        expected = max(0, row['cap_seconds'] - spent)
        if not math.isclose(row['reserved_seconds'], expected, rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError('Ledger reservation disagrees with cumulative spending')
        spent += row['elapsed_seconds']
        if not math.isfinite(spent):
            raise ValueError('Cumulative ledger spending overflow')


class LocalStudy:
    """OS lock, immutable output binding, and cumulative wall budget across restarts.

    Only for a local filesystem on one host. A crash charges reserved time until
    recovery, bounded by that attempt's remaining allowance, as a conservative
    estimate. A caller may explicitly raise the total cap; amendments are logged.
    Call check() between bounded work units, including evaluation and preflight.
    """
    def __init__(self, folder, identity, total_seconds):
        if isinstance(total_seconds, bool) or not isinstance(total_seconds, (int, float)) or not math.isfinite(total_seconds) or total_seconds <= 0:
            raise ValueError('Wall budget must be positive and finite')
        if not isinstance(identity, dict) or not identity:
            raise ValueError('Study identity must be nonempty')
        self.folder = Path(folder).resolve()
        self.identity = identity
        self.cap = total_seconds
        self.cleanups = []
        self.lock = None

    def __enter__(self):
        self.folder.mkdir(parents=True, exist_ok=True)
        self.lock = (self.folder / '.study.lock').open('a+b')
        self.lock.seek(0, 2)
        if self.lock.tell() == 0:
            self.lock.write(b'0'); self.lock.flush()
        self.lock.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._bind()
            return self
        except BaseException:
            self.lock.close()
            self.lock = None
            raise

    def _bind(self):
        marker = self.folder / 'run-identity.json'
        self.ledger_path = self.folder / 'wall-budget.json'
        if marker.exists():
            existing = _read_json(marker)
            canonical = lambda data: json.dumps(data, sort_keys=True, allow_nan=False, separators=(',', ':'))
            if canonical(existing) != canonical(self.identity):
                raise ValueError('Output belongs to a different experiment; use a new output directory')
            if not self.ledger_path.is_file():
                raise ValueError('Bound study lost its wall ledger; recover verified accounting before restart')
            self.ledger = _read_json(self.ledger_path)
            _validate_ledger(self.ledger)
        else:
            allowed = {'.study.lock', 'asset-manifest.json'}
            if any(p.name not in allowed for p in self.folder.iterdir()):
                raise ValueError('Unbound nonempty output; historical results will not be overwritten')
            atomic_json(marker, self.identity)
            self.ledger = {'attempts': []}
        attempts = self.ledger['attempts']
        if attempts and attempts[-1]['status'] == 'active':
            last = attempts[-1]
            last['elapsed_seconds'] = max(last['elapsed_seconds'], min(max(0, time.time() - last['started_unix']), last['reserved_seconds']))
            last['status'] = 'crash_reserved_time_estimate'
        self.spent = sum(row['elapsed_seconds'] for row in attempts)
        self.started = time.monotonic()
        self.attempt = {'started_unix': time.time(), 'cap_seconds': self.cap,
                        'reserved_seconds': max(0, self.cap - self.spent),
                        'elapsed_seconds': 0.0, 'status': 'active'}
        attempts.append(self.attempt)
        atomic_json(self.ledger_path, self.ledger)
        try:
            self.check()
        except BudgetExceeded:
            self.attempt['status'] = 'budget_exhausted'
            atomic_json(self.ledger_path, self.ledger)
            raise

    def check(self):
        elapsed = time.monotonic() - self.started
        self.attempt['elapsed_seconds'] = elapsed
        if self.spent + elapsed >= self.cap:
            raise BudgetExceeded('Cumulative wall budget exhausted; resume needs an explicitly larger total cap')

    def __exit__(self, kind, value, traceback):
        cleanup_error = None
        try:
            for cleanup in reversed(self.cleanups):
                try:
                    cleanup()
                except Exception as error:
                    cleanup_error = error
            self.attempt['elapsed_seconds'] = time.monotonic() - self.started
            self.attempt['status'] = ('budget_exhausted' if kind and issubclass(kind, BudgetExceeded)
                                      else 'failed' if kind or cleanup_error else 'finished')
            atomic_json(self.ledger_path, self.ledger)
        finally:
            self.lock.close()
            self.lock = None
        if cleanup_error and kind is None:
            raise cleanup_error
        return False
