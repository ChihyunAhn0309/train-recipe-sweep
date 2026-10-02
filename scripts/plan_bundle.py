"""Pack, verify and safely reconstruct a portable recipe-plan document. Never executes it."""
import argparse
import base64
import binascii
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys

from sweep_guard import parse_json
from runtime_guard import atomic_json

START = '<!-- TRAIN_RECIPE_BUNDLE_V1 -->'
END = '<!-- END_TRAIN_RECIPE_BUNDLE_V1 -->'
PHASES = {'prepare', 'verify_target', 'profile', 'run', 'resume'}
CHECKS = {'hardware_environment', 'assets_and_gt', 'model_smoke',
          'controller_recovery', 'performance_profile', 'budget_feasibility'}
SCIENCE = {'model', 'data', 'methods', 'search', 'convergence', 'budgets', 'selection', 'sources'}
MAX_BYTES = 16 * 1024 * 1024
RESERVED = {'CON', 'PRN', 'AUX', 'NUL'} | {f'{p}{i}' for p in ('COM', 'LPT') for i in range(1, 10)}


def safe_path(name):
    if not isinstance(name, str) or not name or any(c in name for c in '\\:<>"|?*') or any(ord(c)<32 for c in name):
        raise ValueError('File paths must be portable relative POSIX paths')
    path = PurePosixPath(name)
    if not path.parts or path.is_absolute() or path.as_posix() != name or '..' in path.parts:
        raise ValueError(f'Noncanonical/escaping file path: {name}')
    for part in path.parts:
        if part in ('.', '..') or part[-1] in ' .' or part.split('.')[0].upper() in RESERVED:
            raise ValueError(f'Nonportable filename: {name}')
    if path.parts[0].casefold() == 'portable-plan-manifest.json':
        raise ValueError('Reserved manifest filename')
    return path


def validate_spec(spec, paths):
    if not isinstance(spec, dict) or type(spec.get('schema_version')) is not int or spec['schema_version'] != 1:
        raise ValueError('schema_version must be integer 1')
    if not isinstance(spec.get('plan_id'), str) or not spec['plan_id'].strip():
        raise ValueError('plan_id required')
    if not isinstance(spec.get('purpose'), str) or not spec['purpose'].strip():
        raise ValueError('purpose required')
    science = spec.get('scientific_spec')
    if not isinstance(science, dict) or not SCIENCE <= science.keys() or any(not science[k] for k in SCIENCE):
        raise ValueError('Scientific model/data/method/search/convergence/budget/selection/source contract incomplete')
    if not isinstance(spec.get('external_inputs'), list):
        raise ValueError('Declare external_inputs, even when empty')
    checks = spec.get('target_checks')
    if not isinstance(checks, dict) or not CHECKS <= checks.keys() or any(not checks[k] for k in CHECKS):
        raise ValueError('All six target acceptance checks must be specified')
    entrypoints = spec.get('entrypoints')
    if not isinstance(entrypoints, dict) or not PHASES <= entrypoints.keys():
        raise ValueError('prepare/verify_target/profile/run/resume entrypoints required')
    for phase, argv in entrypoints.items():
        if not isinstance(argv, list) or len(argv) < 2 or not all(isinstance(a, str) and a for a in argv):
            raise ValueError(f'Invalid argv for {phase}')
        if argv[0] != '{python}' or argv[1] not in paths:
            raise ValueError(f'Entrypoint must reference a bundled script: {phase}')
        if argv[1].startswith('-'):
            raise ValueError(f'Entrypoint cannot be interpreted as a Python option: {phase}')
        if any('\x00' in item for item in argv):
            raise ValueError('NUL in command argument')


def validate_payload(payload):
    if not isinstance(payload, dict) or set(payload) != {'spec', 'files'}:
        raise ValueError('Invalid bundle object')
    entries = payload['files']
    if not isinstance(entries, list) or not 1 <= len(entries) <= 256:
        raise ValueError('Bundle requires 1..256 text files')
    decoded = {}; folded = set(); spellings = {}; total = 0
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {'path', 'sha256', 'bytes', 'base64'}:
            raise ValueError('Invalid file record')
        name = entry['path']; safe_path(name)
        if name.casefold() in folded:
            raise ValueError('Duplicate/case-colliding file path')
        folded.add(name.casefold())
        for prefix in (PurePosixPath(name), *PurePosixPath(name).parents):
            if prefix.as_posix() == '.':
                continue
            spelling = prefix.as_posix()
            key = spelling.casefold()
            if key in spellings and spellings[key] != spelling:
                raise ValueError('Case-colliding file/directory spelling')
            spellings[key] = spelling
        if not isinstance(entry['bytes'], int) or isinstance(entry['bytes'], bool) or entry['bytes'] < 0:
            raise ValueError('Invalid file byte count')
        try:
            data = base64.b64decode(entry['base64'], validate=True)
            text = data.decode('utf-8')
        except (ValueError, TypeError, binascii.Error, UnicodeDecodeError):
            raise ValueError(f'Invalid UTF-8/base64 content: {name}') from None
        if '\x00' in text or len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise ValueError(f'File integrity failure: {name}')
        total += len(data)
        if total > MAX_BYTES:
            raise ValueError('Bundle text payload exceeds limit; reference pinned external assets instead')
        decoded[name] = data
    for name in decoded:
        if any(parent.as_posix().casefold() in folded for parent in PurePosixPath(name).parents if parent.as_posix() != '.'):
            raise ValueError('A file path is also a parent directory')
    validate_spec(payload['spec'], set(decoded))
    return decoded


def read_document(path):
    if Path(path).stat().st_size > MAX_BYTES * 2:
        raise ValueError('Plan document exceeds text size limit')
    text = Path(path).read_text(encoding='utf-8-sig')
    if text.count(START) != 1 or text.count(END) != 1:
        raise ValueError('Exactly one complete portable bundle is required')
    middle = text.split(START, 1)[1].split(END, 1)[0].strip()
    match = re.fullmatch(r'```json\s*\n([\s\S]+)\n```', middle)
    if not match:
        raise ValueError('Malformed portable JSON fence')
    payload = parse_json(match.group(1))
    decoded = validate_payload(payload)
    return payload, decoded


def pack(spec_path, files_root, overview_path, output):
    root = Path(files_root).resolve()
    entries = []
    total = 0
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('Bundle source may not contain symlinks')
        if not path.is_file():
            continue
        if not path.resolve().is_relative_to(root):
            raise ValueError('Bundle source escapes root')
        relative = path.relative_to(root).as_posix(); safe_path(relative)
        total += path.stat().st_size
        if total > MAX_BYTES or len(entries) >= 256:
            raise ValueError('Source package exceeds text bundle limits')
        data = path.read_bytes()
        entries.append({'path': relative, 'sha256': hashlib.sha256(data).hexdigest(),
                        'bytes': len(data), 'base64': base64.b64encode(data).decode('ascii')})
    payload = {'spec': parse_json(Path(spec_path).read_text(encoding='utf-8-sig')), 'files': entries}
    validate_payload(payload)
    overview = Path(overview_path).read_text(encoding='utf-8')
    if START in overview or END in overview:
        raise ValueError('Overview contains a reserved bundle marker')
    document = (overview.rstrip() + '\n\n<details><summary>Portable execution files (verify before execution)</summary>\n\n'
                + START + '\n```json\n' + json.dumps(payload, ensure_ascii=False, indent=2)
                + '\n```\n' + END + '\n\n</details>\n')
    if len(document.encode('utf-8')) > MAX_BYTES * 2:
        raise ValueError('Plan document exceeds text size limit')
    destination = Path(output)
    with destination.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(document)
    return {'plan_id': payload['spec']['plan_id'], 'file_count': len(entries),
            'document_sha256': hashlib.sha256(destination.read_bytes()).hexdigest()}


def extract(document, output):
    payload, decoded = read_document(document)  # Validate everything before creating any file.
    out = Path(output).absolute()
    if out.exists() or out.is_symlink():
        raise ValueError('Extraction destination must not exist; never overwrite an earlier study')
    if any(parent.is_symlink() for parent in out.parents):
        raise ValueError('Extraction parent is a symlink')
    out.mkdir(parents=True, exist_ok=False)
    for name, data in decoded.items():
        path = out.joinpath(*PurePosixPath(name).parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        if any(parent.is_symlink() for parent in (path.parent, *path.parent.parents)):
            raise ValueError('Extraction parent is a symlink')
        if not path.resolve().is_relative_to(out.resolve()):
            raise ValueError('Extraction path escaped destination')
        with path.open('xb') as stream:
            stream.write(data)
    manifest = {'spec': payload['spec'], 'files': [{k:v for k,v in row.items() if k!='base64'} for row in payload['files']],
                'source_document_sha256': hashlib.sha256(Path(document).read_bytes()).hexdigest(),
                'reconstructed': True, 'code_executed': False, 'target_verified': False}
    atomic_json(out/'portable-plan-manifest.json', manifest)
    return {'plan_id': payload['spec']['plan_id'], 'file_count': len(decoded),
            'output': str(out), 'code_executed': False, 'target_verified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    packing = commands.add_parser('pack')
    for name in ('spec', 'files', 'overview', 'output'):
        packing.add_argument('--'+name, required=True, type=Path)
    verifying = commands.add_parser('verify'); verifying.add_argument('document', type=Path)
    extracting = commands.add_parser('extract'); extracting.add_argument('document', type=Path)
    extracting.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == 'pack':
            result = pack(args.spec, args.files, args.overview, args.output)
        elif args.command == 'extract':
            result = extract(args.document, args.output)
        else:
            payload, decoded = read_document(args.document)
            result = {'plan_id':payload['spec']['plan_id'], 'file_count':len(decoded), 'transport_valid':True,
                      'semantic_training_correctness_verified':False, 'target_verified':False}
        print(json.dumps(result, ensure_ascii=False, indent=2)); return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps({'status':'invalid_plan','error':str(error)},ensure_ascii=False),file=sys.stderr); return 2


if __name__ == '__main__':
    sys.exit(main())
