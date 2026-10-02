"""Reconcile source recipe leaves, per-method decisions and compiled trial coverage."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

from sweep_guard import parse_json


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def leaves(value, pointer=''):
    if isinstance(value, dict) and value:
        return {p: v for key, child in value.items()
                for p, v in leaves(child, pointer + '/' + key.replace('~', '~0').replace('/', '~1')).items()}
    if isinstance(value, list) and value:
        return {p: v for index, child in enumerate(value)
                for p, v in leaves(child, pointer + '/' + str(index)).items()}
    return {pointer: value}


def validate_pointer(pointer):
    if not isinstance(pointer, str) or not pointer.startswith('/') or re.search(r'~(?:[^01]|$)', pointer):
        raise ValueError('Target pointers must be non-root JSON pointers')


def resolve(value, pointer):
    validate_pointer(pointer)
    for part in pointer[1:].split('/'):
        part = part.replace('~1', '/').replace('~0', '~')
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif isinstance(value, list) and part.isdigit() and str(int(part)) == part and int(part) < len(value):
            value = value[int(part)]
        else:
            raise ValueError('Missing target config pointer: ' + pointer)
    return value


def required_text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(name + ' must be nonempty text')


def inventory(source):
    raw = Path(source).read_bytes()
    value = parse_json(raw.decode('utf-8'))
    if not isinstance(value, dict) or not value:
        raise ValueError('Source recipe must be a nonempty JSON object')
    return {'source_sha256': hashlib.sha256(raw).hexdigest(), 'leaves': leaves(value)}


def check(source, decisions, trials=None):
    original = inventory(source)
    if not isinstance(decisions, dict) or decisions.get('schema_version') != 1:
        raise ValueError('Expected decisions schema_version=1')
    if decisions.get('source_sha256') != original['source_sha256']:
        raise ValueError('Source recipe hash changed')
    methods = decisions.get('methods')
    if not isinstance(methods, list) or not methods or any(not isinstance(m, str) or not m.strip() for m in methods):
        raise ValueError('A nonempty requested method list is required')
    if len(set(methods)) != len(methods):
        raise ValueError('Duplicate requested method')
    rows = decisions.get('rows')
    if not isinstance(rows, list) or not rows:
        raise ValueError('A nonempty factor ledger is required')
    trial_rows = None
    if trials is not None:
        trial_rows = trials.get('trials') if isinstance(trials, dict) else None
        if not isinstance(trial_rows, list) or not trial_rows:
            raise ValueError('Compiled study must contain nonempty trials')
        if any(not isinstance(t, dict) or t.get('method') not in methods or not isinstance(t.get('config'), dict)
               or t['config'].get('method') != t['method'] for t in trial_rows):
            raise ValueError('Every compiled trial needs a requested method and a matching config.method')
        if {t['method'] for t in trial_rows} != set(methods):
            raise ValueError('Compiled trials omit a requested method')
    covered, factors, unknown, exceptions = set(), set(), [], []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Factor rows must be objects')
        factor = row.get('factor')
        required_text(factor, 'factor')
        if factor in factors:
            raise ValueError('Duplicate factor: ' + factor)
        factors.add(factor)
        required_text(row.get('source_stage'), 'source_stage')
        evidence = row.get('evidence')
        if not isinstance(evidence, list) or not evidence:
            raise ValueError('Source evidence locations are required')
        for item in evidence:
            required_text(item, 'evidence location')
        if row.get('evidence_status') not in {'verified', 'derived', 'proposed', 'unknown'}:
            raise ValueError('Invalid source evidence status')
        if row['evidence_status'] != 'verified':
            unknown.append(factor)
        keys = row.get('source_keys')
        if not isinstance(keys, list) or not keys or any(not isinstance(k, str) for k in keys):
            raise ValueError('Every factor needs source leaf keys')
        for key in keys:
            if key not in original['leaves'] or key in covered:
                raise ValueError('Unknown or multiply mapped source key: ' + key)
            covered.add(key)
        if not isinstance(row.get('ablatable'), bool):
            raise ValueError('ablatable must explicitly be boolean')
        choices = row.get('choices')
        if not isinstance(choices, dict) or set(choices) != set(methods):
            raise ValueError('Every factor needs exactly all requested method choices')
        for method, choice in choices.items():
            if not isinstance(choice, dict):
                raise ValueError('Method choice must be an object')
            disposition = choice.get('disposition')
            if disposition not in {'sweep', 'fixed', 'conditional', 'excluded'}:
                raise ValueError('Invalid factor disposition')
            required_text(choice.get('reason'), 'decision reason')
            candidates = choice.get('candidates')
            if not isinstance(candidates, list):
                raise ValueError('Candidates must be an explicit list')
            encoded = [canonical(c) for c in candidates]
            if len(set(encoded)) != len(encoded):
                raise ValueError('Duplicate factor candidates')
            if disposition == 'excluded':
                if candidates:
                    raise ValueError('Excluded factors cannot declare candidates')
            elif not candidates or (disposition == 'fixed' and len(candidates) != 1) or (disposition == 'sweep' and len(candidates) < 2):
                raise ValueError('Candidate count conflicts with disposition')
            if row['ablatable']:
                enabled = [c if isinstance(c, bool) else c.get('enabled') if isinstance(c, dict) else None for c in candidates]
                if any(not isinstance(flag, bool) for flag in enabled):
                    raise ValueError('Ablatable candidates need explicit boolean enabled states')
                if set(enabled) != {False, True}:
                    required_text(choice.get('ablation_reason'), 'reason for not testing both enabled and disabled')
                    exceptions.append({'factor': factor, 'method': method, 'reason': choice['ablation_reason']})
            if disposition == 'excluded':
                if 'target_pointer' in choice or 'when' in choice:
                    raise ValueError('Excluded choices must use exclusion_checks, not unchecked target_pointer/when')
                assertions = choice.get('exclusion_checks')
                if not isinstance(assertions, list) or not assertions:
                    raise ValueError('Excluded choices require explicit absence or inactive-value assertions')
                for assertion in assertions:
                    if not isinstance(assertion, dict):
                        raise ValueError('Exclusion assertions must be objects')
                    validate_pointer(assertion.get('pointer'))
                    absent = set(assertion) == {'pointer', 'absent'} and assertion['absent'] is True
                    equal = set(assertion) == {'pointer', 'equals'}
                    if not absent and not equal:
                        raise ValueError('An exclusion assertion needs exactly absent=true or equals')
                    if trial_rows is not None:
                        for trial in trial_rows:
                            if trial['method'] != method:
                                continue
                            try:
                                actual = resolve(trial['config'], assertion['pointer'])
                            except ValueError:
                                if absent:
                                    continue
                                raise
                            if absent or canonical(actual) != canonical(assertion['equals']):
                                raise ValueError(f'Excluded factor remains present or conflicts with inactive value: {factor} / {method}')
                continue
            pointer = choice.get('target_pointer')
            validate_pointer(pointer)
            conditions = choice.get('when', [])
            if not isinstance(conditions, list):
                raise ValueError('when must be a list of explicit pointer/equality conditions')
            if disposition == 'conditional' and not conditions:
                raise ValueError('Conditional factors need a concrete condition')
            for condition in conditions:
                if not isinstance(condition, dict) or not isinstance(condition.get('pointer'), str) or not condition['pointer'].startswith('/') or 'equals' not in condition:
                    raise ValueError('Invalid condition')
                validate_pointer(condition['pointer'])
            if trial_rows is not None:
                observed = set()
                for trial in trial_rows:
                    if trial['method'] != method:
                        continue
                    config = trial['config']
                    if not all(canonical(resolve(config, c['pointer'])) == canonical(c['equals']) for c in conditions):
                        continue
                    observed.add(canonical(resolve(config, pointer)))
                if observed != set(encoded):
                    raise ValueError(f'Compiled candidate coverage mismatch: {factor} / {method}')
    missing = sorted(set(original['leaves']) - covered)
    if missing:
        raise ValueError('Unreviewed source recipe leaves: ' + ', '.join(missing))
    return {'status': 'passed', 'source_sha256': original['source_sha256'],
            'source_leaves_reviewed': len(covered), 'factors': len(factors), 'methods': methods,
            'candidate_coverage_verified': trial_rows is not None,
            'nonverified_source_factors': unknown, 'ablation_exceptions': exceptions,
            'source_recovery_completeness_verified': False, 'trainer_behavior_verified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('inventory', 'check'))
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--decisions', type=Path)
    parser.add_argument('--trials', type=Path)
    args = parser.parse_args()
    try:
        if args.command == 'inventory':
            result = inventory(args.source)
        else:
            if args.decisions is None:
                raise ValueError('--decisions is required for check')
            decisions = parse_json(args.decisions.read_text(encoding='utf-8'))
            trials = parse_json(args.trials.read_text(encoding='utf-8')) if args.trials else None
            result = check(args.source, decisions, trials)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, TypeError) as error:
        print(json.dumps({'status': 'invalid_recipe_coverage', 'error': str(error)}), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
