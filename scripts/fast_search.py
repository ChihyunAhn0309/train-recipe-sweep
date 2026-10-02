#!/usr/bin/env python3
"""Deterministic, synchronous successive-halving decisions; not a launcher.

Inputs describe one comparable method/fidelity cohort and immutable observations.
The caller owns leases, actual budgets, GPU acceptance and checkpoint semantics.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key: ' + key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON value: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8'),
                      object_pairs_hook=unique, parse_constant=invalid)


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(name + ' must be an integer >= ' + str(minimum))
    return value


def finite(value, name, minimum=None):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(name + ' must be finite and numeric')
    if minimum is not None and value < minimum:
        raise ValueError(name + ' is below its minimum')
    return value


def nonempty(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(name + ' must be a nonempty string')
    return value


def sha_string(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError(name + ' must be a full lowercase SHA-256')


def validate_plan(plan):
    if not isinstance(plan, dict) or plan.get('schema_version') != 1 or type(plan['schema_version']) is not int:
        raise ValueError('Expected fast-search schema_version 1')
    canonical(plan)
    nonempty(plan.get('method'), 'method')
    integer(plan.get('seed'), 'seed')
    sha_string(plan.get('comparison_context_sha256'), 'comparison context')
    metric = plan.get('metric')
    if not isinstance(metric, dict) or metric.get('direction') not in ('min', 'max'):
        raise ValueError('Metric needs min/max direction')
    nonempty(metric.get('name'), 'metric name')
    if plan.get('resource_unit') not in ('updates', 'samples', 'tokens', 'epochs'):
        raise ValueError('Declare one comparable resource unit')
    rungs = plan.get('rungs')
    if not isinstance(rungs, list) or len(rungs) < 2:
        raise ValueError('At least two rungs are required')
    for rung in rungs:
        integer(rung, 'rung', 1)
    if rungs != sorted(set(rungs)):
        raise ValueError('Rungs must be strictly increasing')
    keeps = plan.get('keep_counts')
    if not isinstance(keeps, list) or len(keeps) != len(rungs) - 1:
        raise ValueError('One keep count is required per promotion')
    for count in keeps:
        integer(count, 'keep count', 1)
    if keeps != sorted(keeps, reverse=True):
        raise ValueError('Keep counts must be nonincreasing')
    finite(plan.get('tie_margin'), 'tie margin', 0)
    if plan.get('ranking') not in ('rung_metric', 'best_so_far'):
        raise ValueError('Declare rung_metric or best_so_far ranking before search')
    candidates = plan.get('candidates')
    if not isinstance(candidates, list) or len(candidates) < 2:
        raise ValueError('At least two candidates are required')
    by_id = {}
    baseline = False
    for candidate in candidates:
        if not isinstance(candidate, dict) or not isinstance(candidate.get('config'), dict):
            raise ValueError('Each candidate needs its immutable config')
        config = candidate['config']
        for key in ('method', 'seed', 'comparison_context_sha256', 'metric'):
            if canonical(config.get(key)) != canonical(plan[key]):
                raise ValueError('Candidate differs from cohort ' + key)
        if type(config.get('max_resource')) is not int or config['max_resource'] != rungs[-1]:
            raise ValueError('Candidate must retain the full immutable horizon')
        if not isinstance(config.get('schedule'), dict) or not config['schedule']:
            raise ValueError('Candidate needs its complete immutable schedule')
        roles = candidate.get('roles')
        if (not isinstance(roles, list) or not roles or len(set(roles)) != len(roles)
                or any(role not in ('candidate', 'baseline', 'anchor', 'late_control') for role in roles)):
            raise ValueError('Invalid candidate roles')
        baseline |= 'baseline' in roles
        eligible = integer(candidate.get('prune_after_resource'), 'pruning grace', 1)
        if eligible > rungs[-1]:
            raise ValueError('Pruning grace exceeds full horizon')
        identity = fingerprint(config)
        if identity in by_id:
            raise ValueError('Duplicate config: combine its roles instead of retraining')
        by_id[identity] = candidate
    if not baseline:
        raise ValueError('Protect a source-derived or proposed baseline')
    return by_id


def artifact(reference, root):
    if not isinstance(reference, dict) or set(reference) != {'path', 'sha256', 'bytes'}:
        raise ValueError('Artifact needs path, sha256 and bytes')
    name = nonempty(reference['path'], 'artifact path')
    relative = PurePosixPath(name)
    if ('\\' in name or ':' in name or relative.is_absolute() or '..' in relative.parts
            or relative.as_posix() != name):
        raise ValueError('Artifact path must be contained relative POSIX')
    size = integer(reference['bytes'], 'artifact bytes', 1)
    sha_string(reference['sha256'], 'artifact hash')
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Missing or escaping artifact')
    if path.stat().st_size != size:
        raise ValueError('Artifact byte size mismatch')
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    if hasher.hexdigest() != reference['sha256']:
        raise ValueError('Artifact hash mismatch')


def decide(plan, history, root):
    """Replay a complete observation prefix and propose only the next work.

    Protected controls and candidates inside their grace period are extra to the
    top-k quota. Ties within a predeclared absolute metric margin are also kept.
    A caller must fund those survivors before launching; no budget is increased.
    """
    candidates = validate_plan(plan)
    plan_hash = fingerprint(plan)
    if not isinstance(history, dict) or history.get('plan_sha256') != plan_hash:
        raise ValueError('History belongs to a different immutable plan')
    rows = history.get('observations')
    if not isinstance(rows, list):
        raise ValueError('History observations must be a list')
    canonical(history)
    observed = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Observation must be an object')
        identity, resource = row.get('trial_id'), row.get('resource')
        integer(resource, 'observation resource', 1)
        if identity not in candidates or resource not in plan['rungs']:
            raise ValueError('Unknown candidate or rung')
        key = (identity, resource)
        if key in observed:
            raise ValueError('Duplicate trial/rung observation')
        if row.get('status') == 'failed':
            nonempty(row.get('reason'), 'failure reason')
        elif row.get('status') == 'ok':
            finite(row.get('metric'), 'rung metric')
            finite(row.get('best_metric'), 'best metric')
            best_at = integer(row.get('best_resource'), 'best checkpoint resource', 1)
            if best_at > resource:
                raise ValueError('Best checkpoint lies beyond observed resource')
            if ((plan['metric']['direction'] == 'max' and row['best_metric'] < row['metric'])
                    or (plan['metric']['direction'] == 'min' and row['best_metric'] > row['metric'])):
                raise ValueError('Best metric is worse than current metric')
            for role in ('resume_checkpoint', 'best_checkpoint', 'metrics'):
                artifact(row.get(role), root)
        else:
            raise ValueError('Observation status must be ok or failed')
        observed[key] = row

    result = {'schema_version': 1, 'plan_sha256': plan_hash,
              'history_sha256': fingerprint(history), 'method': plan['method'],
              'single_seed': True, 'fresh_seed_confirmation_complete': False,
              'pruned': [], 'failed': [], 'rounds': [], 'actions': []}
    active = sorted(candidates)
    previous = {}
    consumed = set()
    direction = -1 if plan['metric']['direction'] == 'max' else 1
    ranking_key = 'metric' if plan['ranking'] == 'rung_metric' else 'best_metric'
    valid_rows = [row for row in rows if row['status'] == 'ok']
    if valid_rows:
        best = min(valid_rows, key=lambda row: (direction * row['best_metric'], row['trial_id'], row['resource']))
        result['best_observed_any_fidelity'] = {key: best[key] for key in
            ('trial_id', 'best_metric', 'best_resource', 'best_checkpoint')}
        result['best_observed_any_fidelity']['reached_final_fidelity'] = (
            best['trial_id'], plan['rungs'][-1]) in observed and observed[(best['trial_id'], plan['rungs'][-1])]['status'] == 'ok'
    def protected(identity):
        return any(role in candidates[identity]['roles'] for role in ('baseline', 'anchor', 'late_control'))
    def order(identity, at, metric='metric'):
        return (direction * observed[(identity, at)][metric], identity)
    for index, resource in enumerate(plan['rungs']):
        missing = [identity for identity in active if (identity, resource) not in observed]
        current_keys = {(identity, resource) for identity in active if (identity, resource) in observed}
        consumed |= current_keys
        for identity, at in current_keys:
            current, prior = observed[(identity, at)], previous.get(identity)
            if prior is None or current['status'] != 'ok':
                continue
            if direction * current['best_metric'] > direction * prior['best_metric']:
                raise ValueError('Best metric regressed across continuation')
            if current['best_resource'] <= prior['resource'] and (
                current['best_metric'] != prior['best_metric'] or current['best_resource'] != prior['best_resource']
                or any(current['best_checkpoint'][key] != prior['best_checkpoint'][key] for key in ('sha256', 'bytes'))):
                raise ValueError('Historical best checkpoint changed across continuation')
        if missing:
            if set(observed) - consumed:
                raise ValueError('Future or pruned observations bypass the synchronous barrier')
            result['status'] = 'needs_work'
            for identity in missing:
                prior = previous.get(identity)
                result['actions'].append({
                    'trial_id': identity, 'target_resource': resource,
                    'from_resource': prior['resource'] if prior else 0,
                    'resume_checkpoint': prior['resume_checkpoint'] if prior else None,
                    'config_sha256': identity,
                })
            return result
        failures = [identity for identity in active if observed[(identity, resource)]['status'] == 'failed']
        result['failed'].extend({'trial_id': identity, 'resource': resource,
                                 'reason': observed[(identity, resource)]['reason']} for identity in failures)
        if any(protected(identity) for identity in failures):
            if set(observed) - consumed:
                raise ValueError('Observations continue beyond a failed control')
            result['status'] = 'blocked_control_failure'
            return result
        active = [identity for identity in active if identity not in failures]
        if index == len(plan['rungs']) - 1:
            if set(observed) != consumed:
                raise ValueError('Unexpected observations outside admitted lineage')
            ranked = sorted(active, key=lambda identity: order(identity, resource, 'best_metric'))
            result['status'] = 'complete'
            result['final_ranking'] = ranked
            winner = observed[(ranked[0], resource)]
            result['winner'] = {key: winner[key] for key in
                                ('trial_id', 'best_metric', 'best_resource', 'best_checkpoint')}
            result['winner']['claim'] = 'best_observed_single_seed_at_declared_final_fidelity'
            result['better_nonfinal_checkpoint_exists'] = (
                direction * result['best_observed_any_fidelity']['best_metric']
                < direction * winner['best_metric'] - plan['tie_margin'])
            return result
        guarded = [identity for identity in active if protected(identity)
                   or resource < candidates[identity]['prune_after_resource']]
        pool = sorted((identity for identity in active if identity not in guarded),
                      key=lambda identity: order(identity, resource, ranking_key))
        count = min(plan['keep_counts'][index], len(pool))
        kept = pool[:count]
        if kept:
            cutoff = observed[(kept[-1], resource)][ranking_key]
            kept.extend(identity for identity in pool[count:]
                        if abs(observed[(identity, resource)][ranking_key] - cutoff) <= plan['tie_margin'])
        survivors = sorted(set(guarded + kept))
        removed = sorted(set(active) - set(survivors))
        result['pruned'].extend({'trial_id': identity, 'resource': resource,
                                'reason': 'rung allocation decision; not convergence'} for identity in removed)
        result['rounds'].append({'resource': resource, 'guarded': guarded,
                                 'survivors': survivors, 'pruned': removed,
                                 'top_k_quota': plan['keep_counts'][index]})
        previous = {identity: observed[(identity, resource)] for identity in survivors}
        active = survivors
    raise AssertionError('Unreachable')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan')
    parser.add_argument('--history')
    parser.add_argument('--artifact-root', default='.')
    args = parser.parse_args()
    plan = read_json(args.plan)
    validate_plan(plan)
    result = (decide(plan, read_json(args.history), args.artifact_root)
              if args.history else {'plan_sha256': fingerprint(plan), 'candidate_ids': sorted(validate_plan(plan))})
    print(json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False))


if __name__ == '__main__':
    main()
