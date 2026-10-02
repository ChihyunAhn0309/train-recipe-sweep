"""Tiny real CPU classifier demo; generated data, no pretrained model or GPU."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from fast_search import decide, fingerprint, read_json, validate_plan


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


def reference(path, root):
    data = path.read_bytes()
    return {'path': path.relative_to(root).as_posix(), 'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest()}


def tuples(value):
    return tuple(tuples(v) for v in value) if isinstance(value, list) else value


def data():
    rng = random.Random(20261003)
    rows = []
    for _ in range(192):
        a, b = rng.gauss(0, 1), rng.gauss(0, 1)
        label = int(1.7 * a - b + rng.gauss(0, 0.7) > 0)
        rows.append(([a, b, 1.0], label))
    return rows[:128], rows[128:]


def sigmoid(value):
    return 1 / (1 + math.exp(-max(-35.0, min(35.0, value))))


def loss(weights, rows):
    result = 0.0
    for features, label in rows:
        score = sigmoid(sum(x * w for x, w in zip(features, weights)))
        result -= label * math.log(max(score, 1e-15)) + (1 - label) * math.log(max(1 - score, 1e-15))
    return result / len(rows)


def initial(config):
    rng = random.Random(config['seed'])
    return {'trial_id': fingerprint(config), 'resource': 0, 'updates': 0,
            'weights': [rng.gauss(0, 0.01) for _ in range(3)], 'velocity': [0.0] * 3,
            'rng': rng.getstate(), 'best_metric': None, 'best_resource': None,
            'best_weights': None, 'history': []}


def train_to(config, state, target, train, validation):
    if state['trial_id'] != fingerprint(config) or not state['resource'] < target <= config['max_resource']:
        raise ValueError('Incompatible checkpoint or continuation target')
    state = copy.deepcopy(state)
    rng = random.Random()
    rng.setstate(tuples(state['rng']))
    updates_per_epoch = len(train) // config['batch_size']
    full_steps = config['max_resource'] * updates_per_epoch
    for epoch in range(state['resource'] + 1, target + 1):
        order = list(range(len(train)))
        rng.shuffle(order)
        for start in range(0, len(train), config['batch_size']):
            batch = [train[i] for i in order[start:start + config['batch_size']]]
            gradient = [0.0] * 3
            for features, label in batch:
                error = sigmoid(sum(x * w for x, w in zip(features, state['weights']))) - label
                for j in range(3):
                    gradient[j] += error * features[j] / len(batch)
            lr = config['lr'] * (1 + math.cos(math.pi * state['updates'] / full_steps)) / 2
            for j in range(3):
                state['velocity'][j] = config['momentum'] * state['velocity'][j] + gradient[j]
                state['weights'][j] -= lr * state['velocity'][j]
            state['updates'] += 1
        value = loss(state['weights'], validation)
        state['history'].append({'resource': epoch, 'metric': value})
        if state['best_metric'] is None or value < state['best_metric']:
            state['best_metric'], state['best_resource'] = value, epoch
            state['best_weights'] = list(state['weights'])
        state['resource'] = epoch
    state['rng'] = rng.getstate()
    return state


def make_plan(train, validation):
    context = fingerprint({'train': train, 'validation': validation, 'evaluator': 'binary-cross-entropy-v1'})
    shared = {'method': 'toy_classifier', 'seed': 17, 'comparison_context_sha256': context,
              'metric': {'name': 'validation_bce', 'direction': 'min'}}
    candidates = []
    for index, lr in enumerate([0.025, 0.001, 0.003, 0.009, 0.02, 0.06, 0.18, 0.5, 1.5, 4.5]):
        config = dict(shared, lr=lr, batch_size=16, momentum=0.9, max_resource=27,
                      schedule={'kind': 'cosine', 'full_epochs': 27, 'updates_per_epoch': 8})
        candidates.append({'config': config, 'roles': ['baseline'] if index == 0 else ['candidate'],
                           'prune_after_resource': 3})
    return dict(shared, schema_version=1, resource_unit='epochs', rungs=[3, 9, 27],
                keep_counts=[3, 1], ranking='best_so_far', tie_margin=0.0, candidates=candidates)


def run_demo(destination, compare_exhaustive=False):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    train, validation = data()
    plan = make_plan(train, validation)
    candidates = validate_plan(plan)
    history = {'plan_sha256': fingerprint(plan), 'observations': []}
    save(destination / 'plan.json', plan)
    save(destination / 'data.json', {'train': train, 'validation': validation})
    shutil.copyfile(Path(__file__).resolve().parents[1] / 'scripts/fast_search.py', destination / 'fast_search.py')
    states, trained_epochs, resumed = {}, 0, 0
    while True:
        decision = decide(plan, history, destination)
        save(destination / f'decision-{len(history["observations"])}.json', decision)
        if decision['status'] != 'needs_work':
            break
        for action in decision['actions']:
            identity = action['trial_id']
            config = candidates[identity]['config']
            if action['resume_checkpoint']:
                state = read_json(destination / action['resume_checkpoint']['path'])
                resumed += 1
            else:
                state = initial(config)
            if state['resource'] != action['from_resource']:
                raise AssertionError('Replayed or skipped training prefix')
            trained_epochs += action['target_resource'] - state['resource']
            state = train_to(config, state, action['target_resource'], train, validation)
            states[identity] = state
            folder = destination / identity
            checkpoint = folder / f'resume-{state["resource"]}.json'
            best = folder / f'best-{state["best_resource"]}.json'
            metrics = folder / f'metrics-{state["resource"]}.json'
            save(checkpoint, state)
            save(best, {'trial_id': identity, 'resource': state['best_resource'],
                        'weights': state['best_weights'], 'metric': state['best_metric']})
            save(metrics, state['history'])
            history['observations'].append({
                'trial_id': identity, 'resource': state['resource'], 'status': 'ok',
                'metric': state['history'][-1]['metric'], 'best_metric': state['best_metric'],
                'best_resource': state['best_resource'], 'resume_checkpoint': reference(checkpoint, destination),
                'best_checkpoint': reference(best, destination), 'metrics': reference(metrics, destination)})
        save(destination / 'history.json', history)
    if decision['status'] != 'complete':
        raise AssertionError(decision['status'])
    winner = decision['winner']
    reloaded = read_json(destination / winner['best_checkpoint']['path'])
    reload_metric = loss(reloaded['weights'], validation)
    if reload_metric != winner['best_metric']:
        raise AssertionError('Exported checkpoint metric mismatch')
    fast_seconds = time.monotonic() - started
    report = {'kind': 'tiny_generated_data_cpu_training', 'actual_gpu_training': False,
              'pretrained_full_ft_or_lora': False, 'candidates': len(candidates),
              'rungs': plan['rungs'], 'candidate_epochs_trained': trained_epochs,
              'full_reference_candidate_epochs': len(candidates) * plan['rungs'][-1],
              'resumed_segments': resumed, 'fast_seconds_including_io': fast_seconds,
              'pruned_candidates': len(decision['pruned']), 'finalists': len(decision['final_ranking']),
              'winner': winner, 'checkpoint_reload_metric': reload_metric,
              'exposure_reduction_fraction': 1 - trained_epochs / (len(candidates) * plan['rungs'][-1]),
              'fresh_seed_confirmation_complete': False}
    if compare_exhaustive:
        reference_started = time.monotonic()
        full = {identity: train_to(row['config'], initial(row['config']), plan['rungs'][-1], train, validation)
                for identity, row in candidates.items()}
        for identity in decision['final_ranking']:
            if fingerprint(states[identity]) != fingerprint(full[identity]):
                raise AssertionError('Resumed final state differs from uninterrupted training')
        reference_winner = min(full, key=lambda identity: (full[identity]['best_metric'], identity))
        report['reference_check'] = {'kind': 'separate_exhaustive_validation_overhead',
            'seconds_excluding_io': time.monotonic() - reference_started,
            'all_finalist_states_equal_uninterrupted': True,
            'best_metric': full[reference_winner]['best_metric'], 'trial_id': reference_winner,
            'same_winning_config': reference_winner == winner['trial_id'],
            'metric_regret': winner['best_metric'] - full[reference_winner]['best_metric']}
    save(destination / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compare-exhaustive', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run_demo(args.output, args.compare_exhaustive), indent=2))


if __name__ == '__main__':
    main()
