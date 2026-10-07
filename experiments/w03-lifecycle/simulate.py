"""Deterministic teaching trace. Does not import or benchmark vLLM."""
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import csv
import json
import math


@dataclass
class Request:
    name: str
    prompt: int
    target: int
    arrival: int
    computed: int = 0
    emitted: int = 0
    state: str = 'NOT_ARRIVED'
    blocks: int = 0
    first_output: int | None = None
    finish: int | None = None


def simulate(budget=4, max_running=2, block_size=2, capacity=8, cancel_b=False):
    for value in [budget, max_running, block_size, capacity]:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError('Budgets, sizes and capacity must be positive integers')
    requests = [Request('A', 4, 2, 0), Request('B', 2, 3, 0), Request('C', 6, 1, 1)]
    by_name = {r.name: r for r in requests}
    running, waiting, trace = [], [], []
    status = 'COMPLETE'
    for step in range(1000):
        events = []
        for r in requests:
            if r.arrival == step:
                r.state = 'WAITING'
                waiting.append(r.name)
                events.append(f'{r.name}:arrive')
        if cancel_b and step == 2:
            r = by_name['B']
            if r.state in ['WAITING', 'RUNNING']:
                if r.name in running: running.remove(r.name)
                if r.name in waiting: waiting.remove(r.name)
                r.state, r.blocks, r.finish = 'ABORTED', 0, step
                events.append('B:abort')
        before = [asdict(r) for r in requests]
        plans, remaining = {}, budget

        def reserve(r, n):
            required = math.ceil((r.computed + n) / block_size)
            used_elsewhere = sum(q.blocks for q in requests if q is not r)
            if used_elsewhere + required > capacity:
                events.append(f'{r.name}:allocation_denied')
                return False
            r.blocks = required
            plans[r.name] = n
            return True

        # RUNNING first, then FCFS waiting; no priority, speculation, APC or preemption.
        for name in list(running):
            if not remaining: break
            r = by_name[name]
            n = min(r.prompt + r.emitted - r.computed, remaining)
            if n and reserve(r, n): remaining -= n
        while waiting and remaining and len(running) < max_running:
            r = by_name[waiting[0]]
            n = min(r.prompt + r.emitted - r.computed, remaining)
            if not reserve(r, n): break
            remaining -= n
            waiting.pop(0)
            running.append(r.name)
            r.state = 'RUNNING'

        peak_blocks = sum(r.blocks for r in requests)
        outputs = []
        for name, n in plans.items():
            r = by_name[name]
            known_before = r.prompt + r.emitted
            r.computed += n
            if r.computed == known_before:
                r.emitted += 1
                outputs.append(name)
                if r.first_output is None: r.first_output = step + 1
                if r.emitted == r.target:
                    r.state, r.finish, r.blocks = 'FINISHED_LENGTH', step + 1, 0
                    running.remove(name)
                    events.append(f'{name}:finish')
        trace.append({'step': step, 'plans': plans, 'outputs': outputs, 'before': before,
                      'after': [asdict(r) for r in requests], 'running_after': list(running),
                      'waiting_after': list(waiting), 'peak_blocks': peak_blocks,
                      'used_after': sum(r.blocks for r in requests), 'events': events})
        if all(r.state in ['FINISHED_LENGTH', 'ABORTED'] for r in requests): break
        if not plans and (running or waiting):
            status = 'BLOCKED_NO_PREEMPTION'
            break
    else:
        raise RuntimeError('Simulator did not terminate within 1000 steps')
    return {'kind': 'teaching_simulation_not_vllm_measurement', 'status': status,
            'config': {'budget': budget, 'max_running': max_running, 'block_size': block_size,
                       'capacity': capacity, 'cancel_b': cancel_b},
            'trace': trace, 'requests': [asdict(r) for r in requests]}


SCENARIOS = {
    'baseline': {}, 'budget2': {'budget': 2}, 'serial': {'max_running': 1},
    'wide': {'budget': 8, 'max_running': 3}, 'cancel': {'cancel_b': True},
    'blocked': {'capacity': 2},
}


def generate(output):
    output.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, options in SCENARIOS.items():
        result = simulate(**options)
        (output/f'{name}.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
        summary.append({'scenario': name, 'status': result['status'], 'steps': len(result['trace']),
                        'input_positions': sum(sum(s['plans'].values()) for s in result['trace']),
                        'output_tokens': sum(r['emitted'] for r in result['requests']),
                        'peak_blocks': max(s['peak_blocks'] for s in result['trace'])})
    with (output/'summary.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]), lineterminator='\n'); w.writeheader(); w.writerows(summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path(__file__).parent/'data')
    args = p.parse_args()
    generate(args.output)
