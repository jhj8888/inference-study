"""Run deterministic correctness experiments and produce the lesson's data."""
import copy
import csv
import hashlib
import json
import platform
from pathlib import Path
import torch
from attention import TinyDecoder

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
DATA.mkdir(exist_ok=True)
SEED = 20261007
TOLERANCE = 1e-10


def write_json(name, value):
    (DATA / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_csv(name, rows):
    with (DATA / name).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def max_error(a, b):
    return float((a - b).abs().max())


@torch.inference_mode()
def run():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    checks = []
    for hkv in (4, 2, 1):
        model = TinyDecoder(hkv).double().eval()
        for batch, length in ((1, 1), (2, 6), (1, 9)):
            ids = torch.randint(0, 41, (batch, length))
            reference, _ = model(ids)
            schedules = {'token': [1] * length, 'chunk': [min(3, length)] + [1] * max(0, length - 3)}
            if length == 9:
                schedules['chunk'] = [3, 2, 4]
            for name, sizes in schedules.items():
                outputs, cache, start = [], None, 0
                for size in sizes:
                    y, cache = model(ids[:, start:start + size], cache)
                    outputs.append(y)
                    start += size
                    for k, v in cache:
                        assert k.shape == v.shape == (batch, hkv, start, 8)
                incremental = torch.cat(outputs, dim=1)
                torch.testing.assert_close(incremental, reference, atol=TOLERANCE, rtol=TOLERANCE)
                checks.append({'case': f'hkv{hkv}_B{batch}_T{length}_{name}',
                               'type': 'positive', 'max_abs_error': max_error(incremental, reference),
                               'passed': True})
    # Prefix outputs must be unchanged when only future token IDs are changed.
    model = TinyDecoder(2).double().eval()
    ids = torch.randint(0, 41, (2, 9))
    reference, _ = model(ids)
    future_changed = ids.clone()
    future_changed[:, 5:] = (future_changed[:, 5:] + 1) % 41
    altered, _ = model(future_changed)
    torch.testing.assert_close(altered[:, :5], reference[:, :5], atol=TOLERANCE, rtol=TOLERANCE)
    checks.append({'case': 'future_token_invariance', 'type': 'positive',
                   'max_abs_error': max_error(altered[:, :5], reference[:, :5]), 'passed': True})
    _, prefix_cache = model(ids[:, :5])
    for name, options in [('wrong_mask', {'wrong_mask': True}), ('wrong_position', {'reset_positions': True})]:
        wrong, _ = model(ids[:, 5:], prefix_cache, **options)
        error = max_error(wrong, reference[:, 5:])
        assert error > 1e-5, f'Negative control did not detect {name}'
        checks.append({'case': name, 'type': 'negative_control', 'max_abs_error': error, 'passed': True})
    changed = copy.deepcopy(model)
    changed.blocks[0].attention.k.weight.add_(0.05 * torch.randn_like(changed.blocks[0].attention.k.weight))
    fresh, _ = changed(ids)
    stale, _ = changed(ids[:, 5:], prefix_cache)
    error = max_error(stale, fresh[:, 5:])
    assert error > 1e-5
    checks.append({'case': 'stale_weight_cache', 'type': 'negative_control', 'max_abs_error': error, 'passed': True})
    _, rebuilt = changed(ids[:, :5])
    repaired, _ = changed(ids[:, 5:], rebuilt)
    torch.testing.assert_close(repaired, fresh[:, 5:], atol=TOLERANCE, rtol=TOLERANCE)
    checks.append({'case': 'rebuild_after_weight_change', 'type': 'positive',
                   'max_abs_error': max_error(repaired, fresh[:, 5:]), 'passed': True})
    edited_ids = ids.clone()
    edited_ids[:, 0] = (edited_ids[:, 0] + 1) % 41
    edited_ref, _ = model(edited_ids)
    stale_prefix, _ = model(edited_ids[:, 5:], prefix_cache)
    error = max_error(stale_prefix, edited_ref[:, 5:])
    assert error > 1e-5
    checks.append({'case': 'edited_prefix_stale_cache', 'type': 'negative_control', 'max_abs_error': error, 'passed': True})
    # Generate eight tokens: one prefill + seven decode calls; do not feed token 8 back.
    prompt = ids[:, :4]
    full_tokens, cached_tokens, cache, input_step = prompt.clone(), prompt.clone(), None, prompt
    generation_errors = []
    for _ in range(8):
        full_logits, _ = model(full_tokens)
        cached_logits, cache = model(input_step, cache)
        generation_errors.append(max_error(full_logits[:, -1], cached_logits[:, -1]))
        torch.testing.assert_close(full_logits[:, -1], cached_logits[:, -1], atol=TOLERANCE, rtol=TOLERANCE)
        full_next = full_logits[:, -1].argmax(-1, keepdim=True)
        cached_next = cached_logits[:, -1].argmax(-1, keepdim=True)
        assert torch.equal(full_next, cached_next)
        full_tokens = torch.cat((full_tokens, full_next), dim=1)
        cached_tokens = torch.cat((cached_tokens, cached_next), dim=1)
        input_step = cached_next
    assert torch.equal(full_tokens, cached_tokens)
    assert cache[0][0].shape[2] == 4 + 8 - 1
    checks.append({'case': 'greedy_generation_8_tokens', 'type': 'positive',
                   'max_abs_error': max(generation_errors), 'passed': True})
    q = torch.tensor([[1., 0.], [0., 1.], [1., 1.]], dtype=torch.float64)
    k = q.clone()
    v = torch.tensor([[1., 0.], [0., 2.], [3., 1.]], dtype=torch.float64)
    scores = q @ k.T / (2 ** 0.5)
    allowed = torch.ones((3, 3), dtype=torch.bool).tril()
    weights = scores.masked_fill(~allowed, -torch.inf).softmax(-1)
    write_json('attention_example.json', {'Q': q.tolist(), 'K': k.tolist(), 'V': v.tolist(),
               'scores': scores.tolist(), 'weights': weights.tolist(), 'output': (weights @ v).tolist()})
    capacity = []
    for name, hkv in [('MHA', 32), ('GQA', 8), ('MQA', 1)]:
        for requests in (1, 4, 8, 16):
            for length in (1024, 2048, 4096, 8192, 16384, 32768):
                for item_bytes in (1, 2, 4):
                    size = 2 * 32 * requests * length * hkv * 128 * item_bytes
                    capacity.append(dict(mode=name, layers=32, query_heads=32, kv_heads=hkv,
                        head_dim=128, requests=requests, tokens=length, bytes_per_element=item_bytes,
                        kv_bytes=size, kv_gib=size / 2**30))
    write_csv('capacity.csv', capacity)
    work = []
    full_sum = cached_sum = 0
    for step in range(128):
        n = 128 + step
        full, cached = n * n, n * n if step == 0 else n
        full_sum += full
        cached_sum += cached
        work.append(dict(output_token=step + 1, context_tokens=n,
                         full_qkv_token_rows=n, cached_qkv_token_rows=n if step == 0 else 1,
                         full_dense_score_cells=full, cached_dense_score_cells=cached,
                         full_cumulative_cells=full_sum, cached_cumulative_cells=cached_sum))
    write_csv('work.csv', work)
    write_json('checks.json', {'evidence_type': 'CPU float64 correctness experiment; no latency benchmark',
        'seed': SEED, 'atol': TOLERANCE, 'rtol': TOLERANCE, 'checks': checks,
        'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                        'platform': platform.platform(), 'threads': 1, 'device': 'cpu'},
        'script_sha256': {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                          for name in ('attention.py', 'run.py')},
        'generation': {'prompt_tokens': 4, 'emitted_tokens': 8, 'final_cache_tokens': 11,
                       'token_ids': cached_tokens.tolist(), 'step_errors': generation_errors}})
    print(f'PASS: {len(checks)} checks; {sum(c["type"] == "negative_control" for c in checks)} expected failures detected.')
    print(f'Max positive error: {max(c["max_abs_error"] for c in checks if c["type"] == "positive"):.3e}')
    print('Wrote attention_example.json, checks.json, capacity.csv, work.csv.')


if __name__ == '__main__':
    run()
