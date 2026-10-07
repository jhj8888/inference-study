"""Transparent teaching models. No GPU operations or wall-clock benchmarks."""
import math
from statistics import mean


def positive(value, name, allow_zero=False):
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f'{name} must be {">= 0" if allow_zero else "> 0"}')


def integer(value, name, allow_zero=False):
    positive(value, name, allow_zero)
    if int(value) != value:
        raise ValueError(f'{name} must be an integer')


def kv_bytes(layers=32, requests=1, context=8192, kv_heads=8, head_dim=128, element_bytes=2):
    for name, value in locals().copy().items():
        integer(value, name, allow_zero=name == 'context')
    return 2 * layers * requests * context * kv_heads * head_dim * element_bytes


def transfer(payload_bytes, bandwidth_gbs=25, chunk_bytes=4 * 2**20, setup_us=40):
    """One path: ideal payload time and assumed, serialized chunk overhead."""
    integer(payload_bytes, 'payload_bytes', True)
    positive(bandwidth_gbs, 'bandwidth_gbs')
    integer(chunk_bytes, 'chunk_bytes')
    positive(setup_us, 'setup_us', True)
    chunks = int((payload_bytes + chunk_bytes - 1) // chunk_bytes)
    payload_ms = payload_bytes / (bandwidth_gbs * 1e9) * 1000
    setup_ms = chunks * setup_us / 1000
    return dict(payload_bytes=payload_bytes, payload_gib=payload_bytes / 2**30,
                chunks=chunks, payload_ms=payload_ms, setup_ms=setup_ms,
                serialized_ms=payload_ms + setup_ms)


def gemm(m, n=4096, k=4096, element_bytes=2, peak_tflops=200, bandwidth_gbs=1500):
    """Y[M,N]=X[M,K]W[K,N]; each input read once, output written once."""
    for name, value in [('m', m), ('n', n), ('k', k), ('element_bytes', element_bytes)]:
        integer(value, name)
    positive(peak_tflops, 'peak_tflops'); positive(bandwidth_gbs, 'bandwidth_gbs')
    flops = 2 * m * n * k
    traffic = element_bytes * (m*k + k*n + m*n)
    compute_ms = flops / (peak_tflops * 1e12) * 1000
    memory_ms = traffic / (bandwidth_gbs * 1e9) * 1000
    return dict(m=m, flops=flops, traffic_bytes=traffic, intensity=flops/traffic,
                compute_ms=compute_ms, memory_ms=memory_ms,
                lower_ms=max(compute_ms, memory_ms),
                roof_tflops=min(peak_tflops, bandwidth_gbs * flops/traffic / 1000))


def decode_batch(batch, context=8192):
    """Hypothetical dense model: 7B active dense parameters, 14 GB weights.

    Each request emits one token per step. No queue, communication or preemption.
    Approximate KV read once per step; GQA reuse idealized. 1 ms assumed overhead.
    """
    integer(batch, 'batch'); integer(context, 'context')
    flops = 2 * 7e9 * batch + 4 * 32 * batch * 32 * context * 128
    read_bytes = 14e9 + kv_bytes(requests=batch, context=context)
    write_bytes = kv_bytes(requests=batch, context=1)
    step_ms = max(flops/200e12, (read_bytes+write_bytes)/1500e9)*1000 + 1
    return dict(batch=batch, context=context, step_ms=step_ms,
                output_tokens_s=batch*1000/step_ms,
                per_request_tokens_s=1000/step_ms,
                kv_gib=kv_bytes(requests=batch, context=context)/2**30)


def nearest_rank(values, percentile):
    if not 0 < percentile <= 100:
        raise ValueError('percentile must be in (0,100]')
    if not values:
        return None
    if any(not math.isfinite(x) for x in values):
        raise ValueError('values must be finite')
    return sorted(values)[math.ceil(len(values)*percentile/100)-1]


def request_metrics(request):
    """Synthetic client trace. Chunks count token-bearing stream outputs.

    E2E_token ends at last output; completion latency ends at HTTP done.
    TPOT uses E2E_token, matching the inspected vLLM completion endpoint.
    """
    scheduled, sent, done = (request[k] for k in ('scheduled_ms', 'sent_ms', 'done_ms'))
    for name, value in [('scheduled', scheduled), ('sent', sent), ('done', done)]:
        positive(value, name, True)
    if not scheduled <= sent <= done:
        raise ValueError('invalid request boundary timestamps')
    chunks = request['chunks']
    if not request['success']:
        return dict(id=request['id'], success=False, output_tokens=0,
                    client_queue_ms=sent-scheduled, completion_ms=done-sent)
    if not chunks:
        raise ValueError('successful request needs at least one output')
    previous = sent
    for chunk in chunks:
        t, count = chunk['at_ms'], chunk['tokens']
        positive(t, 'chunk time', True); integer(count, 'chunk tokens')
        if not previous <= t <= done:
            raise ValueError('non-monotonic or out-of-bound chunk')
        previous = t
    times = [c['at_ms'] for c in chunks]
    output_tokens = sum(c['tokens'] for c in chunks)
    gaps = [b-a for a,b in zip(times, times[1:])]
    ttft = times[0]-sent
    token_e2e = times[-1]-sent
    return dict(id=request['id'], success=True, output_tokens=output_tokens,
                client_queue_ms=sent-scheduled, ttft_ms=ttft,
                scheduled_ttft_ms=times[0]-scheduled, e2e_token_ms=token_e2e,
                completion_ms=done-sent, scheduled_completion_ms=done-scheduled,
                tpot_ms=(token_e2e-ttft)/(output_tokens-1) if output_tokens>1 else None,
                itl_ms=gaps, max_itl_ms=max(gaps) if gaps else None)


def summarize(requests, window_start_ms, window_end_ms):
    """A finite cohort, including its drain and failures, inside explicit window."""
    positive(window_end_ms-window_start_ms, 'window duration')
    if not requests or any(r['scheduled_ms'] < window_start_ms or r['done_ms'] > window_end_ms for r in requests):
        raise ValueError('every request must fit inside the cohort window')
    per_request = [request_metrics(r) for r in requests]
    ok = [r for r in per_request if r['success']]
    gaps = [g for r in ok for g in r['itl_ms']]
    tpots = [r['tpot_ms'] for r in ok if r['tpot_ms'] is not None]
    duration_s = (window_end_ms-window_start_ms)/1000
    passes_mean_slo = lambda r: r['ttft_ms'] <= 150 and (r['tpot_ms'] is None or r['tpot_ms'] <= 100)
    good = [r for r in ok if passes_mean_slo(r)]
    smooth = [r for r in good if r['max_itl_ms'] is None or r['max_itl_ms'] <= 80]
    return dict(evidence_type='constructed trace; no server benchmark',
                window_ms=[window_start_ms, window_end_ms], offered=len(requests),
                completed=len(ok), failed=len(requests)-len(ok),
                request_throughput=len(ok)/duration_s,
                output_throughput=sum(r['output_tokens'] for r in ok)/duration_s,
                total_token_throughput=(sum(r['prompt_tokens'] for r in requests if r['success'])+
                                        sum(r['output_tokens'] for r in ok))/duration_s,
                mean_request_tpot_ms=mean(tpots) if tpots else None,
                mean_pooled_itl_ms=mean(gaps) if gaps else None,
                p95_pooled_itl_ms=nearest_rank(gaps,95),
                mean_slo_goodput=len(good)/duration_s,
                gap_slo_goodput=len(smooth)/duration_s,
                goodput_policy='TTFT<=150ms AND TPOT<=100ms; extra gap SLO: max ITL<=80ms; single-token TPOT/ITL not applicable',
                per_request=per_request)


def fcfs(arrivals, services):
    """Single-server, non-preemptive, work-conserving FCFS recurrence."""
    if len(arrivals) != len(services):
        raise ValueError('length mismatch')
    rows, free, previous = [], 0.0, 0.0
    for i,(a,s) in enumerate(zip(arrivals, services)):
        positive(a, 'arrival', True); positive(s, 'service')
        if a < previous:
            raise ValueError('arrivals must be ordered')
        start = max(a, free); end = start+s
        rows.append(dict(id=i, arrival_s=a, service_s=s, start_s=start,
                         end_s=end, wait_s=start-a, response_s=end-a))
        previous, free = a, end
    return rows


def mm1_mean(arrival_rate, service_rate=200):
    positive(arrival_rate, 'arrival_rate', True); positive(service_rate, 'service_rate')
    return 1/(service_rate-arrival_rate) if arrival_rate < service_rate else None
