"""Rebuild deterministic, explicitly synthetic teaching data with stdlib only."""
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import random
from statistics import mean
from models import decode_batch, fcfs, gemm, kv_bytes, mm1_mean, nearest_rank, summarize, transfer

HERE=Path(__file__).resolve().parent
DATA=HERE/'data'
SEEDS=[20261012,20261013,20261014,20261015,20261016]


def write_json(name,value):
    (DATA/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def write_csv(name,rows):
    with (DATA/name).open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def trace():
    specs=[('A',0,20,380,[120,140,160,360]),('B',0,0,320,[100,300]),
           ('C',50,50,210,[150,160,170,180,190]),('D',60,60,230,[210]),('E',80,80,500,[])]
    return [dict(id=i,scheduled_ms=s,sent_ms=t,done_ms=e,success=bool(times),prompt_tokens=128,
                 chunks=[dict(at_ms=at,tokens=1) for at in times]) for i,s,t,e,times in specs]


def main():
    DATA.mkdir(exist_ok=True)
    write_json('request_trace.json',dict(evidence_type='constructed timestamps; not measured',requests=trace()))
    write_json('metrics.json',summarize(trace(),0,500))
    write_csv('gemm.csv',[gemm(m) for m in [1,8,32,128,512,2048]])
    write_csv('batch.csv',[decode_batch(b,s) for s in [1024,8192,32768] for b in [1,2,4,8,16,32,64]])
    transfers=[]
    for s in [1024,4096,8192,16384,32768]:
        for b in [1,8]:
            for bw in [7,25,50]:
                for chunk in [0.25,4,16]:
                    transfers.append(dict(context=s,requests=b,bandwidth_gbs=bw,chunk_mib=chunk,
                                          **transfer(kv_bytes(requests=b,context=s),bw,int(chunk*2**20),40)))
    write_csv('transfer.csv',transfers)
    queue=[]
    for rho in [0.2,0.5,0.7,0.85,0.9,0.95]:
        for seed in SEEDS:
            rng=random.Random(seed);arrival=0;arrivals=[];services=[]
            for _ in range(12000):
                arrival+=rng.expovariate(200*rho)
                arrivals.append(arrival);services.append(rng.expovariate(200))
            rows=fcfs(arrivals,services);observed=rows[2000:]
            responses=[r['response_s']*1000 for r in observed]
            queue.append(dict(rho=rho,arrival_rate=200*rho,service_rate=200,seed=seed,warmup=2000,n=10000,
                              mean_ms=mean(responses),p50_ms=nearest_rank(responses,50),
                              p95_ms=nearest_rank(responses,95),p99_ms=nearest_rank(responses,99),
                              analytic_mean_ms=1000*mm1_mean(200*rho)))
            if rho==0.9 and seed==SEEDS[0]:write_csv('queue_sample.csv',observed[:200])
    write_csv('queue_summary.csv',queue)
    write_json('metadata.json',dict(evidence_type='formula calculations + constructed traces + FCFS simulation; no GPU/CPU latency measurements',
                python=platform.python_version(),seeds=SEEDS,quantile='nearest rank ceil(n*p/100)-1',
                queue=dict(model='M/M/1 FCFS, infinite buffer; exponential independent interarrivals/services',
                           service_rate=200,warmup_requests=2000,observed_requests_per_run=10000,runs=30,
                           sample='first 200 post-warmup requests at rho=0.9, first seed; not the full quantile sample'),
                assumed_hardware=dict(peak_tflops=200,hbm_gbs=1500,not_a_real_gpu_spec=True),
                code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'models.py',HERE/'run.py',HERE/'estimate.py']}))
    print('Generated GEMM, batching, transfer, streaming metrics and 30 seeded queue simulations. No hardware timed.')


if __name__=='__main__':main()
