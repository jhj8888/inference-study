"""CLI for logical KV capacity and an explicitly assumed transfer path."""
import argparse
import json
from models import kv_bytes, transfer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag,default in [('layers',32),('requests',1),('context',8192),('kv-heads',8),('head-dim',128),('element-bytes',2)]:
        p.add_argument('--'+flag, type=int, default=default)
    p.add_argument('--bandwidth-gbs',type=float,default=25,help='Decimal GB/s, not Gbit/s or GiB/s')
    p.add_argument('--chunk-mib',type=float,default=4)
    p.add_argument('--setup-us',type=float,default=40)
    a=p.parse_args()
    try:
        payload=kv_bytes(a.layers,a.requests,a.context,a.kv_heads,a.head_dim,a.element_bytes)
        result=transfer(payload,a.bandwidth_gbs,a.chunk_mib*2**20,a.setup_us)
    except ValueError as e:
        p.error(str(e))
    print(json.dumps(dict(evidence_type='theoretical payload + assumed serialized overhead; NOT measured bandwidth',
                         inputs=vars(a),**result),indent=2))


if __name__=='__main__':main()
