"""Checks of independent hand examples, boundary cases and trace semantics."""
import math
import unittest
from models import decode_batch, fcfs, gemm, kv_bytes, mm1_mean, nearest_rank, request_metrics, summarize, transfer
from run import trace


class PerformanceModels(unittest.TestCase):
    def test_one_gib(self):
        self.assertEqual(kv_bytes(),2**30)

    def test_kv_scaling(self):
        self.assertEqual(kv_bytes(requests=8),8*2**30)
        self.assertEqual(kv_bytes(kv_heads=32),4*2**30)
        self.assertEqual(kv_bytes(element_bytes=1),2**29)
        self.assertEqual(kv_bytes(context=0),0)

    def test_capacity_validation(self):
        for kwargs in [dict(requests=0),dict(context=-1),dict(layers=1.5),dict(head_dim=math.nan)]:
            with self.assertRaises(ValueError):kv_bytes(**kwargs)

    def test_transfer_units(self):
        r=transfer(2**30)
        self.assertEqual(r['chunks'],256)
        self.assertAlmostEqual(r['payload_ms'],42.94967296)
        self.assertAlmostEqual(r['serialized_ms'],53.18967296)

    def test_partial_chunk(self):
        r=transfer(2**20+1,chunk_bytes=2**20,setup_us=40)
        self.assertEqual(r['chunks'],2)
        self.assertAlmostEqual(r['setup_ms'],.08)
        self.assertEqual(transfer(0)['serialized_ms'],0)

    def test_transfer_invalid(self):
        for kwargs in [dict(bandwidth_gbs=0),dict(chunk_bytes=0),dict(setup_us=-1),dict(bandwidth_gbs=math.inf)]:
            with self.assertRaises(ValueError):transfer(1024,**kwargs)

    def test_tiny_gemm_by_hand(self):
        r=gemm(2,3,4,2,1,1)
        self.assertEqual(r['flops'],48)
        self.assertEqual(r['traffic_bytes'],52)
        self.assertAlmostEqual(r['intensity'],12/13)
        self.assertAlmostEqual(r['lower_ms'],52e-6)

    def test_compute_bound_gemm(self):
        r=gemm(2048)
        self.assertEqual(r['intensity'],1024)
        self.assertAlmostEqual(r['lower_ms'],.34359738368)

    def test_batch_tradeoff(self):
        a,b=decode_batch(1),decode_batch(32)
        self.assertGreater(b['output_tokens_s'],a['output_tokens_s'])
        self.assertGreater(b['step_ms'],a['step_ms'])
        self.assertLess(b['per_request_tokens_s'],a['per_request_tokens_s'])

    def test_stall_trace(self):
        r=request_metrics(trace()[0])
        self.assertEqual(r['client_queue_ms'],20)
        self.assertEqual(r['ttft_ms'],100)
        self.assertEqual(r['tpot_ms'],80)
        self.assertEqual(r['itl_ms'],[20,20,200])
        self.assertEqual(r['e2e_token_ms'],340)
        self.assertEqual(r['completion_ms'],360)

    def test_bundled_outputs(self):
        r=request_metrics(dict(id='bundle',scheduled_ms=0,sent_ms=0,done_ms=190,success=True,
            chunks=[dict(at_ms=80,tokens=1),dict(at_ms=110,tokens=2),dict(at_ms=170,tokens=3)]))
        self.assertEqual(r['itl_ms'],[30,60])
        self.assertEqual(r['output_tokens'],6)
        self.assertEqual(r['tpot_ms'],18)

    def test_single_token(self):
        r=request_metrics(trace()[3])
        self.assertIsNone(r['tpot_ms']);self.assertEqual(r['itl_ms'],[])

    def test_cohort_and_goodput(self):
        r=summarize(trace(),0,500)
        self.assertEqual((r['completed'],r['failed']),(4,1))
        self.assertEqual(r['request_throughput'],8)
        self.assertEqual(r['output_throughput'],24)
        self.assertEqual(r['total_token_throughput'],1048)
        self.assertEqual(r['mean_slo_goodput'],6)
        self.assertEqual(r['gap_slo_goodput'],4)

    def test_aggregation_weights(self):
        r=summarize(trace()[1:3],0,500)
        self.assertEqual(r['mean_request_tpot_ms'],105)
        self.assertEqual(r['mean_pooled_itl_ms'],48)

    def test_failed_and_invalid_trace(self):
        r=summarize([trace()[-1]],0,500)
        self.assertIsNone(r['mean_request_tpot_ms']);self.assertEqual(r['request_throughput'],0)
        bad=trace()[0];bad['chunks'][1]['at_ms']=119
        with self.assertRaises(ValueError):request_metrics(bad)
        with self.assertRaises(ValueError):summarize(trace(),0,400)

    def test_percentile_convention(self):
        self.assertEqual(nearest_rank([4,1,3,2],50),2)
        self.assertEqual(nearest_rank([4,1,3,2],99),4)
        self.assertIsNone(nearest_rank([],95))

    def test_fcfs_idle_and_wait(self):
        r=fcfs([0,.005,.1],[.01,.01,.01])
        for actual,expected in zip([x['end_s'] for x in r],[.01,.02,.11]):
            self.assertAlmostEqual(actual,expected)
        self.assertAlmostEqual(r[1]['wait_s'],.005)
        self.assertEqual(r[2]['wait_s'],0)

    def test_mm1_stability(self):
        self.assertAlmostEqual(mm1_mean(180),.05)
        self.assertIsNone(mm1_mean(200));self.assertIsNone(mm1_mean(220))
        with self.assertRaises(ValueError):fcfs([1,0],[1,1])
        with self.assertRaises(ValueError):fcfs([0],[math.nan])


if __name__=='__main__':unittest.main(verbosity=2)
