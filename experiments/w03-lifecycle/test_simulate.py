import unittest
from simulate import simulate


class LifecycleTests(unittest.TestCase):
    def test_baseline_hand_calculated_schedule(self):
        r = simulate()
        self.assertEqual([s['plans'] for s in r['trace']],
                         [{'A': 4}, {'A': 1, 'B': 2}, {'B': 1, 'C': 3}, {'B': 1, 'C': 3}])
        self.assertEqual([s['outputs'] for s in r['trace']], [['A'], ['A','B'], ['B'], ['B','C']])
        self.assertEqual([s['peak_blocks'] for s in r['trace']], [2,4,4,5])
        self.assertEqual([s['used_after'] for s in r['trace']], [2,1,4,0])

    def test_last_sampled_token_has_not_been_forwarded(self):
        for r in simulate()['requests']:
            self.assertEqual(r['computed'], r['prompt'] + r['target'] - 1)

    def test_partial_prefill_does_not_emit(self):
        r = simulate(budget=2)
        self.assertEqual(r['trace'][0]['outputs'], [])
        self.assertEqual(r['trace'][1]['outputs'], ['A'])

    def test_cancellation_removes_future_work_and_releases(self):
        r = simulate(cancel_b=True)
        self.assertEqual([s['plans'] for s in r['trace']],
                         [{'A':4}, {'A':1,'B':2}, {'C':4}, {'C':2}])
        b = r['requests'][1]
        self.assertEqual((b['state'], b['computed'], b['emitted'], b['blocks']), ('ABORTED',2,1,0))

    def test_blocked_is_not_silently_reported_complete(self):
        r = simulate(capacity=2)
        self.assertEqual(r['status'], 'BLOCKED_NO_PREEMPTION')
        self.assertTrue(any('allocation_denied' in e for e in r['trace'][-1]['events']))

    def test_serial_admission_keeps_order(self):
        r = simulate(max_running=1)
        self.assertEqual([s['plans'] for s in r['trace']],
                         [{'A':4},{'A':1},{'B':2},{'B':1},{'B':1},{'C':4},{'C':2}])

    def test_budget_capacity_and_accounting_sweep(self):
        for budget in range(1,9):
            for slots in range(1,4):
                for capacity in [2,4,8,16]:
                    for cancel in [False,True]:
                        result = simulate(budget=budget,max_running=slots,capacity=capacity,cancel_b=cancel)
                        for s in result['trace']:
                            self.assertLessEqual(sum(s['plans'].values()), budget)
                            self.assertLessEqual(s['peak_blocks'], capacity)
                            self.assertLessEqual(len(s['running_after']), slots)
                            for r in s['after']:
                                self.assertLessEqual(r['computed'], r['prompt']+r['emitted'])
                                self.assertLessEqual(r['emitted'],r['target'])
                                if r['state'] in ['ABORTED','FINISHED_LENGTH']:
                                    self.assertEqual(r['blocks'],0)
                        if result['status'] == 'COMPLETE':
                            self.assertEqual(result['trace'][-1]['used_after'],0)

    def test_bad_inputs(self):
        for value in [0,-1,1.5,True]:
            with self.assertRaises(ValueError): simulate(budget=value)


if __name__ == '__main__': unittest.main()
