"""Tiny stdlib-only synthetic tests: no torch, subprocess, training or network."""
import copy
import unittest

import compare_pair as pair


def receipt(arm, seconds, start):
    return dict(schema='l1-whole-step-observation-v1', arm=arm, label=arm + '-test',
        host='synthetic-host', train_py='/synthetic/exact/train.py',
        train_sha256=pair.SOURCE_SHA, warmup=20, measured_steps=80,
        measured_seconds=80*seconds, seconds_per_step=seconds, poll_seconds=.005,
        max_observed_poll_gap_seconds=.005, timing_uncertainty_seconds_per_step=.01/80,
        events=[dict(step=i, monotonic=start+i*seconds, observed_batch=1) for i in range(100)])


class PairTests(unittest.TestCase):
    def setUp(self):
        self.off = receipt('off', 1., 1000.)
        self.on = receipt('on', .96, 2000.)

    def test_four_percent_passes(self):
        result = pair.compare(self.off, self.on)
        self.assertEqual(result['timing_verdict'], 'PASS')
        self.assertAlmostEqual(result['step_time_reduction'], .04)
        self.assertGreater(result['conservative_step_time_reduction'], .03)

    def test_two_percent_fails(self):
        self.assertEqual(pair.compare(self.off, receipt('on', .98, 2000.))['timing_verdict'], 'FAIL')

    def test_three_percent_is_polling_uncertain(self):
        result = pair.compare(self.off, receipt('on', .97, 2000.))
        self.assertEqual(result['timing_verdict'], 'INCONCLUSIVE')
        self.assertLess(result['conservative_step_time_reduction'], .03)
        self.assertGreater(result['optimistic_step_time_reduction'], .03)

    def test_distributions_keep_all_intervals(self):
        result = pair.compare(self.off, self.on)
        for row in result['distributions_seconds']:
            self.assertEqual(row['n'], 80)
            self.assertEqual(len(row['block10_means']), 8)
            self.assertAlmostEqual(row['median'], row['mean'])
            self.assertAlmostEqual(row['population_stddev'], 0.)

    def test_slow_interval_not_trimmed(self):
        for event in self.on['events'][50:]:
            event['monotonic'] += 4.
        self.on['measured_seconds'] += 4.
        self.on['seconds_per_step'] += 4./80
        result = pair.compare(self.off, self.on)
        self.assertEqual(result['timing_verdict'], 'FAIL')
        self.assertGreater(result['distributions_seconds'][1]['max'], 4.)

    def test_reject_identity_or_window_mismatch(self):
        for key, value in [('schema','wrong'), ('arm','off'), ('host','other'),
                           ('label',self.off['label']), ('train_py','/other/train.py'),
                           ('train_sha256','0'*64), ('warmup',21), ('measured_steps',79)]:
            with self.subTest(key=key):
                row = copy.deepcopy(self.on); row[key] = value
                with self.assertRaises(ValueError): pair.compare(self.off, row)

    def test_reject_forged_or_invalid_numeric_summaries(self):
        for key, value in [('seconds_per_step',.5), ('measured_seconds',10.),
                           ('timing_uncertainty_seconds_per_step',0.),
                           ('max_observed_poll_gap_seconds',float('nan')),
                           ('poll_seconds',.01), ('seconds_per_step',float('inf'))]:
            with self.subTest(key=key):
                row = copy.deepcopy(self.on); row[key] = value
                with self.assertRaises(ValueError): pair.compare(self.off, row)

    def test_reject_missing_or_duplicate_events(self):
        for mode in ('missing','duplicate','coalesced','backward','nan'):
            with self.subTest(mode=mode):
                row = copy.deepcopy(self.on)
                if mode == 'missing': row['events'].pop()
                if mode == 'duplicate': row['events'][20]['step'] = 19
                if mode == 'coalesced': row['events'][20]['observed_batch'] = 2
                if mode == 'backward': row['events'][20]['monotonic'] = 1.
                if mode == 'nan': row['events'][20]['monotonic'] = float('nan')
                with self.assertRaises(ValueError): pair.compare(self.off, row)

    def test_reject_reversed_boundary_chronology(self):
        with self.assertRaises(ValueError):
            pair.compare(self.off, receipt('on', .96, 900.))

    def test_large_polling_bound_is_inconclusive(self):
        self.on['max_observed_poll_gap_seconds'] = 1.
        self.on['timing_uncertainty_seconds_per_step'] = 2./80
        self.assertEqual(pair.compare(self.off, self.on)['timing_verdict'], 'INCONCLUSIVE')


if __name__ == '__main__':
    unittest.main()
