import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import permutation_test as scipy_permutation_test

from period_statistics import compute_period_statistics,holm_adjust,mean_difference_statistic,save_period_statistics


def activity(rates):
    rows=[]
    for slot,values in enumerate(rates,1):
        rows.append(dict(slot=slot,time_s=0,valid_interval=0,distance=""))
        for time,rate in zip((30,60,90),values):
            rows.append(dict(slot=slot,time_s=time,valid_interval=int(rate is not None),
                             distance=30*rate if rate is not None else ""))
    return rows


class PeriodStatisticsTests(unittest.TestCase):
    def test_old_and_new_scipy_keywords_preserve_randomized_results(self):
        def old_api(data,statistic,*,random_state=None,**kwargs):
            self.assertNotIn("rng",kwargs)
            return scipy_permutation_test(data,statistic,random_state=random_state,**kwargs)

        def new_api(data,statistic,*,rng=None,**kwargs):
            self.assertNotIn("random_state",kwargs)
            return scipy_permutation_test(data,statistic,random_state=rng,**kwargs)

        rows=activity([[i,i+2,i+5] for i in range(1,6)])
        with patch("period_statistics.permutation_test",old_api):
            old=compute_period_statistics(rows,90,resamples=71)
        with patch("period_statistics.permutation_test",new_api):
            new=compute_period_statistics(rows,90,resamples=71)
        self.assertFalse(old["omnibus_exact"])
        self.assertEqual(old,new)

    def test_exact_boundary_split_and_distance_conservation(self):
        times=(0,25,35,55,65,90)
        rows=[dict(slot=1,time_s=t,valid_interval=int(k>0),distance=2*(t-times[k-1]) if k else "") for k,t in enumerate(times)]
        r=compute_period_statistics(rows,90)
        self.assertEqual(r["status"],"insufficient")
        periods=r["individuals"][0]["periods"]
        np.testing.assert_allclose([p["distance"] for p in periods],[60,60,60])
        np.testing.assert_allclose([p["valid_s"] for p in periods],[30,30,30])

    def test_missing_period_excluded_and_not_zero(self):
        r=compute_period_statistics(activity([[1,2,3],[2,3,4],[1,None,3]]),90,resamples=99)
        self.assertEqual(r["included_slots"],[1,2])
        self.assertEqual(r["excluded_slots"],[3])
        self.assertIsNone(r["individuals"][2]["periods"][1]["mean_distance_per_s"])
        self.assertEqual(r["periods"][1]["mean"],2.5)

    def test_small_sample_omnibus_matches_exhaustive_enumeration(self):
        values=np.array([[1,2,7],[5,4,9]],float)
        r=compute_period_statistics(activity(values),90,resamples=99)
        observed=mean_difference_statistic(*values.T)
        null=[]
        for first,second in itertools.product(itertools.permutations(values[0]),itertools.permutations(values[1])):
            null.append(mean_difference_statistic(*np.array([first,second]).T))
        self.assertAlmostEqual(r["omnibus_p"],np.mean(np.array(null)>=observed-1e-12))
        self.assertTrue(r["omnibus_exact"])

    def test_constant_within_individual_is_not_significant(self):
        r=compute_period_statistics(activity([[1,1,1],[2,2,2],[5,5,5]]),90,resamples=999)
        self.assertEqual(r["omnibus_p"],1)
        self.assertTrue(all(p["p_holm"]==1 for p in r["pairs"]))

    def test_clear_paired_difference_and_holm(self):
        r=compute_period_statistics(activity([[i,i+10,i+20] for i in range(1,9)]),90,resamples=999)
        self.assertTrue(r["omnibus_significant"])
        self.assertTrue(all(p["significant"] for p in r["pairs"]))
        self.assertTrue(all(p["exact"] for p in r["pairs"]))
        np.testing.assert_allclose(holm_adjust([.03,.001,.2]),[.06,.003,.2])

    def test_short_video_and_legacy_data_are_explicit(self):
        r=compute_period_statistics(activity([[1,2,3],[2,3,4]])[:3],60)
        self.assertEqual(r["status"],"insufficient")
        self.assertIn("60秒以降",r["reason"])
        self.assertEqual(compute_period_statistics([dict(slot=1,time_s=0)],90)["status"],"unavailable")

    def test_saved_summary_and_csv(self):
        result=compute_period_statistics(activity([[1,2,3],[2,3,4]]),90,resamples=99)
        with tempfile.TemporaryDirectory() as directory:
            save_period_statistics(directory,result)
            folder=Path(directory)
            self.assertEqual(json.loads((folder/"period_statistics.json").read_text(encoding="utf-8"))["n"],2)
            self.assertEqual(len((folder/"period_individuals.csv").read_text(encoding="utf-8-sig").splitlines()),7)


if __name__=="__main__":unittest.main()
