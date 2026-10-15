import csv
import json
from pathlib import Path
import tempfile
import unittest

from results_view import load_results,formatted


class ResultsTests(unittest.TestCase):
    def write_csv(self,path,rows):
        with path.open("w",encoding="utf-8-sig",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

    def test_group_direction_totals_and_unknown_time(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            (p/"summary_group.json").write_text(json.dumps(dict(expected_flies=2,duration_s=1,unit="mm")))
            self.write_csv(p/"direction_summary.csv",[
                dict(slot=1,forward_s=.6,backward_s=0,stopped_s=0,lateral_s=0,unknown_s=.4),
                dict(slot=2,forward_s=0,backward_s=.7,stopped_s=0,lateral_s=0,unknown_s=.3)])
            self.write_csv(p/"activity.csv",[
                dict(slot=1,time_s=0,locomotion_direction="unknown",direction_reason="tracking_missing"),
                dict(slot=1,time_s=.4,locomotion_direction="unknown",direction_reason="tracking_missing"),
                dict(slot=1,time_s=1,locomotion_direction="forward",direction_reason="body_relative_projection"),
                dict(slot=2,time_s=0,locomotion_direction="unknown",direction_reason="automatic_head_uncertain"),
                dict(slot=2,time_s=.3,locomotion_direction="unknown",direction_reason="automatic_head_uncertain"),
                dict(slot=2,time_s=1,locomotion_direction="backward",direction_reason="body_relative_projection")])
            before={f.name:f.read_bytes() for f in p.iterdir()}
            r=load_results(p)
            self.assertAlmostEqual(r["totals"]["forward"],.6)
            self.assertAlmostEqual(r["total_direction_fly_seconds"],2)
            self.assertAlmostEqual(r["unknown_fraction"],.35)
            self.assertAlmostEqual(sum(r["reasons"].values()),.7)
            self.assertEqual(before,{f.name:f.read_bytes() for f in p.iterdir()})

    def test_older_results_without_optional_files(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);(p/"summary_group.json").write_text('{"expected_flies":7,"duration_s":1}')
            r=load_results(p)
            self.assertIsNone(r["unknown_fraction"])
            self.assertEqual(r["evaluation"],{})
            self.assertEqual(r["previews"],{})

    def test_required_summary_missing_has_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError,"summary_group.json"):load_results(directory)

    def test_missing_metrics_not_shown_as_zero(self):
        self.assertEqual(formatted(None,percent=True),"—")
        self.assertEqual(formatted(0,percent=True),"0.0%")
        self.assertEqual(formatted(float("nan")),"—")
        self.assertEqual(formatted(.9186,percent=True),"91.9%")

    def test_legacy_time_bins_render_without_writing_results(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);(p/"summary_group.json").write_text('{"expected_flies":1,"duration_s":1}')
            self.write_csv(p/"time_bins.csv",[dict(start_s=0,end_s=1,slot=1,distance=5,valid_s=1,unit="px")])
            before={f.name:f.read_bytes() for f in p.iterdir()}
            result=load_results(p)
            self.assertIsNotNone(result["rate_image"])
            self.assertGreater(result["rate_image"].width,100)
            self.assertEqual(before,{f.name:f.read_bytes() for f in p.iterdir()})


if __name__=="__main__":unittest.main()
