import unittest
import tempfile
from pathlib import Path
import csv
import numpy as np

from analyze_flies import DEFAULTS, window_speed, analyze, classify_intervals
from direction import body_axis, classify_direction, resolve_headings, analyze_direction, remap_anchors
from annotate_heads import requests


class DirectionTests(unittest.TestCase):
    def rows(self, positions, segments=None):
        return [dict(frame=k, slot=1, segment=segments[k] if segments else 1,
                     x_px=float(p[0]), y_px=float(p[1]), status="observed",
                     axis_dx=(-1.)**k, axis_dy=0., axis_ratio=3.)
                for k, p in enumerate(positions)]

    def test_forward_backward_lateral_and_stopped(self):
        self.assertEqual(classify_direction(1, "moving", .5), "forward")
        self.assertEqual(classify_direction(-1, "moving", .5), "backward")
        self.assertEqual(classify_direction(0, "moving", .5), "lateral")
        self.assertEqual(classify_direction(np.nan, "stopped", .5), "stopped")
        self.assertEqual(classify_direction(1, "unknown", .5), "unknown")

    def test_no_head_guess_from_movement(self):
        rows = self.rows([(k,0) for k in range(10)])
        h, _ = resolve_headings(rows, {}, DEFAULTS)
        self.assertTrue(np.all(np.isnan(h)))

    def test_axis_sign_flip_is_not_a_head_flip(self):
        rows = self.rows([(k,0) for k in range(10)])
        h, _ = resolve_headings(rows, {(0,1): np.array([10.,0.])}, DEFAULTS)
        np.testing.assert_allclose(h, np.tile([1.,0.], (10,1)))

    def test_backward_detection_with_fixed_head(self):
        xy = np.array([(-k,0) for k in range(10)], float)
        rows = self.rows(xy)
        times = np.arange(10)/10
        valid = np.r_[False, np.ones(9, bool)]
        result = analyze_direction(rows, xy, times, valid, np.array(["moving"]*10),
                                   DEFAULTS, 1., {(0,1):np.array([10.,0.])}, window_speed)
        self.assertTrue(np.all(result[4][1:] == "backward"))
        self.assertTrue(np.all(result[2][1:] < 0))

    def test_gap_and_round_body_clear_polarity(self):
        rows = self.rows([(k,0) for k in range(5)], [1,1,2,2,2])
        h, _ = resolve_headings(rows, {(0,1):np.array([10.,0.])}, DEFAULTS)
        self.assertTrue(np.all(np.isnan(h[2:])))
        rows = self.rows([(k,0) for k in range(5)])
        rows[2]["axis_ratio"] = 1.
        h, _ = resolve_headings(rows, {(0,1):np.array([10.,0.])}, DEFAULTS)
        self.assertTrue(np.all(np.isnan(h[2:])))

    def test_rotation_discontinuity_clears_head(self):
        rows = self.rows([(k,0) for k in range(5)])
        rows[2].update(axis_dx=0., axis_dy=1.)
        h, sources = resolve_headings(rows, {(0,1):np.array([10.,0.])}, DEFAULTS)
        self.assertEqual(sources[2], "heading_discontinuity")
        self.assertTrue(np.all(np.isnan(h[2:])))

    def test_body_axis_rotation(self):
        points = np.array([(x,y) for x in range(-10,11) for y in [-1,0,1]], float)
        angle = .7
        rotation = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        axis, ratio = body_axis(points@rotation.T)
        self.assertGreater(ratio, 1.5)
        self.assertAlmostEqual(abs(np.dot(axis, rotation[:,0])), 1.)

    def test_confirmation_requests_after_axis_loss(self):
        rows = self.rows([(k,0) for k in range(6)])
        rows[2]["axis_ratio"] = 1.
        rows[4]["status"] = "unknown"
        todo = requests(rows)
        self.assertEqual([r["frame"] for r in todo], [0,3,5])

    def test_full_analysis_saves_backward_distance(self):
        rows = self.rows([(-k,0) for k in range(11)])
        for row in rows:
            row["time_s"] = row["frame"]/10
        config = dict(DEFAULTS, expected_flies=1, smooth_window_s=0,
                      move_start_speed=5, stop_start_speed=2)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            analyze(rows, config, out, {(0,1):np.array([10.,0.])})
            with (out/"direction_summary.csv").open(encoding="utf-8-sig") as f:
                summary = next(csv.DictReader(f))
            self.assertAlmostEqual(float(summary["backward_s"]), 1.)
            self.assertAlmostEqual(float(summary["backward_measured_distance"]), 10.)
            self.assertAlmostEqual(float(summary["forward_s"]), 0.)

    def test_slow_forward_not_blocked_by_original_stopped_state(self):
        xy = np.array([(.8*k,0) for k in range(11)], float)
        rows = self.rows(xy)
        times = np.arange(11)/10
        valid = np.r_[False, np.ones(10,bool)]
        result = analyze_direction(rows, xy, times, valid, np.array(["stopped"]*11),
                                   DEFAULTS,1.,{(0,1):np.array([10.,0.])}, window_speed, classify_intervals)
        self.assertTrue(np.all(result[4][1:] == "forward"))

    def test_direction_noise_floor_and_mm_invariance(self):
        xy = np.array([(.1*k,0) for k in range(11)], float)
        rows = self.rows(xy)
        times = np.arange(11)/10
        valid = np.r_[False, np.ones(10,bool)]
        for scale in [1.,.05]:
            result = analyze_direction(rows, xy, times, valid, np.array(["moving"]*11),
                                       DEFAULTS,scale,{(0,1):np.array([10.,0.])},window_speed,classify_intervals)
            self.assertTrue(np.all(result[4][1:] == "stopped"))

    def test_head_transfer_follows_position_not_display_slot(self):
        old = [dict(frame=0,slot=1,x_px=10,y_px=20,status="observed"),
               dict(frame=0,slot=2,x_px=100,y_px=20,status="observed")]
        new = [dict(frame=0,slot=2,x_px=10,y_px=20,status="observed"),
               dict(frame=0,slot=1,x_px=100,y_px=20,status="observed")]
        anchors,audit = remap_anchors({(0,1):np.array([15.,20.])},old,new)
        self.assertIn((0,2),anchors)
        self.assertEqual(audit[0]["status"],"transferred")

    def test_head_transfer_does_not_guess_missing_position(self):
        old = [dict(frame=0,slot=1,x_px=10,y_px=20,status="observed")]
        new = [dict(frame=0,slot=1,x_px=100,y_px=20,status="observed")]
        anchors,audit = remap_anchors({(0,1):np.array([15.,20.])},old,new)
        self.assertFalse(anchors)
        self.assertEqual(audit[0]["status"],"not_transferred")


if __name__ == "__main__":
    unittest.main()
