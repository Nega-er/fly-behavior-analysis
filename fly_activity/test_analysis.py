import tempfile
import unittest
from pathlib import Path

import numpy as np
from analyze_flies import DEFAULTS, Tracker, analyze, classify_intervals, window_speed


class AnalysisTests(unittest.TestCase):
    def rows(self, coords, segments=None):
        return [dict(frame=k, time_s=k/10, slot=1,
                     segment=(segments[k] if segments else 1) if p is not None else 0,
                     x_px=p[0] if p is not None else "", y_px=p[1] if p is not None else "",
                     status="observed" if p is not None else "unknown") for k, p in enumerate(coords)]

    def run_analysis(self, rows, **overrides):
        c = dict(DEFAULTS, expected_flies=1, smooth_window_s=0,
                 confirmation_s=.2, move_start_speed=5, stop_start_speed=2)
        c.update(overrides)
        with tempfile.TemporaryDirectory() as directory:
            return analyze(rows, c, Path(directory))

    def test_known_straight_distance_and_mm(self):
        rows = self.rows([(k, 0) for k in range(11)])
        result = self.run_analysis(rows)
        self.assertAlmostEqual(result["measured_distance"], 10.)
        self.assertAlmostEqual(result["valid_fly_seconds"], 1.)
        result = self.run_analysis(rows, arena_radius_px=100, arena_diameter_mm=100)
        self.assertAlmostEqual(result["measured_distance"], 5.)
        self.assertEqual(result["unit"], "mm")

    def test_missing_and_new_segment_never_bridge(self):
        rows = self.rows([(0, 0), (1, 0), None, (100, 0), (101, 0), (200, 0), (201, 0)],
                         [1, 1, 0, 2, 2, 3, 3])
        result = self.run_analysis(rows)
        self.assertAlmostEqual(result["measured_distance"], 3.)
        self.assertAlmostEqual(result["valid_fly_seconds"], .3)

    def test_stationary_noise_reduced_without_inventing_movement(self):
        rows = self.rows([(.2 if k % 5 == 2 else 0, 0) for k in range(31)])
        result = self.run_analysis(rows, smooth_window_s=.5)
        self.assertAlmostEqual(result["measured_distance"], 0.)
        c = dict(DEFAULTS, confirmation_s=.2, move_start_speed=5, stop_start_speed=2)
        states = classify_intervals(np.r_[np.nan, np.ones(30)], np.r_[0., np.ones(30)/10], c)
        self.assertTrue(np.all(states[1:] == "stopped"))

    def test_collision_unknown_and_recovery_new_segments(self):
        c = dict(DEFAULTS, expected_flies=2, collision_distance_px=5, max_speed_px_s=100)
        t = Tracker(c)
        first, _, _ = t.update([(0, 0), (20, 0)], 0, 10)
        self.assertEqual([r["segment"] for r in first], [1, 1])
        collision, close, _ = t.update([(9, 0), (11, 0)], 1, 10)
        self.assertEqual(close, 2)
        self.assertTrue(all(r["status"] == "unknown" for r in collision))
        recovered, _, _ = t.update([(0, 0), (20, 0)], 2, 10)
        self.assertEqual([r["segment"] for r in recovered], [2, 2])

    def test_short_burst_and_missing_reset(self):
        c = dict(DEFAULTS, confirmation_s=.2, move_start_speed=5, stop_start_speed=2)
        states = classify_intervals(np.array([np.nan, 0, 0, 10, 0, np.nan, 0]),
                                    np.r_[0., np.ones(6)/10], c)
        self.assertEqual(states[3], "stopped")
        self.assertEqual(states[-1], "unknown")

    def test_repeated_video_frames_preserve_movement(self):
        times = np.arange(31)/30
        distance = np.r_[np.nan, [3. if k % 3 == 0 else 0. for k in range(1, 31)]]
        speed = window_speed(distance, times, .2)
        c = dict(DEFAULTS, move_start_speed=12, stop_start_speed=6)
        states = classify_intervals(speed, np.r_[0., np.diff(times)], c)
        self.assertTrue(np.all(states[9:] == "moving"))
        self.assertAlmostEqual(speed[-1], 30.)

    def test_duplicate_csv_rows_rejected(self):
        rows = self.rows([(0, 0), (1, 0)])
        with self.assertRaises(ValueError):
            self.run_analysis(rows + [rows[0]])

    def test_tracking_uses_image_update_interval(self):
        c = dict(DEFAULTS, expected_flies=1, max_speed_px_s=600)
        tracker = Tracker(c)
        for frame, x in enumerate([0,0,0,50,50,50,100]):
            updated = frame % 3 == 0
            dt = .1 if updated and frame else 1/30
            rows, _, _ = tracker.update([(x,0)], frame,30,dt,updated)
            self.assertEqual(rows[0]["status"], "observed")
            self.assertEqual(rows[0]["segment"],1)

    def test_distance_gate_preserves_repeated_frame_motion(self):
        rows = self.rows([(x,0) for x in [0,0,0,50,50,50,100]])
        for row in rows:
            k = row["frame"]
            row["time_s"] = k/30
            row["measurement_dt_s"] = .1 if k % 3 == 0 else 1/30
        result = self.run_analysis(rows,max_speed_px_s=600)
        self.assertAlmostEqual(result["measured_distance"],100.)
        self.assertAlmostEqual(result["interval_coverage"],1.)


if __name__ == "__main__":
    unittest.main()
