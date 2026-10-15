"""Verify the desktop progress protocol through a complete video analysis."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

from progress_protocol import PREFIX


class ProgressPipelineTests(unittest.TestCase):
    def test_long_multi_fly_video_completes_statistics_and_preview(self):
        here=Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);video=folder/"long.avi";out=folder/"result"
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*"MJPG"),1,(256,240))
            self.assertTrue(writer.isOpened())
            for frame in range(90):
                image=np.zeros((240,256,3),np.uint8)
                for y in (40,120,200):
                    cv2.ellipse(image,(50+4*(frame%30),y),(7,3),0,0,360,(255,255,255),-1)
                writer.write(image)
            writer.release()
            run=subprocess.run([sys.executable,str(here/"analyze_flies.py"),"--video",str(video),
                                "--flies","3","--head-model",str(here/"learning/seed_model.json"),
                                "--out",str(out),"--progress"],capture_output=True,text=True,encoding="utf-8")
            self.assertEqual(run.returncode,0,run.stderr)
            statistics=json.loads((out/"period_statistics.json").read_text(encoding="utf-8"))
            self.assertEqual(statistics["status"],"ok")
            self.assertEqual(statistics["n"],3)
            self.assertEqual(len(statistics["pairs"]),3)
            self.assertTrue((out/"metadata.json").exists())
            self.assertGreater((out/"direction_preview.mp4").stat().st_size,0)

    def test_video_pipeline_reports_work_and_preserves_results(self):
        here = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            video = folder / "test.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30, (120, 80))
            self.assertTrue(writer.isOpened())
            for frame in range(30):
                image = np.zeros((80, 120, 3), np.uint8)
                cv2.ellipse(image, (20 + frame*2, 40), (7, 3), 0, 0, 360, (255, 255, 255), -1)
                writer.write(image)
            writer.release()
            command = [sys.executable, str(here / "analyze_flies.py"), "--video", str(video),
                       "--flies", "1", "--head-model", str(here / "learning/seed_model.json")]
            run = subprocess.run(command + ["--out", str(folder / "progress"), "--progress"],
                                 capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(run.returncode, 0, run.stderr)
            events = [json.loads(line[len(PREFIX):]) for line in run.stdout.splitlines() if line.startswith(PREFIX)]
            ordered = list(dict.fromkeys(event["stage"] for event in events))
            self.assertEqual(ordered, ["prepare", "track", "analyze", "save", "preview"])
            for stage in ordered:
                reports = [event for event in events if event["stage"] == stage]
                self.assertEqual(reports[-1]["current"], reports[-1]["total"])
                self.assertTrue(all(0 <= event["current"] <= event["total"] for event in reports))
            for stage in ("prepare", "track", "preview"):
                self.assertTrue(any(0 < event["current"] < event["total"] for event in events if event["stage"] == stage))
            plain = subprocess.run(command + ["--out", str(folder / "plain")],
                                   capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(plain.returncode, 0, plain.stderr)
            self.assertNotIn(PREFIX, plain.stdout)
            for name in ("tracks.csv", "activity.csv", "summary_group.json", "direction_summary.csv"):
                self.assertEqual((folder / "progress" / name).read_bytes(), (folder / "plain" / name).read_bytes())


if __name__ == "__main__":
    unittest.main()
