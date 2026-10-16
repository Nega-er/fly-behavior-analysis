import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from evaluate import save_csv,read_csv
from manual_behavior import ManualSession,grooming_period_figure


def fixture(folder):
    video=folder/'video.avi';tracks=folder/'tracks.csv'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),10,(240,160))
    if not writer.isOpened():raise RuntimeError('Cannot create test video')
    rows=[]
    for frame in range(25):
        image=np.full((160,240,3),210,np.uint8)
        for slot,y in ((1,50),(2,110)):
            x=30+frame*4
            cv2.ellipse(image,(x,y),(10,5),0,0,360,(30,30,30),-1)
            missing=slot==2 and 10<=frame<20
            rows.append(dict(frame=frame,time_s=frame/10,slot=slot,segment=0 if missing else 1,
                             x_px='' if missing else x,y_px='' if missing else y,status='unknown' if missing else 'observed'))
        writer.write(image)
    writer.release();save_csv(tracks,rows,list(rows[0]))
    return ManualSession.create(video,tracks,folder/'annotations')


class ManualBehaviorTests(unittest.TestCase):
    def test_grooming_boundaries_partial_end_and_group_totals(self):
        with tempfile.TemporaryDirectory() as directory:
            session=fixture(Path(directory))
            # Aggregation uses annotation times, independently of video decoding.
            session.duration=60.5;session.seconds=61
            session.labels.update({'1:29':'groom_head','1:30':'groom_body',
                                   '1:59':'groom_head','1:60':'groom_body',
                                   '2:0':'groom_body','2:30':'unknown'})
            rows=session.grooming_periods()
            first=[row for row in rows if row['slot']==1]
            self.assertEqual([r['grooming_total_s'] for r in first],[1,2,.5])
            self.assertEqual([r['groom_head_s'] for r in first],[1,1,0])
            self.assertEqual([r['available_duration_s'] for r in first],[30,30,.5])
            group=[row for row in rows if row['slot']=='all']
            self.assertEqual([r['grooming_total_s'] for r in group],[2,2,.5])
            self.assertEqual(group[1]['unknown_s'],1)
            self.assertEqual(group[1]['unlabeled_s'],57)
            for row in rows:
                self.assertAlmostEqual(row['assessed_s']+row['unknown_s']+row['unlabeled_s'],row['available_duration_s'])
            self.assertEqual(sum(r['grooming_total_s'] for r in first),3.5)
            figure=grooming_period_figure(session,1)
            self.assertEqual([p.get_height() for p in figure.axes[0].patches],[1,1,0,0,1,.5])

    def test_short_video_grooming_export_marks_absent_periods(self):
        with tempfile.TemporaryDirectory() as directory:
            session=fixture(Path(directory));session.set_label(1,2,'groom_body')
            folder=session.export();rows=read_csv(folder/'manual_grooming_periods.csv')
            self.assertEqual(len(rows),9)
            own=[r for r in rows if r['slot']=='1']
            self.assertEqual(float(own[0]['grooming_total_s']),.5)
            self.assertEqual([float(r['available_duration_s']) for r in own],[2.5,0,0])
            self.assertGreater((folder/'manual_grooming_periods.png').stat().st_size,1000)
            figure=grooming_period_figure(session)
            self.assertEqual([text.get_text() for text in figure.axes[0].texts],['0.5','対象外','対象外'])

    def test_flying_survives_resume_csv_and_duration_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            session=fixture(Path(directory));session.set_label(1,2,'flying')
            loaded=ManualSession.load(session.path)
            self.assertEqual(loaded.label(1,2),'flying')
            folder=loaded.export()
            row=next(row for row in read_csv(folder/'manual_behavior.csv') if row['slot']=='1' and row['start_s']=='2')
            self.assertEqual(row['behavior'],'flying')
            self.assertEqual(row['behavior_label'],'飛んでいる')
            self.assertEqual(float(row['duration_s']),.5)
            summary=next(row for row in read_csv(folder/'manual_behavior_summary.csv') if row['slot']=='1' and row['behavior']=='flying')
            self.assertEqual(float(summary['duration_s']),.5)
            self.assertGreater((folder/'manual_behavior.png').stat().st_size,1000)

    def test_all_behaviors_resume_and_partial_second_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);session=fixture(folder)
            self.assertEqual(session.seconds,3)
            for slot,labels in ((1,['groom_head','groom_body','stopped']),(2,['forward','unknown','backward'])):
                for second,label in enumerate(labels):session.set_label(slot,second,label)
            loaded=ManualSession.load(session.path)
            self.assertEqual(loaded.labels,session.labels)
            self.assertEqual(len(loaded.data['history']),6)
            self.assertEqual(loaded.interval(2,1)['observed_frame_fraction'],0)
            self.assertEqual(loaded.label(2,1),'unknown')
            exported=loaded.export();rows=read_csv(exported/'manual_behavior.csv')
            self.assertEqual(len(rows),6)
            self.assertEqual(float(rows[-1]['duration_s']),.5)
            for slot in (1,2):
                self.assertAlmostEqual(sum(r['duration_s'] for r in loaded.summary() if r['slot']==slot),2.5)
            self.assertGreater((exported/'manual_behavior.png').stat().st_size,1000)
            self.assertTrue((exported/'manual_behavior_summary.csv').exists())

    def test_unlabeled_and_unknown_are_separate_and_correction_preserves_history(self):
        with tempfile.TemporaryDirectory() as directory:
            session=fixture(Path(directory));session.set_label(1,0,'unknown')
            self.assertEqual(session.label(1,1),'')
            self.assertEqual(session.interval(1,0)['annotated'],1)
            self.assertEqual(session.interval(1,1)['annotated'],0)
            session.set_label(1,0,'forward');session.set_label(1,0,'')
            self.assertEqual([r['behavior'] for r in session.data['history']],['unknown','forward',''])
            self.assertEqual(ManualSession.load(session.path).label(1,0),'')

    def test_failure_keeps_saved_annotations_and_source_files_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            session=fixture(Path(directory));session.set_label(1,0,'forward')
            previous=session.path.read_bytes();source=session.tracks_path.read_bytes()
            with patch('manual_behavior.atomic_json',side_effect=OSError('disk full')):
                with self.assertRaises(OSError):session.set_label(1,0,'backward')
            self.assertEqual(session.path.read_bytes(),previous)
            self.assertEqual(session.label(1,0),'forward')
            session.export()
            self.assertEqual(session.tracks_path.read_bytes(),source)
            with self.assertRaises(ValueError):session.set_label(1,3,'forward')
            with self.assertRaises(ValueError):session.set_label(99,0,'forward')
            with self.assertRaises(ValueError):session.set_label(1,0,'made_up')
            with self.assertRaises(ValueError):ManualSession.create(session.video,session.tracks_path,session.path.parent)

    def test_source_modification_is_detected_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            session=fixture(Path(directory))
            with session.tracks_path.open('a') as stream:stream.write('\n')
            with self.assertRaisesRegex(ValueError,'変更'):ManualSession.load(session.path)


if __name__=='__main__':unittest.main()
