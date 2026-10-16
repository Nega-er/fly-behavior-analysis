import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout
import io

import cv2
import numpy as np

from grooming import (PARAMETERS,aligned_patch,classify,run_prototype,read_rows,sha256,
                      validation_report,body_geometry,aggregate,movement_speed,REGIONS)


def fixture(folder,moving=False):
    folder=Path(folder);video=folder/'source.avi';tracks=folder/'tracks.csv'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),30,(240,180))
    if not writer.isOpened():raise RuntimeError('Synthetic video could not be created')
    rows=[]
    for k in range(60):
        center=(90+k if moving else 100,90)
        image=np.zeros((180,240,3),np.uint8);cv2.ellipse(image,center,(10,4),0,0,360,(220,220,220),-1)
        if k<30:
            x=center[0]+(7 if k%2 else 13)
            for y in (center[1]-8,center[1]+5):cv2.rectangle(image,(x,y),(x+4,y+3),(220,220,220),-1)
        writer.write(image)
        rows.append(dict(frame=k,time_s=k/30,slot=1,segment=1,x_px=center[0],y_px=center[1],status='observed',
                         axis_dx=-1 if k%2 else 1,axis_dy=0,axis_ratio=3,image_updated=1))
    writer.release()
    with tracks.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    session=folder/'session.json'
    session.write_text(json.dumps(dict(schema_version=1,video_sha256=sha256(video),tracks_sha256=sha256(tracks),
                                       labels={'1:0':'groom_body','1:1':'stopped'})),encoding='utf-8')
    return video,tracks,session


class GroomingTests(unittest.TestCase):
    def test_one_end_motion_is_not_diluted_by_the_other_body_half(self):
        features=dict(core_change=.02,periphery_change=.045,end_a_change=.09,end_b_change=0,
                      illumination_step=0,background_change=0)
        self.assertEqual(classify(features,0,0,PARAMETERS),('grooming','one_end_appendage_motion_candidate'))
        self.assertEqual(classify(dict(features,core_change=.2),0,0,PARAMETERS)[0],'unknown')
        self.assertEqual(classify(features,2,0,PARAMETERS)[0],'unknown')
        self.assertEqual(classify(features,0,40,PARAMETERS)[0],'unknown')

    def test_strokes_bridge_short_valid_pauses_but_not_rejected_observations(self):
        frames=[]
        for index in range(20):
            positive=index%2==0
            row=dict(slot=1,time_s=(index+1)*.05,dt_s=.05,valid=1,
                     candidate='grooming' if positive else 'non_grooming',reason='test')
            for name in REGIONS:
                row[name+'_change']=.1 if positive else 0.;row[name+'_energy']=.02
            frames.append(row)
        row=aggregate(frames,[1],1,PARAMETERS)[0]
        self.assertEqual(row['grooming_state'],'grooming')
        self.assertAlmostEqual(row['longest_candidate_s'],.5) # pauses add no evidence
        for index in range(1,20,2):frames[index]['valid']=0
        self.assertEqual(aggregate(frames,[1],1,PARAMETERS)[0]['grooming_state'],'unknown')
        for row in frames:row.update(valid=1,candidate='non_grooming')
        self.assertEqual(aggregate(frames,[1],1,PARAMETERS)[0]['grooming_state'],'non_grooming')

    def test_centroid_jitter_is_separated_from_sustained_walking(self):
        jitter=[(i*.04,np.array([100+(i%2),90.])) for i in range(5)]
        self.assertEqual(movement_speed(jitter,20),0)
        walking=[(i*.04,np.array([100+i,90.])) for i in range(5)]
        self.assertGreater(movement_speed(walking,20),PARAMETERS['max_speed_body_lengths_s'])

    def test_body_core_geometry_excludes_dim_reflection_and_supports_dark_flies(self):
        patch=np.zeros((72,72),np.uint8)
        cv2.line(patch,(0,36),(71,36),90,2);cv2.ellipse(patch,(36,36),(10,4),0,0,360,220,-1)
        geometry=body_geometry(patch,25)
        self.assertIsNotNone(geometry);self.assertLess(geometry[0],15)
        self.assertEqual(body_geometry(255-patch,25),geometry)

    def test_alignment_removes_translation_without_erasing_appendage_change(self):
        original=np.zeros((160,160),np.uint8);cv2.ellipse(original,(80,80),(14,5),0,0,360,200,-1)
        shifted=cv2.warpAffine(original,np.float32([[1,0,7],[0,1,-4]]),(160,160))
        a=aligned_patch(original,(80,80),(1,0),72);b=aligned_patch(shifted,(87,76),(1,0),72)
        np.testing.assert_array_equal(a,b)
        shifted[82:86,94:98]=220
        self.assertGreater(np.abs(a.astype(float)-aligned_patch(shifted,(87,76),(1,0),72)).sum(),0)

    def test_large_change_is_not_sufficient_for_grooming(self):
        features=dict(core_change=.2,periphery_change=.4,illumination_step=0,background_change=0)
        self.assertEqual(classify(features,0,0,PARAMETERS)[0],'unknown')
        features['core_change']=.01
        self.assertEqual(classify(features,0,0,PARAMETERS)[0],'grooming')
        for key,value in (('illumination_step',40),('background_change',.8)):
            trial=dict(features,**{key:value});self.assertEqual(classify(trial,0,0,PARAMETERS)[0],'unknown')
        self.assertEqual(classify(features,2,0,PARAMETERS)[0],'unknown')
        self.assertEqual(classify(features,0,40,PARAMETERS)[0],'unknown')

    def test_video_manual_comparison_and_immutable_example_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);video,tracks,session=fixture(root);out=root/'prototype';learning=root/'learning'
            originals={p:p.read_bytes() for p in (video,tracks,session)}
            run_prototype(video,tracks,out,session,learning=learning)
            rows=read_rows(out/'grooming_intervals.csv')
            self.assertEqual([r['grooming_state'] for r in rows],['grooming','non_grooming'])
            report=json.loads((out/'grooming_validation.json').read_text())
            self.assertEqual(report['labeled_intervals'],2);self.assertEqual(report['precision'],1)
            self.assertFalse(report['trained']);self.assertTrue(report['thresholds_fitted_to_labels'])
            self.assertFalse(report['evaluation_independent'])
            self.assertTrue((out/'grooming_report.png').is_file())
            cap=cv2.VideoCapture(str(out/'grooming_preview.mp4'))
            self.assertTrue(cap.read()[0]);cap.release()
            examples=list((learning/'grooming/examples').glob('*.json'));self.assertEqual(len(examples),2)
            provenance=json.loads((out/'grooming_metadata.json').read_text())
            raw=read_rows(out/'grooming_frames.csv')
            self.assertEqual(len(raw),60)
            for path,contents in originals.items():self.assertEqual(path.read_bytes(),contents)
            with self.assertRaises(ValueError):run_prototype(video,tracks,out,session)
            run_prototype(video,tracks,root/'repeat',session,learning=learning,preview=False)
            self.assertEqual(len(list((learning/'grooming/examples').glob('*.json'))),2)
            self.assertEqual({p:p.read_bytes() for p in originals},originals)

    def test_excluded_neighbor_is_still_masked_and_bad_axis_still_has_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);video,tracks,_=fixture(root)
            rows=read_rows(tracks)
            for row in rows:row['axis_ratio']=1.2
            with tracks.open('w',encoding='utf-8-sig',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            out=run_prototype(video,tracks,root/'bad_axis',preview=False)
            intervals=read_rows(out/'grooming_intervals.csv')
            self.assertTrue(all(r['grooming_state']=='unknown' for r in intervals))
            self.assertTrue(all(r['periphery_change']!='' for r in intervals))
            doubled=[]
            for row in rows:
                row['axis_ratio']=3;doubled.append(row)
                doubled.append(dict(row,slot=2,x_px=float(row['x_px'])+10))
            with tracks.open('w',encoding='utf-8-sig',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(doubled)
            out=run_prototype(video,tracks,root/'neighbor',excluded_slots=[2],preview=False)
            frames=read_rows(out/'grooming_frames.csv')
            self.assertEqual({r['slot'] for r in frames},{'1'})
            self.assertTrue(all(r['reason']=='neighbor_overlap' for r in frames))

    def test_wrong_tracking_ids_are_not_compared_by_video_alone(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);video,tracks,session=fixture(root)
            data=json.loads(session.read_text());data['tracks_sha256']='different-tracking'
            session.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'追跡'):
                run_prototype(video,tracks,root/'wrong',session)
            self.assertFalse((root/'wrong').exists())

    def test_tracking_repeat_flag_does_not_hide_local_appendage_motion(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);video,tracks,_=fixture(root);rows=read_rows(tracks)
            for row in rows:row['image_updated']=0
            with tracks.open('w',encoding='utf-8-sig',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            out=run_prototype(video,tracks,root/'local_motion',preview=False)
            self.assertEqual(read_rows(out/'grooming_intervals.csv')[0]['grooming_state'],'grooming')

    def test_walking_remains_unknown_and_unlabeled_frames_are_not_ground_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);video,tracks,_=fixture(root,moving=True)
            out=run_prototype(video,tracks,root/'moving',preview=False)
            self.assertNotIn('grooming',[r['grooming_state'] for r in read_rows(out/'grooming_intervals.csv')])
            report=json.loads((out/'grooming_validation.json').read_text())
            self.assertEqual(report['labeled_intervals'],0);self.assertIsNone(report['precision'])
            out=run_prototype(video,tracks,root/'short_speed_window',preview=False,parameters={'speed_window_s':.001})
            self.assertNotIn('grooming',[r['grooming_state'] for r in read_rows(out/'grooming_intervals.csv')])

    def test_unknown_predictions_count_as_misses_and_human_unknown_is_excluded(self):
        report=validation_report([dict(human_grooming='grooming',grooming_state='unknown'),
                                  dict(human_grooming='grooming',grooming_state='grooming'),
                                  dict(human_grooming='non_grooming',grooming_state='grooming'),
                                  dict(human_grooming='',grooming_state='unknown')])
        self.assertEqual(report['labeled_intervals'],3);self.assertEqual(report['recall'],.5)
        self.assertEqual(report['precision'],.5);self.assertEqual(report['unknown_predictions'],1)

    def test_automatic_pipeline_adds_grooming_as_a_separate_result(self):
        from analyze_flies import main,DEFAULTS
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);video,tracks,_=fixture(root)
            config=root/'config.json';config.write_text(json.dumps(dict(DEFAULTS,expected_flies=1,grooming_enabled=True)))
            out=root/'automatic'
            args=['analyze_flies.py','--tracks',str(tracks),'--flies','1','--config',str(config),'--direction-video',str(video),'--out',str(out),'--progress']
            capture=io.StringIO()
            with patch('sys.argv',args),patch('analyze_flies.active_model_path',return_value=None),redirect_stdout(capture):main()
            from progress_protocol import PREFIX
            progress=[json.loads(line[len(PREFIX):]) for line in capture.getvalue().splitlines() if line.startswith(PREFIX)]
            fractions=[row['current']/row['total'] for row in progress if row['stage']=='preview']
            self.assertEqual(fractions,sorted(fractions))
            activity=read_rows(out/'activity.csv')
            self.assertTrue(activity)
            self.assertTrue(all(r['locomotion_direction'] in ('forward','backward','stopped','lateral','unknown') for r in activity))
            self.assertTrue((out/'grooming/grooming_frames.csv').exists())


if __name__=='__main__':unittest.main()
