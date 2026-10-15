import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from analyze_flies import analyze,DEFAULTS,main
from slot_selection import parse_excluded_slots


class SlotExclusionTests(unittest.TestCase):
    def test_result_action_reuses_tracks_and_fixed_model_without_retracking(self):
        from app import App
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            for name in ('tracks.csv','video.avi','model.json'):(folder/name).write_text('test')
            app=App.__new__(App);app.running=False;app.root=None
            app.results=SimpleNamespace(excluded_ids=SimpleNamespace(get=lambda:'2'),data=dict(
                folder=folder,config=dict(DEFAULTS,expected_flies=3),
                metadata=dict(source=str(folder/'video.avi'),head_model=str(folder/'model.json'))))
            with patch.object(app,'launch_analysis') as launch:
                app.exclude_ids()
            self.assertEqual(launch.call_args.args[1]['excluded_slots'],[2])
            self.assertEqual(launch.call_args.kwargs['tracks'],folder/'tracks.csv')
            self.assertEqual(launch.call_args.kwargs['head_model'],str(folder/'model.json'))
            app.results.excluded_ids.get=lambda:'1,2,3'
            with patch.object(app,'launch_analysis') as launch,patch('app.messagebox.showerror') as error:
                app.exclude_ids();launch.assert_not_called();error.assert_called_once()

    def test_explicit_ids_and_invalid_input(self):
        self.assertEqual(parse_excluded_slots('2, 5、2 ７'),[2,5,7])
        self.assertEqual(parse_excluded_slots(''),[])
        for text in ('0','-1','2.5','a','2,'):
            with self.assertRaises(ValueError):parse_excluded_slots(text)

    def test_totals_denominators_csv_and_plots_use_remaining_original_ids(self):
        rows=[dict(frame=k,time_s=k/10,slot=slot,segment=1,x_px=k*slot,y_px=0,status='observed')
              for slot in (1,2,3) for k in range(11)]
        c=dict(DEFAULTS,expected_flies=3,excluded_slots=[2],smooth_window_s=0)
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            with patch('analyze_flies.plot_report') as report,patch('analyze_flies.plot_direction') as direction:
                group=analyze(rows,c,out)
            self.assertEqual(group['included_slots'],[1,3]);self.assertEqual(group['excluded_slots'],[2])
            self.assertEqual(group['analyzed_flies'],2);self.assertEqual(group['possible_fly_seconds'],2)
            self.assertAlmostEqual(group['measured_distance'],40)
            self.assertAlmostEqual(group['interval_coverage'],1)
            for name in ('activity','summary_by_slot','direction_summary','segments','time_bins'):
                with (out/(name+'.csv')).open(encoding='utf-8-sig') as stream:
                    self.assertEqual({int(r['slot']) for r in csv.DictReader(stream)},{1,3})
            with (out/'movement_rate.csv').open(encoding='utf-8-sig') as stream:
                self.assertAlmostEqual(float(next(csv.DictReader(stream))['interval_coverage']),1)
            self.assertEqual([r['slot'] for r in report.call_args.args[3]],[1,3])
            self.assertEqual(direction.call_args.kwargs['slot_ids'],[1,3])
            self.assertEqual(direction.call_args.args[1].shape,(2,11))
            for excluded in ([1,2,3],[4],[0]):
                with self.assertRaises(ValueError):analyze(rows,dict(c,excluded_slots=excluded),out)

    def test_reaggregation_preserves_source_and_can_restore_excluded_ids(self):
        from test_manual_behavior import fixture
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);session=fixture(root)
            source=session.tracks_path;before=source.read_bytes()
            config=root/'config.json'
            config.write_text(json.dumps(dict(DEFAULTS,expected_flies=2,excluded_slots=[1])))
            out=root/'excluded'
            arguments=['analyze_flies.py','--tracks',str(source),'--config',str(config),
                       '--flies','2','--direction-video',str(session.video),'--out',str(out)]
            with patch('sys.argv',arguments),patch('analyze_flies.active_model_path',return_value=None):main()
            self.assertEqual(source.read_bytes(),before)
            self.assertEqual((out/'tracks.csv').read_bytes(),before)
            self.assertTrue((out/'direction_preview.mp4').is_file())
            saved=json.loads((out/'summary_group.json').read_text())
            self.assertEqual(saved['included_slots'],[2])
            self.assertEqual(saved['analyzed_flies'],1)
            with (out/'tracks.csv').open(encoding='utf-8-sig') as stream:
                tracks=list(csv.DictReader(stream))
            restored=root/'restored';restored.mkdir()
            group=analyze(tracks,dict(DEFAULTS,expected_flies=2),restored)
            self.assertEqual(group['included_slots'],[1,2])


if __name__=='__main__':unittest.main()
