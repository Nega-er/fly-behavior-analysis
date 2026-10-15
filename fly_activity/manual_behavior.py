"""Human behavior annotations; independent of automatic labels and learning data."""
import csv
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'fly_activity_mpl'))

import cv2

from cumulative_learning import atomic_json
from evaluate import read_csv,save_csv

BEHAVIORS={"groom_head":"グルーミング頭","groom_body":"グルーミング体","stopped":"停止",
           "forward":"前進","backward":"後退","flying":"飛んでいる","unknown":"Unknown","":"未入力"}
COLORS={"groom_head":"#e6a23c","groom_body":"#b47ac7","stopped":"#5095c5","forward":"#57a773",
        "backward":"#dc6666","flying":"#24a5a8","unknown":"#89929f","":"#e7e9ed"}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def video_info(video):
    cap=cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():raise ValueError('動画を開けません。')
        info=dict(fps=float(cap.get(5)),frames=int(cap.get(7)),width=int(cap.get(3)),height=int(cap.get(4)))
        if not math.isfinite(info['fps']) or info['fps']<=0 or info['frames']<1:raise ValueError('動画の時刻情報が不正です。')
        return info
    finally:cap.release()


class ManualSession:
    def __init__(self,path,data):
        self.path=Path(path).resolve();self.data=data
        self.video=Path(data['video']);self.tracks_path=Path(data['tracks'])
        self.fps=float(data['fps']);self.frames=int(data['frames']);self.duration=self.frames/self.fps
        self.seconds=math.ceil(self.duration-1e-9)
        self.tracks={};self.slots=sorted(map(int,data['slots']))
        for row in read_csv(self.tracks_path):
            key=int(row['frame']),int(row['slot'])
            if key in self.tracks:raise ValueError('追跡座標に重複したフレーム・個体があります。')
            self.tracks[key]=row
        self.labels=data.setdefault('labels',{})
        for key,value in self.labels.items():
            slot,second=map(int,key.split(':'));self.check(slot,second)
            if value not in BEHAVIORS:raise ValueError('未対応の行動ラベルがあります。')

    @classmethod
    def create(cls,video,tracks,folder):
        video=Path(video).resolve();tracks=Path(tracks).resolve();folder=Path(folder).resolve()
        if (folder/'session.json').exists():raise ValueError('既存の入力があります。「入力を再開」で開いてください。')
        info=video_info(video);rows=read_csv(tracks)
        if not rows:raise ValueError('追跡座標がありません。')
        frames=max(int(r['frame']) for r in rows)+1
        if frames>info['frames']:raise ValueError('追跡座標と動画の長さが一致しません。')
        if any(abs(float(r['time_s'])-int(r['frame'])/info['fps'])>1e-4 for r in rows):
            raise ValueError('追跡座標と動画のFPSが一致しません。')
        info['frames']=frames
        data=dict(schema_version=1,video=str(video),tracks=str(tracks),video_sha256=digest(video),
                  tracks_sha256=digest(tracks),slots=sorted({int(r['slot']) for r in rows}),
                  interval_s=1,labels={},history=[],**info)
        session=cls(folder/'session.json',data);folder.mkdir(parents=True,exist_ok=True);session.save();return session

    @classmethod
    def load(cls,path):
        path=Path(path);data=json.loads(path.read_text(encoding='utf-8'))
        if data.get('schema_version')!=1:raise ValueError('未対応の入力ファイルです。')
        for name in ('video','tracks'):
            source=Path(data[name])
            if not source.is_file():raise ValueError(f'元ファイルが見つかりません：{source}')
            if digest(source)!=data[name+'_sha256']:raise ValueError(f'元の{name}が変更されています。対応を確認してください。')
        return cls(path,data)

    def check(self,slot,second):
        if slot not in self.slots or not 0<=second<self.seconds:raise ValueError('個体または秒の指定が範囲外です。')

    def label(self,slot,second):return self.labels.get(f'{slot}:{second}','')

    def set_label(self,slot,second,value):
        self.check(slot,second)
        if value not in BEHAVIORS:raise ValueError('行動を選択してください。')
        key=f'{slot}:{second}';previous=self.label(slot,second)
        if previous==value:return
        entry=dict(slot=slot,start_s=second,previous=previous,behavior=value,updated_utc=datetime.now(timezone.utc).isoformat())
        self.labels[key]=value;self.data['history'].append(entry)
        try:self.save()
        except Exception:
            self.labels[key]=previous;self.data['history'].pop();raise

    def save(self):atomic_json(self.path,self.data)

    def frame_range(self,second):
        return max(0,math.ceil(second*self.fps-1e-8)),min(self.frames,math.ceil(min(second+1,self.duration)*self.fps-1e-8))

    def point(self,frame,slot):
        row=self.tracks.get((frame,slot),{})
        if row.get('status')!='observed':return None
        try:x,y=float(row['x_px']),float(row['y_px'])
        except (ValueError,TypeError,KeyError):return None
        if not math.isfinite(x+y) or not (0<=x<self.data['width'] and 0<=y<self.data['height']):return None
        return x,y

    def interval(self,slot,second):
        self.check(slot,second);a,b=self.frame_range(second)
        observed=[self.tracks[k,slot] for k in range(a,b) if self.point(k,slot) is not None]
        segments=sorted({int(r['segment']) for r in observed if r.get('segment')})
        label=self.label(slot,second);end=min(second+1,self.duration)
        return dict(slot=slot,start_s=second,end_s=end,duration_s=end-second,behavior=label,
                    behavior_label=BEHAVIORS[label],annotated=int(bool(label)),
                    observed_frame_fraction=len(observed)/(b-a) if b>a else 0,
                    segments=';'.join(map(str,segments)))

    def rows(self):
        return [self.interval(slot,second) for slot in self.slots for second in range(self.seconds)]

    def summary(self):
        result=[]
        for slot in self.slots:
            totals={value:0. for value in BEHAVIORS}
            for second in range(self.seconds):totals[self.label(slot,second)]+=min(second+1,self.duration)-second
            for value,seconds in totals.items():
                result.append(dict(slot=slot,behavior=value,behavior_label=BEHAVIORS[value],duration_s=seconds,
                                   fraction_total=seconds/self.duration))
        return result

    def export(self,folder=None):
        folder=Path(folder) if folder else self.path.parent;folder.mkdir(parents=True,exist_ok=True)
        rows=self.rows();summary=self.summary()
        save_csv(folder/'manual_behavior.csv',rows,list(rows[0]))
        save_csv(folder/'manual_behavior_summary.csv',summary,list(summary[0]))
        figure=behavior_figure(self)
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        FigureCanvasAgg(figure)
        figure.savefig(folder/'manual_behavior.png',dpi=160,bbox_inches='tight')
        return folder


def behavior_figure(session):
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch
    from matplotlib import font_manager
    available={font.name for font in font_manager.fontManager.ttflist}
    font=next((name for name in ('Hiragino Sans','Yu Gothic','Meiryo','Noto Sans CJK JP','Arial Unicode MS') if name in available),'DejaVu Sans')
    from matplotlib import rc_context
    with rc_context({'font.family':font}):
        fig=Figure(figsize=(12,max(5,2.8+len(session.slots)*.23)),layout='constrained')
        timeline,summary=fig.subplots(2,1,gridspec_kw={'height_ratios':[max(1,len(session.slots)/5),1]})
        for index,slot in enumerate(session.slots):
            runs=[]
            for second in range(session.seconds):
                label=session.label(slot,second);end=min(second+1,session.duration)
                if runs and runs[-1][2]==label:runs[-1]=(runs[-1][0],end,label)
                else:runs.append((second,end,label))
            for start,end,label in runs:
                timeline.broken_barh([(start,end-start)],(index-.38,.76),facecolors=COLORS[label])
        timeline.set(yticks=range(len(session.slots)),yticklabels=[f'個体 {s}' for s in session.slots],
                     xlabel='時刻（秒）',xlim=(0,session.duration),ylim=(len(session.slots)-.5,-.5),title='人手による1秒ごとの代表行動')
        left=[0.]*len(session.slots);totals=session.summary()
        for label in BEHAVIORS:
            values=[next(r['duration_s'] for r in totals if r['slot']==slot and r['behavior']==label) for slot in session.slots]
            summary.barh(range(len(session.slots)),values,left=left,color=COLORS[label])
            left=[a+b for a,b in zip(left,values)]
        summary.set(yticks=range(len(session.slots)),yticklabels=[f'個体 {s}' for s in session.slots],xlabel='時間（秒）',title='行動別の合計時間',xlim=(0,session.duration),ylim=(len(session.slots)-.5,-.5))
        fig.legend(handles=[Patch(facecolor=COLORS[k],label=v) for k,v in BEHAVIORS.items()],loc='outside lower center',ncol=4)
        return fig
