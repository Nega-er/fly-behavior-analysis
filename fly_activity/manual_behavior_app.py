#!/usr/bin/env python3
"""Desktop manual ethogram editor using the existing fly detector."""
import argparse
import faulthandler
from datetime import datetime
import json
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import ttk,filedialog,messagebox

import cv2
from PIL import Image
from tk_images import photo_image

from analyze_flies import DEFAULTS,validate_config,track_video
from video_setup import prepare_video
from manual_behavior import ManualSession,BEHAVIORS,behavior_figure
from progress import StackedProgress,STAGES

HERE=Path(__file__).resolve().parent
MANUAL_STAGES=(STAGES[0],STAGES[1],STAGES[-1])


class ManualApp:
    def __init__(self,root):
        self.root=root;root.title('ハエの個体行動を人手で記録');root.geometry('1180x880')
        self.session=None;self.cap=None;self.job=None;self.playing=False;self.busy=False;self.dialog_active=False;self.events=queue.Queue()
        self.graph_canvas=None;self.current_frame=0;self.crop=None;self.roi=None
        self.video=tk.StringVar();self.count=tk.StringVar(value='auto');self.polarity=tk.StringVar(value='bright')
        self.threshold=tk.StringVar(value='30');self.minimum=tk.StringVar(value='40');self.maximum=tk.StringVar(value='500')
        self.slot=tk.StringVar();self.second=tk.StringVar(value='0');self.behavior=tk.StringVar(value='未入力')
        self.rate=tk.StringVar(value='1');self.advance=tk.BooleanVar(value=True);self.status=tk.StringVar(value='動画を検出するか、既存の追跡結果・入力を開いてください。')
        self.interval_text=tk.StringVar();self.quality_text=tk.StringVar()
        self.source_text=tk.StringVar(value='解析済み：tracks.csvのある結果フォルダ ／ 入力の続き：session.json')
        top=ttk.Frame(root,padding=10);top.pack(fill='x')
        ttk.Label(top,text='動画').grid(row=0,column=0)
        ttk.Entry(top,textvariable=self.video,width=60).grid(row=0,column=1,columnspan=5,sticky='ew',padx=5)
        ttk.Button(top,text='選択…',command=self.choose_video).grid(row=0,column=6,padx=4)
        ttk.Button(top,text='動画を検出して開始',command=self.detect).grid(row=0,column=7,padx=4)
        top.columnconfigure(1,weight=1)
        settings=ttk.Frame(top);settings.grid(row=1,column=0,columnspan=8,sticky='w',pady=6)
        for title,var,width in [('個体数（auto可）',self.count,6),('輝度差',self.threshold,5),('最小面積',self.minimum,5),('最大面積',self.maximum,5)]:
            ttk.Label(settings,text=title).pack(side='left',padx=(8,3));ttk.Entry(settings,textvariable=var,width=width).pack(side='left')
        ttk.Combobox(settings,textvariable=self.polarity,values=['bright','dark','both'],width=7,state='readonly').pack(side='left',padx=8)
        ttk.Button(settings,text='検出領域を選択',command=self.select_roi).pack(side='left',padx=4)
        ttk.Button(settings,text='全画面',command=self.clear_roi).pack(side='left',padx=4)
        links=ttk.Frame(top);links.grid(row=2,column=0,columnspan=8,sticky='w')
        for title,command in [('既存の追跡結果を使う',self.open_tracking),('入力を再開',self.open_session),('CSV・グラフを書き出す',self.export)]:
            ttk.Button(links,text=title,command=command).pack(side='left',padx=4)
        ttk.Label(top,textvariable=self.source_text,wraplength=1100).grid(row=3,column=0,columnspan=8,sticky='w',pady=4)
        footer=ttk.Frame(root,padding=(10,5));footer.pack(side='bottom',fill='x')
        self.progress=StackedProgress(footer);self.progress.begin(MANUAL_STAGES);self.progress.pack(fill='x')
        ttk.Label(footer,textvariable=self.status,wraplength=1100).pack(anchor='w')
        self.tabs=ttk.Notebook(root);self.tabs.pack(fill='both',expand=True,padx=10)
        editor=ttk.Frame(self.tabs);self.graph_tab=ttk.Frame(self.tabs)
        self.tabs.add(editor,text='行動を入力');self.tabs.add(self.graph_tab,text='グラフ・集計')
        controls=ttk.Frame(editor);controls.pack(fill='x',pady=5)
        ttk.Label(controls,text='個体').pack(side='left',padx=4)
        self.slot_combo=ttk.Combobox(controls,textvariable=self.slot,state='readonly',width=8);self.slot_combo.pack(side='left')
        self.slot_combo.bind('<<ComboboxSelected>>',lambda e:self.select_interval())
        ttk.Label(controls,text='開始秒').pack(side='left',padx=(12,4))
        self.second_spin=ttk.Spinbox(controls,from_=0,to=0,textvariable=self.second,width=7,command=self.select_interval)
        self.second_spin.pack(side='left');self.second_spin.bind('<Return>',lambda e:self.select_interval())
        for title,command in [('◀ 前の秒',lambda:self.move(-1)),('次の秒 ▶',lambda:self.move(1)),('次の未入力',self.next_unlabeled)]:
            ttk.Button(controls,text=title,command=command).pack(side='left',padx=4)
        ttk.Label(controls,textvariable=self.interval_text).pack(side='left',padx=10)
        images=ttk.Frame(editor);images.pack(fill='both',expand=True)
        self.full_image=ttk.Label(images,anchor='center');self.full_image.pack(side='left',fill='both',expand=True)
        self.zoom_image=ttk.Label(images,anchor='center');self.zoom_image.pack(side='left',fill='both',expand=True)
        playback=ttk.Frame(editor);playback.pack(fill='x',pady=4)
        self.play_button=ttk.Button(playback,text='1秒区間を再生',command=self.toggle);self.play_button.pack(side='left')
        ttk.Label(playback,text='速度').pack(side='left',padx=5)
        ttk.Combobox(playback,textvariable=self.rate,values=['0.25','0.5','1','2'],state='readonly',width=5).pack(side='left')
        ttk.Label(playback,text='行動を選ぶまで繰り返し再生し、選択後は次の秒を自動再生します。右は選択個体の拡大です。').pack(side='left',padx=8)
        ttk.Label(editor,textvariable=self.quality_text,wraplength=1100).pack(anchor='w',pady=3)
        labels=ttk.Frame(editor);labels.pack(fill='x',pady=4)
        ttk.Label(labels,text='この区間の代表行動').pack(side='left',padx=4)
        self.behavior_combo=ttk.Combobox(labels,textvariable=self.behavior,values=list(BEHAVIORS.values()),state='readonly',width=18)
        self.behavior_combo.pack(side='left');self.behavior_combo.bind('<<ComboboxSelected>>',lambda e:self.record())
        ttk.Checkbutton(labels,text='選択後に次の秒へ',variable=self.advance).pack(side='left',padx=8)
        shortcuts=ttk.Frame(editor);shortcuts.pack(fill='x',pady=3)
        for index,(value,title) in enumerate(((key,title) for key,title in BEHAVIORS.items() if key not in ('unknown','')),1):
            ttk.Button(shortcuts,text=f'{title} {index}',command=lambda v=value:self.record(v)).pack(side='left',padx=2)
            root.bind(str(index),lambda e,v=value:self.shortcut(e,v))
        ttk.Button(shortcuts,text='Unknown U',command=lambda:self.record('unknown')).pack(side='left',padx=2)
        root.bind('u',lambda e:self.shortcut(e,'unknown'));root.bind('U',lambda e:self.shortcut(e,'unknown'))
        root.bind('<Left>',lambda e:self.arrow(e,-1));root.bind('<Right>',lambda e:self.arrow(e,1))
        ttk.Label(editor,text='1秒内に複数の行動がある場合は代表するものを1つ記録します。判別不能はUnknown、未入力は空欄です。自動学習には使いません。',wraplength=1100).pack(anchor='w',pady=3)
        table_frame=ttk.Frame(editor);table_frame.pack(fill='x',pady=4)
        columns=('time','behavior','coverage','segments')
        self.table=ttk.Treeview(table_frame,columns=columns,show='headings',height=5,selectmode='browse')
        for key,title,width in [('time','区間（秒）',180),('behavior','代表行動',240),('coverage','検出された割合',160),('segments','追跡区間',180)]:
            self.table.heading(key,text=title);self.table.column(key,width=width)
        scrollbar=ttk.Scrollbar(table_frame,orient='vertical',command=self.table.yview);self.table.configure(yscrollcommand=scrollbar.set)
        self.table.pack(side='left',fill='x',expand=True);scrollbar.pack(side='right',fill='y')
        self.table.bind('<Double-1>',self.table_select)
        ttk.Button(self.graph_tab,text='グラフを更新',command=self.show_graph).pack(anchor='w',pady=4)
        self.tabs.bind('<<NotebookTabChanged>>',lambda e:self.show_graph() if self.tabs.select()==str(self.graph_tab) else None)
        root.protocol('WM_DELETE_WINDOW',self.close);root.after(100,self.poll)

    def choose_video(self):
        if self.busy:return
        path=self.choose_path(filedialog.askopenfilename,title='動画を選択',filetypes=[('動画','*.avi *.mp4 *.mov *.mkv'),('すべて','*')])
        if path:self.video.set(path);self.roi=None

    def choose_path(self,dialog,**options):
        # Native file panels run a nested event loop. Do not decode or update Tk
        # images from an outstanding playback timer while that panel is active.
        was_playing=self.playing;self.pause();self.dialog_active=True
        chosen=None
        try:
            chosen=dialog(parent=self.root,**options)
            return chosen
        finally:
            self.dialog_active=False
            if not chosen and was_playing and not self.busy:self.start_loop()

    def clear_roi(self):
        if not self.busy:self.roi=None;self.status.set('全画面を検出します。')

    def select_roi(self):
        if self.busy:return
        cap=cv2.VideoCapture(self.video.get());ok,image=cap.read();cap.release()
        if not ok:messagebox.showerror('動画未選択','先に動画を選んでください。',parent=self.root);return
        factor=min(1,900/image.shape[1],650/image.shape[0])
        name='Select arena - ENTER confirm / ESC cancel'
        x,y,w,h=cv2.selectROI(name,cv2.resize(image,None,fx=factor,fy=factor),showCrosshair=True);cv2.destroyWindow(name)
        if w and h:self.roi=[round(v/factor) for v in (x,y,w,h)];self.status.set(f'検出領域：{self.roi}')

    def work(self,task):
        if self.busy:return
        self.pause();self.busy=True;self.progress.begin(MANUAL_STAGES)
        self.status.set('準備中です…')
        def run():
            try:self.events.put(('done',task()))
            except Exception as error:self.events.put(('error',str(error)))
        threading.Thread(target=run,daemon=True).start()

    def report(self,stage,current=0,total=1,detail=''):
        self.events.put(('progress',dict(stage=stage,current=current,total=total,detail=detail)))

    def detect(self):
        if self.busy:return
        try:
            video=Path(self.video.get()).resolve()
            if not video.is_file():raise ValueError('動画を選択してください。')
            count=self.count.get().strip().lower()
            config=dict(DEFAULTS,foreground_polarity=self.polarity.get(),threshold=float(self.threshold.get()),
                        min_area_px=float(self.minimum.get()),max_area_px=float(self.maximum.get()),
                        max_speed_px_s=1200,roi_rect_px=self.roi,expected_flies=7 if count=='auto' else int(count))
            validate_config(config)
        except Exception as error:messagebox.showerror('入力確認',str(error),parent=self.root);return
        folder=HERE/'outputs'/('manual_'+video.stem+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        def task():
            tracking=folder/'tracking';tracking.mkdir(parents=True)
            background,population,_=prepare_video(video,config,self.report)
            if count=='auto':config['expected_flies']=population['estimated_flies']
            _,metadata=track_video(video,config,tracking,background,progress=self.report)
            metadata['source']=str(video)
            (tracking/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
            (tracking/'config_used.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
            self.report('load',0,1,'入力用データを準備')
            return ManualSession.create(video,tracking/'tracks.csv',folder)
        self.work(task)

    def open_tracking(self):
        if self.busy:return
        chosen=self.choose_path(filedialog.askdirectory,title='tracks.csvのある追跡結果フォルダを選択')
        if not chosen:return
        tracking=Path(chosen);metadata={}
        try:
            if not (tracking/'tracks.csv').is_file():raise ValueError('tracks.csvが入っているフォルダを選択してください。')
            if (tracking/'metadata.json').exists():metadata=json.loads((tracking/'metadata.json').read_text(encoding='utf-8'))
        except (OSError,ValueError) as error:
            messagebox.showerror('追跡結果の読込み',str(error),parent=self.root);return
        video=Path(metadata.get('direction_video',metadata.get('source','')))
        if not video.is_file():
            selected=self.choose_path(filedialog.askopenfilename,title='この追跡結果の元動画を選択')
            if not selected:return
            video=Path(selected)
        folder=HERE/'outputs'/('manual_'+video.stem+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        self.work(lambda:ManualSession.create(video,tracking/'tracks.csv',folder))

    def open_session(self):
        if self.busy:return
        path=self.choose_path(filedialog.askopenfilename,title='入力を再開するsession.jsonを選択',filetypes=[('入力データ（session.json）','*.json')])
        if path:self.work(lambda:ManualSession.load(path))

    def attach(self,session):
        self.pause()
        cap=cv2.VideoCapture(str(session.video))
        if not cap.isOpened():cap.release();raise ValueError('動画を開けません。')
        if self.cap:self.cap.release()
        self.session=session;self.cap=cap;self.video.set(str(session.video))
        self.slot_combo.configure(values=[str(s) for s in session.slots]);self.slot.set(str(session.slots[0]));self.second.set('0')
        self.second_spin.configure(to=session.seconds-1)
        self.source_text.set(f'入力の保存先：{session.path}')
        self.tabs.select(0);self.select_interval()
        self.status.set('入力可能です。行動を選ぶたびに保存します。接触後の個体番号は映像で確認してください。')

    def selected(self):
        slot=int(self.slot.get());second=int(self.second.get());self.session.check(slot,second);return slot,second

    def select_interval(self):
        if self.busy or not self.session:return
        try:slot,second=self.selected()
        except ValueError:self.status.set('秒の指定を確認してください。');return
        self.pause();a,b=self.session.frame_range(second)
        points=[self.session.point(k,slot) for k in range(a,b)];points=[p for p in points if p is not None]
        self.crop=None
        if points:
            xs,ys=zip(*points);width=self.session.data['width'];height=self.session.data['height']
            self.crop=(max(0,int(min(xs)-70)),max(0,int(min(ys)-70)),min(width,int(max(xs)+71)),min(height,int(max(ys)+71)))
        info=self.session.interval(slot,second)
        self.interval_text.set(f'{second}〜{info["end_s"]:g} 秒 / {self.session.duration:g} 秒')
        self.behavior.set(BEHAVIORS[self.session.label(slot,second)])
        self.quality_text.set(f'この区間の検出割合：{info["observed_frame_fraction"]:.0%} / 追跡区間：{info["segments"] or "なし"}。'
                              +' 検出がない・区間が変わる場面は、対象個体を確認して入力してください。')
        self.refresh_table(slot)
        if self.show_frame(a):self.start_loop()

    def refresh_table(self,slot):
        self.table.delete(*self.table.get_children())
        for second in range(self.session.seconds):
            r=self.session.interval(slot,second)
            self.table.insert('', 'end',iid=str(second),values=(f'{second}〜{r["end_s"]:g}',r['behavior_label'],f'{r["observed_frame_fraction"]:.0%}',r['segments']))
        current=str(int(self.second.get()));self.table.selection_set(current);self.table.see(current)

    def image(self,widget,image,size):
        picture=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB));picture.thumbnail(size)
        widget.photo=photo_image(picture);widget.configure(image=widget.photo)

    def show_frame(self,frame):
        if int(self.cap.get(1))!=frame:self.cap.set(1,frame)
        ok,image=self.cap.read()
        if not ok:self.pause();self.status.set('動画を読み込めません。別の秒で確認してください。');return False
        self.current_frame=frame;slot,_=self.selected();zoom=image.copy()
        for other in self.session.slots:
            point=self.session.point(frame,other)
            if point is None:continue
            x,y=map(round,point);color=(0,220,255) if other==slot else (170,170,170)
            cv2.circle(image,(x,y),18 if other==slot else 10,color,2 if other==slot else 1)
            cv2.putText(image,str(other),(x+15,y-10),cv2.FONT_HERSHEY_SIMPLEX,.6,color,2)
        point=self.session.point(frame,slot)
        if point:cv2.circle(zoom,tuple(map(round,point)),18,(0,220,255),1)
        if self.crop:
            x0,y0,x1,y1=self.crop;zoom=zoom[y0:y1,x0:x1]
        self.image(self.full_image,image,(max(200,self.root.winfo_width()//2-25),300))
        self.image(self.zoom_image,zoom,(max(200,self.root.winfo_width()//2-25),300))
        return True

    def start_loop(self):
        if self.playing or self.busy or self.dialog_active or not self.session:return
        self.playing=True;self.play_button.configure(text='一時停止')
        self.job=self.root.after(max(10,round(1000/self.session.fps/float(self.rate.get()))),self.tick)

    def toggle(self):
        if self.busy or not self.session:return
        if self.playing:self.pause();return
        self.start_loop()

    def tick(self):
        self.job=None
        if not self.playing or self.busy or self.dialog_active:return
        _,second=self.selected();a,b=self.session.frame_range(second)
        frame=self.current_frame+1
        if frame>=b:frame=a
        self.show_frame(frame)
        if self.playing:self.job=self.root.after(max(10,round(1000/self.session.fps/float(self.rate.get()))),self.tick)

    def pause(self):
        self.playing=False;self.play_button.configure(text='1秒区間を再生')
        if self.job:self.root.after_cancel(self.job);self.job=None

    def move(self,delta):
        if self.busy or not self.session:return
        try:_,second=self.selected()
        except ValueError:return
        self.second.set(str(min(self.session.seconds-1,max(0,second+delta))));self.select_interval()

    def next_unlabeled(self):
        if self.busy or not self.session:return
        slot,second=self.selected()
        for candidate in list(range(second+1,self.session.seconds))+list(range(second+1)):
            if not self.session.label(slot,candidate):self.second.set(str(candidate));self.select_interval();return
        self.status.set('この個体の全区間が入力済みです。')

    def record(self,value=None):
        if self.busy or not self.session:return
        try:
            slot,second=self.selected()
            if value is None:value=next(k for k,v in BEHAVIORS.items() if v==self.behavior.get())
            self.session.set_label(slot,second,value)
            self.behavior.set(BEHAVIORS[value]);self.refresh_table(slot)
            self.status.set(f'保存済み：個体 {slot} / {second}秒 / {BEHAVIORS[value]}')
            if value and self.advance.get():
                if second<self.session.seconds-1:self.move(1)
                else:
                    self.pause()
                    self.status.set(f'個体 {slot} の最後の区間を保存しました。別の個体を選ぶか、入力内容を確認してください。')
        except Exception as error:messagebox.showerror('入力の保存',str(error),parent=self.root)

    def shortcut(self,event,value):
        if isinstance(event.widget,(ttk.Entry,ttk.Combobox,ttk.Spinbox,tk.Entry,tk.Spinbox)):return
        self.record(value)

    def arrow(self,event,delta):
        if isinstance(event.widget,(ttk.Entry,ttk.Combobox,ttk.Spinbox,tk.Entry,tk.Spinbox)):return
        self.move(delta)

    def table_select(self,event):
        item=self.table.identify_row(event.y)
        if item:self.second.set(item);self.select_interval()

    def show_graph(self):
        if not self.session:return
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        canvas=FigureCanvasAgg(behavior_figure(self.session));canvas.draw()
        image=Image.frombytes('RGBA',canvas.get_width_height(),canvas.buffer_rgba().tobytes())
        image.thumbnail((max(300,self.root.winfo_width()-60),max(200,self.root.winfo_height()-260)))
        if self.graph_canvas:self.graph_canvas.destroy()
        self.graph_canvas=ttk.Label(self.graph_tab,anchor='center')
        self.graph_canvas.photo=photo_image(image)
        self.graph_canvas.configure(image=self.graph_canvas.photo)
        self.graph_canvas.pack(fill='both',expand=True)

    def export(self):
        if self.busy or not self.session:return
        self.pause()
        try:
            folder=self.session.export();self.show_graph();self.tabs.select(self.graph_tab)
            self.status.set(f'CSV・グラフを書き出しました：{folder}')
        except Exception as error:messagebox.showerror('書き出し',str(error),parent=self.root)

    def poll(self):
        try:
            while True:
                kind,payload=self.events.get_nowait()
                if kind=='progress':self.progress.update_progress(**payload);continue
                self.busy=False
                if kind=='error':self.progress.fail();messagebox.showerror('読込み・検出',payload,parent=self.root)
                else:
                    try:self.attach(payload);self.progress.finish()
                    except Exception as error:self.progress.fail();messagebox.showerror('動画を開けません',str(error),parent=self.root)
        except queue.Empty:pass
        self.root.after(100,self.poll)

    def close(self):
        if self.busy:messagebox.showinfo('処理中','検出・読込みが終わってから閉じてください。',parent=self.root);return
        self.pause()
        if self.cap:self.cap.release()
        self.root.destroy()


def main():
    faulthandler.enable()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path);parser.add_argument('--smoke-test',action='store_true')
    args=parser.parse_args();root=tk.Tk();app=ManualApp(root)
    if args.smoke_test:
        root.withdraw()
        if args.session:app.attach(ManualSession.load(args.session))
        root.update_idletasks();root.update();app.close();print('Manual behavior editor constructed.')
    else:
        if args.session:app.work(lambda:ManualSession.load(args.session))
        root.mainloop()


if __name__=='__main__':main()
