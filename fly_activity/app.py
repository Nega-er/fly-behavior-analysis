#!/usr/bin/env python3
"""Desktop entry point: arbitrary video, population selection and blinded review."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk,filedialog,messagebox

from PIL import Image
from tk_images import photo_image
import cv2
from analyze_flies import DEFAULTS,validate_config
from video_setup import prepare_video,roi_mask
from results_view import ResultsView,load_results
from cumulative_learning import STORE,active_model_path
from progress import StackedProgress,STAGES,PREFIX,startup_sync_status
from slot_selection import parse_excluded_slots

HERE=Path(__file__).resolve().parent


class App:
    def __init__(self,root):
        self.root=root;root.title("ハエの移動・前進後退解析")
        root.geometry("1040x840")
        self.video=tk.StringVar();self.count=tk.StringVar(value="auto")
        self.output=tk.StringVar(value=str(HERE/"outputs"));self.background=tk.StringVar()
        self.polarity=tk.StringVar(value="bright");self.threshold=tk.StringVar(value="30")
        self.min_area=tk.StringVar(value="40");self.max_area=tk.StringVar(value="500")
        self.id_hold=tk.StringVar(value="15");self.static_hold=tk.StringVar(value="120");self.id_radius=tk.StringVar(value="240")
        self.speed=tk.StringVar(value="1200");self.eval_count=tk.StringVar(value="80")
        self.learn_enabled=tk.BooleanVar(value=True);self.model_status=tk.StringVar()
        self.sampling_mode=tk.StringVar(value="学習用（優先抽出）")
        self.body_mode=tk.StringVar(value="auto");self.body_brightness=tk.StringVar(value="0");self.bin_seconds=tk.StringVar(value="1")
        self.excluded_slots=tk.StringVar()
        self.grooming_enabled=tk.BooleanVar(value=False)
        self.status=tk.StringVar(value="動画を選択してください。頭側の事前クリックは不要です。")
        self.events=queue.Queue();self.running=False;self.last_result=None;self.roi=None
        self.tabs=ttk.Notebook(root);self.tabs.pack(fill="both",expand=True)
        self.input_tab=ttk.Frame(self.tabs);self.tabs.add(self.input_tab,text="入力・解析設定")
        self.results=ResultsView(self.tabs,self.browse_results,self.refresh_results,self.evaluate,self.open_result,self.rerun_latest)
        self.tabs.add(self.results,text="解析結果")
        main=ttk.Frame(self.input_tab,padding=15);main.pack(fill="both",expand=True)
        main.columnconfigure(1,weight=1)
        def entry(row,label,var,button=None,command=None):
            ttk.Label(main,text=label).grid(row=row,column=0,sticky="w",pady=4)
            ttk.Entry(main,textvariable=var).grid(row=row,column=1,sticky="ew",padx=8,pady=4)
            if button:ttk.Button(main,text=button,command=command).grid(row=row,column=2,sticky="ew")
        entry(0,"動画ファイル",self.video,"選択…",self.choose_video)
        entry(1,"出力先フォルダ",self.output,"選択…",self.choose_output)
        entry(2,"個体数（auto または整数）",self.count,"自動推定を確認",self.preflight)
        entry(3,"背景画像（任意）",self.background,"選択…",self.choose_background)
        controls=ttk.Frame(main);controls.grid(row=4,column=0,columnspan=3,sticky="ew",pady=8)
        ttk.Label(controls,text="ハエの明暗").pack(side="left")
        ttk.Combobox(controls,textvariable=self.polarity,values=["bright","dark","both"],width=8,state="readonly").pack(side="left",padx=5)
        for title,var in [("輝度差",self.threshold),("最小面積",self.min_area),("最大面積",self.max_area),("速度上限",self.speed)]:
            ttk.Label(controls,text=title).pack(side="left",padx=(8,2))
            ttk.Entry(controls,textvariable=var,width=6).pack(side="left")
        body=ttk.Frame(main);body.grid(row=5,column=0,columnspan=3,sticky="ew",pady=5)
        ttk.Label(body,text="本体検出").pack(side="left")
        ttk.Combobox(body,textvariable=self.body_mode,values=["auto","body_core","background_diff"],state="readonly",width=17).pack(side="left",padx=5)
        ttk.Label(body,text="本体輝度下限（0=自動）").pack(side="left",padx=5)
        ttk.Entry(body,textvariable=self.body_brightness,width=5).pack(side="left")
        ttk.Label(body,text="移動量の集計幅（秒）").pack(side="left",padx=10)
        ttk.Entry(body,textvariable=self.bin_seconds,width=5).pack(side="left")
        ttk.Label(body,text="auto：明るい本体を優先し、影を除外").pack(side="left",padx=8)
        identity=ttk.Frame(main);identity.grid(row=6,column=0,columnspan=3,sticky="ew",pady=4)
        for label,var in [("再識別の保持時間（秒）",self.id_hold),("静止ID保持（秒）",self.static_hold),("再識別の最大距離（px）",self.id_radius)]:
            ttk.Label(identity,text=label).pack(side="left",padx=5)
            ttk.Entry(identity,textvariable=var,width=6).pack(side="left")
        region=ttk.Frame(main);region.grid(row=7,column=0,columnspan=3,sticky="ew",pady=6)
        ttk.Button(region,text="解析領域を選ぶ",command=self.select_roi).pack(side="left",padx=3)
        ttk.Button(region,text="全画面に戻す",command=self.clear_roi).pack(side="left",padx=3)
        ttk.Label(region,text="任意動画では容器内を選ぶと反射などを除外できます。面積: px² / 速度: px/s").pack(side="left",padx=8)
        exclusion=ttk.Frame(main);exclusion.grid(row=11,column=0,columnspan=3,sticky='w',pady=4)
        ttk.Label(exclusion,text='解析から除外するID').pack(side='left')
        ttk.Entry(exclusion,textvariable=self.excluded_slots,width=20).pack(side='left',padx=5)
        ttk.Label(exclusion,text='例：2,5（空欄は全個体）。解析後は結果画面でも指定できます。').pack(side='left')
        ttk.Checkbutton(exclusion,text='グルーミング試作も実行',variable=self.grooming_enabled).pack(side='left',padx=8)
        self.results.on_exclude=self.exclude_ids
        self.results.on_grooming=self.grooming_comparison
        sampling=ttk.Frame(main);sampling.grid(row=10,column=0,columnspan=3,sticky="w",pady=4)
        ttk.Label(sampling,text="人手入力の場面選択").pack(side="left",padx=4)
        ttk.Combobox(sampling,textvariable=self.sampling_mode,values=["学習用（優先抽出）","評価用（ランダム）"],state="readonly",width=23).pack(side="left",padx=5)
        ttk.Label(sampling,text="学習用：Unknown・後退候補・頭尾低確信度を優先").pack(side="left",padx=8)
        self.preview=ttk.Label(main,anchor="center");self.preview.grid(row=8,column=0,columnspan=3,pady=8,sticky="nsew")
        main.rowconfigure(8,weight=1)
        action=ttk.Frame(main);action.grid(row=9,column=0,columnspan=3,pady=8,sticky="ew")
        ttk.Button(action,text="自動解析を実行",command=self.run).pack(side="left",padx=4)
        ttk.Button(action,text="過去の結果を表示",command=self.browse_results).pack(side="left",padx=4)
        ttk.Button(action,text="結果フォルダを開く",command=self.open_result).pack(side="left",padx=4)
        ttk.Button(action,text="人手学習・評価",command=self.evaluate).pack(side="left",padx=4)
        ttk.Label(action,text="評価数").pack(side="left",padx=(12,4))
        ttk.Entry(action,textvariable=self.eval_count,width=5).pack(side="left")
        ttk.Checkbutton(action,text="入力ごとに学習",variable=self.learn_enabled).pack(side="left",padx=8)
        footer=ttk.Frame(root,padding=(15,5))
        footer.pack(fill="x",before=self.tabs)
        footer.pack_configure(side="bottom")
        self.progress=StackedProgress(footer);self.progress.pack(fill="x")
        ttk.Label(footer,text=startup_sync_status(),wraplength=990).pack(anchor='w')
        ttk.Label(footer,textvariable=self.status,wraplength=990).pack(anchor="w",pady=(5,0))
        ttk.Label(main,text="自動頭尾判定は外見モデルの推定です。人手評価は自動ラベルを隠して行い、元の判定を変更しません。\n"
                            "頭側入力と人手評価を累積学習します。モデルを履歴保存し、以前の解析結果は固定します。",
                  wraplength=850).grid(row=12,column=0,columnspan=3,sticky="w",pady=5)
        ttk.Label(root,textvariable=self.model_status,wraplength=990).pack(fill="x",padx=12,pady=5)
        self.update_model_status()
        root.after(150,self.poll)

    def update_model_status(self):
        try:active_model_path()
        except (OSError,ValueError) as e:
            self.model_status.set(f"学習結果を読み込めません：{e}（保存先：{STORE}）")
            return
        pointer=STORE/"current.json"
        if pointer.exists():
            data=json.loads(pointer.read_text(encoding="utf-8"))
            self.model_status.set(f"最新学習モデル v{data['version']}：評価入力 {data['labels']}件 / 行動学習 {data['behavior_samples']}件 / 頭尾の追加学習 {data['head_samples']}件。再起動後も引き継ぎます。保存先：{STORE}")
        else:self.model_status.set(f"初期モデルを使用中。評価入力ごとに学習を保存し、再起動後も引き継ぎます。保存先：{STORE}")

    def choose_video(self):
        if self.running:return
        path=filedialog.askopenfilename(title="解析する動画を選択",filetypes=[("動画","*.mp4 *.avi *.mov *.mkv *.m4v"),("すべて","*")])
        if path:
            self.video.set(path);self.background.set("");self.roi=None
            cap=cv2.VideoCapture(path);ok,image=cap.read();cap.release()
            if not ok:messagebox.showerror("動画を開けません","OpenCVで読み取れる動画を選択してください。",parent=self.root);return
            self.display(image);self.status.set("動画を選択しました。必要に応じて解析領域・検出条件を設定してください。")

    def choose_output(self):
        path=filedialog.askdirectory(title="出力先の親フォルダ")
        if path:self.output.set(path)

    def choose_background(self):
        path=filedialog.askopenfilename(title="この動画の背景画像を選択",filetypes=[("画像","*.png *.jpg *.tif *.bmp"),("すべて","*")])
        if path:self.background.set(path)

    def display(self,image):
        image=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB));image.thumbnail((850,340))
        self.preview.photo=photo_image(image);self.preview.configure(image=self.preview.photo)

    def select_roi(self):
        if self.running:return
        cap=cv2.VideoCapture(self.video.get());ok,image=cap.read();cap.release()
        if not ok:messagebox.showerror("動画未選択","先に動画を選択してください。",parent=self.root);return
        factor=min(1.,900/image.shape[1],650/image.shape[0])
        scaled=cv2.resize(image,None,fx=factor,fy=factor)
        x,y,w,h=cv2.selectROI("Select arena: drag, ENTER confirm, ESC cancel",scaled,showCrosshair=True)
        cv2.destroyWindow("Select arena: drag, ENTER confirm, ESC cancel")
        if w and h:
            self.roi=[round(x/factor),round(y/factor),round(w/factor),round(h/factor)]
            self.roi[2]=min(self.roi[2],image.shape[1]-self.roi[0]);self.roi[3]=min(self.roi[3],image.shape[0]-self.roi[1])
            cv2.rectangle(image,tuple(self.roi[:2]),(self.roi[0]+self.roi[2],self.roi[1]+self.roi[3]),(0,255,0),2)
            self.display(image);self.status.set(f"解析領域: {self.roi}")

    def clear_roi(self):
        self.roi=None;self.status.set("全画面を解析します。")

    def config(self):
        c=dict(DEFAULTS,grooming_enabled=self.grooming_enabled.get(),excluded_slots=parse_excluded_slots(self.excluded_slots.get()),background_path=self.background.get() or None,
               foreground_polarity=self.polarity.get(),threshold=float(self.threshold.get()),
               min_area_px=float(self.min_area.get()),max_area_px=float(self.max_area.get()),
               max_speed_px_s=float(self.speed.get()),roi_rect_px=self.roi,
               identity_hold_s=float(self.id_hold.get()),stationary_identity_hold_s=float(self.static_hold.get()),
               identity_recovery_radius_px=float(self.id_radius.get()),
               body_detection_mode=self.body_mode.get(),body_min_brightness=float(self.body_brightness.get()),bin_s=float(self.bin_seconds.get()))
        value=self.count.get().strip().lower()
        if value!="auto":c["expected_flies"]=int(value)
        validate_config(c)
        return c,value

    def report_progress(self,stage,current=0,total=1,detail=""):
        self.events.put(("progress",dict(stage=stage,current=current,total=total,detail=detail)))

    def start(self,task,stages=STAGES):
        if self.running:return
        if stages is not None:
            self.progress.begin(stages)
            self.progress.update_progress(stages[0][0],0,1,"開始しています")
        self.running=True
        def work():
            try:self.events.put(("done",task()))
            except Exception as e:self.events.put(("error",str(e)))
        threading.Thread(target=work,daemon=True).start()

    def preflight(self):
        try:c,_=self.config();video=Path(self.video.get());assert video.is_file(),"動画を選択してください"
        except Exception as e:messagebox.showerror("入力確認",str(e),parent=self.root);return
        self.status.set("複数の時点から個体候補数を推定しています…")
        def task():
            background,population,rows=prepare_video(video,c,self.report_progress)
            return ("population",population)
        self.start(task,STAGES[:1])

    def run(self):
        try:
            c,count=self.config();video=Path(self.video.get());assert video.is_file(),"動画を選択してください"
            parent=Path(self.output.get());assert str(parent)!=".","出力先を選択してください"
        except Exception as e:messagebox.showerror("入力確認",str(e),parent=self.root);return
        self.launch_analysis(video,c,count,parent)

    def rerun_latest(self):
        if self.running or self.results.data is None:return
        data=self.results.data;metadata=data["metadata"]
        video=Path(metadata.get("direction_video",metadata.get("source","")))
        if not video.is_file():messagebox.showerror("動画なし","元動画が見つからないため再解析できません。",parent=self.root);return
        c=dict(DEFAULTS,**data["config"])
        self.launch_analysis(video,c,str(c["expected_flies"]),data["folder"].parent)

    def exclude_ids(self):
        if self.running or self.results.data is None:return
        data=self.results.data
        try:
            c=dict(DEFAULTS,**data['config'])
            c['excluded_slots']=parse_excluded_slots(self.results.excluded_ids.get())
            n=int(c['expected_flies'])
            if any(s>n for s in c['excluded_slots']):raise ValueError(f'除外IDは1〜{n}で指定してください。')
            if len(c['excluded_slots'])>=n:raise ValueError('解析対象を1個体以上残してください。')
            tracks=data['folder']/'tracks.csv'
            if not tracks.is_file():raise ValueError('元のtracks.csvが見つかりません。')
            video=Path(data['metadata'].get('direction_video',data['metadata'].get('source','')))
            if not video.is_file():raise ValueError('元動画が見つかりません。')
            model=data['metadata'].get('head_model')
            if model and not Path(model).is_file():raise ValueError('解析に使用したモデルが見つかりません。')
        except (ValueError,KeyError) as error:
            messagebox.showerror('除外IDの確認',str(error),parent=self.root);return
        self.launch_analysis(video,c,str(n),data['folder'].parent,tracks=tracks,head_model=model)

    def grooming_comparison(self,compare=False):
        if self.running or self.results.data is None:return
        data=self.results.data
        existing=data['folder']/'grooming'
        from grooming import VERSION
        cached=(existing/'grooming_metadata.json')
        if not compare and cached.exists() and json.loads(cached.read_text(encoding='utf-8')).get('method')==VERSION:
            from grooming_view import GroomingWindow
            self.grooming_window=GroomingWindow(self.root,existing);return
        video=Path(data['metadata'].get('direction_video',data['metadata'].get('source','')))
        tracks=data['folder']/'tracks.csv'
        if not video.is_file() or not tracks.is_file():
            messagebox.showerror('元データなし','元動画とtracks.csvが必要です。',parent=self.root);return
        chosen=None
        if compare:
            for player in self.results.players:
                if player.window.winfo_exists():player.pause()
            chosen=filedialog.askopenfilename(parent=self.root,title='比較するsession.jsonを選択',filetypes=[('JSON','*.json')])
            if not chosen:return
        folder=data['folder']/('grooming_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        from grooming import run_prototype
        self.start(lambda:('grooming_result',run_prototype(video,tracks,folder,chosen or None,
                     excluded_slots=data['group'].get('excluded_slots',[]),progress=self.report_progress,learning=HERE/'learning')),STAGES[-1:])

    def launch_analysis(self,video,c,count,parent,tracks=None,head_model=None):
        if self.running:return
        result=parent/(video.stem+"_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        self.status.set("自動解析中です。終了後に人手評価を実行できます。")
        def task():
            parent.mkdir(parents=True,exist_ok=True)
            config_path=parent/(result.name+"_input.json")
            config_path.write_text(json.dumps(c,indent=2,ensure_ascii=False),encoding="utf-8")
            args=[sys.executable,str(HERE/"analyze_flies.py"),"--video",str(video),"--config",str(config_path),
                  "--flies",count,"--out",str(result),"--progress"]
            if tracks:
                args[2:4]=['--tracks',str(tracks)]
                args.extend(['--direction-video',str(video)])
                anchors=tracks.parent/'head_anchors_used.csv'
                if anchors.exists():args.extend(['--head-anchors',str(anchors)])
            if head_model:args.extend(['--head-model',str(head_model)])
            lines=[]
            with subprocess.Popen(args,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,encoding="utf-8",
                                  env=dict(os.environ,PYTHONUTF8="1")) as process:
                for line in process.stdout:
                    lines.append(line)
                    if line.startswith(PREFIX):
                        self.events.put(("progress",json.loads(line[len(PREFIX):])))
                returncode=process.wait()
            log="".join(lines)
            (parent/(result.name+"_log.txt")).write_text(log,encoding="utf-8")
            if returncode:raise RuntimeError(log[-2500:])
            return ("result",str(result))
        self.start(task)

    def open_result(self):
        if not self.last_result:
            messagebox.showinfo("結果未選択","解析を実行するか、精度評価から既存結果を選択してください。",parent=self.root)
            return
        try:
            if sys.platform=="win32":os.startfile(str(self.last_result))
            elif sys.platform=="darwin":subprocess.Popen(["open",str(self.last_result)])
            else:subprocess.Popen(["xdg-open",str(self.last_result)])
        except OSError as e:
            messagebox.showerror("フォルダを開けません",str(e),parent=self.root)

    def browse_results(self):
        if self.running:return
        chosen=filedialog.askdirectory(title="解析結果フォルダを選択",initialdir=str(self.last_result.parent if self.last_result else HERE))
        if chosen:self.load_results(Path(chosen))

    def load_results(self,path,continue_analysis=False):
        if self.running:return
        self.status.set("結果を読み込んでいます…")
        self.results.path.configure(text="結果を読み込んでいます…")
        self.start(lambda:("loaded_results",load_results(path,self.report_progress)),
                   None if continue_analysis else STAGES[-1:])

    def refresh_results(self):
        if self.last_result:self.load_results(self.last_result)
        else:self.browse_results()

    def watch_evaluation(self,process,result):
        if process.poll() is None:
            self.update_model_status()
            self.root.after(500,lambda:self.watch_evaluation(process,result));return
        if process.returncode==0:
            self.update_model_status()
            self.status.set("人手学習・評価を保存しました。")
            if not self.running and self.last_result==result:self.load_results(result)
        else:
            self.status.set("人手評価を終了しました。結果を再読込して評価の保存状況を確認してください。")

    def evaluate(self):
        if self.running:return
        result=self.last_result
        if result is None:
            chosen=filedialog.askdirectory(title="自動解析の結果フォルダを選択")
            if not chosen:return
            result=Path(chosen)
        try:
            count=int(self.eval_count.get());assert count>0,"評価数は正の整数にしてください"
            assert (result/"activity.csv").exists(),"activity.csvのある結果フォルダを選択してください"
            metadata=json.loads((result/"metadata.json").read_text(encoding="utf-8"))
            assert metadata.get("head_mode")=="automatic_appearance","自動頭尾判定を使った新しい解析結果を選択してください"
        except Exception as e:messagebox.showerror("入力確認",str(e),parent=self.root);return
        self.last_result=result
        args=[sys.executable,str(HERE/"evaluate.py"),"--results",str(result),"--samples",str(count),
              "--sampling","learning" if self.sampling_mode.get()=="学習用（優先抽出）" else "random"]
        if not self.learn_enabled.get() or self.sampling_mode.get()=="評価用（ランダム）":args.append("--no-learn")
        process=subprocess.Popen(args)
        self.root.after(500,lambda:self.watch_evaluation(process,result))

    def poll(self):
        try:
            while True:
                kind,payload=self.events.get_nowait()
                if kind=="progress":
                    self.progress.update_progress(**payload)
                    continue
                self.running=False
                if kind=="error":
                    self.progress.fail()
                    self.status.set("処理に失敗しました。入力条件とログを確認してください。")
                    messagebox.showerror("処理エラー",payload,parent=self.root)
                elif payload[0]=="population":
                    self.progress.finish()
                    p=payload[1];self.status.set(f"推定 {p['estimated_flies']}個体（最頻値の割合 {p['modal_fraction']:.0%}、検出数範囲 {p['count_min']}〜{p['count_max']}）。誤推定なら個体数を整数で指定してください。")
                elif payload[0]=="loaded_results":
                    data=payload[1];self.last_result=data["folder"];self.results.show(data);self.tabs.select(self.results)
                    self.progress.finish()
                    self.status.set(f"結果表示：{self.last_result}")
                elif payload[0]=='grooming_result':
                    from grooming_view import GroomingWindow
                    self.grooming_window=GroomingWindow(self.root,payload[1]);self.progress.finish()
                    self.status.set(f'グルーミング試作を保存しました：{payload[1]}')
                else:
                    self.last_result=Path(payload[1]);p=json.loads((self.last_result/"config_used.json").read_text(encoding="utf-8"))
                    group=json.loads((self.last_result/'summary_group.json').read_text(encoding='utf-8'))
                    self.status.set(f"解析完了：{group.get('analyzed_flies',p['expected_flies'])}個体 / {self.last_result}。人間による精度評価を実行できます。")
                    overlay=self.last_result/'first_frame_overlay.jpg'
                    if overlay.exists():self.display(cv2.imread(str(overlay)))
                    self.load_results(self.last_result,continue_analysis=True)
        except queue.Empty:pass
        self.root.after(150,self.poll)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--smoke-test",action="store_true")
    parser.add_argument("--results",type=Path,help="Load an existing result directory at startup")
    a=parser.parse_args()
    root=tk.Tk()
    if a.smoke_test:root.withdraw()
    app=App(root)
    if a.smoke_test:
        root.update_idletasks();app.config()
        if a.results:
            data=load_results(a.results);app.results.show(data);app.tabs.select(app.results);root.update_idletasks()
            assert len(app.results.slot_table.get_children())==len(data["slots"])
        print("GUI and result dashboard constructed; configuration and result rows validated.");root.destroy()
    else:
        if a.results:app.load_results(a.results)
        root.mainloop()
