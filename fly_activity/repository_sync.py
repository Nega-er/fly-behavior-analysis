"""Shared, nonblocking repository controls for both desktop applications."""
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk,messagebox

from progress import startup_sync_status


def run_sync(root,report):
    env=dict(os.environ,PYTHONIOENCODING='utf-8')
    options=dict(creationflags=subprocess.CREATE_NO_WINDOW) if os.name=='nt' else {}
    process=subprocess.Popen([sys.executable,str(Path(root)/'setup/sync_now.py'),'--root',str(root)],
        cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',**options)
    result=None;tail=[]
    for line in process.stdout:
        tail.append(line);tail=tail[-12:]
        if line.startswith('FLY_SYNC_PROGRESS '):
            record=json.loads(line[len('FLY_SYNC_PROGRESS '):]);report(record['detail'])
        elif line.startswith('FLY_SYNC_RESULT '):result=json.loads(line[len('FLY_SYNC_RESULT '):])
    process.stdout.close();code=process.wait()
    if code or result is None:raise RuntimeError('同期処理に失敗しました。 '+''.join(tail)[-2500:])
    return result


class RepositorySync(ttk.Frame):
    def __init__(self,parent,root,app,can_start,on_start,on_finish,on_close,restart_args=lambda:()):
        super().__init__(parent)
        self.root=self.winfo_toplevel();self.repository=Path(root);self.app=app
        self.can_start=can_start;self.on_start=on_start;self.on_finish=on_finish;self.on_close=on_close
        self.restart_args=restart_args;self.active=False;self.events=queue.Queue();self.controls=[]
        self.text=tk.StringVar(value=startup_sync_status());self.result=None
        self.columnconfigure(0,weight=1)
        ttk.Label(self,textvariable=self.text,wraplength=720).grid(row=0,column=0,sticky='w',padx=(0,8))
        self.button=ttk.Button(self,text='更新・同期',command=self.start);self.button.grid(row=0,column=1,sticky='e')
        self.restart_button=ttk.Button(self,text='再起動して反映',command=self.restart)
        self.details_button=ttk.Button(self,text='詳細',command=self.details)
        self.bar=ttk.Progressbar(self,mode='indeterminate')
        self.root.after(100,self.poll)

    def freeze(self):
        self.controls=[]
        def visit(parent):
            for child in parent.winfo_children():
                if child is self:continue
                if isinstance(child,ttk.Widget) and hasattr(child,'state'):
                    state=child.state()
                    child.state(['disabled']);self.controls.append((child,state))
                visit(child)
        visit(self.root)

    def thaw(self):
        for widget,state in self.controls:
            if widget.winfo_exists():widget.state(['!disabled',*state])
        self.controls=[]

    def start(self):
        if self.active:return
        if not self.can_start():
            self.text.set('処理中です。解析・人手評価・読込みの終了後に更新してください。');return
        try:self.on_start()
        except Exception as error:self.text.set('同期準備に失敗しました：'+str(error));return
        self.active=True;self.button.state(['disabled']);self.restart_button.grid_remove();self.details_button.grid_remove()
        self.freeze();self.text.set('更新・同期中：GitHubへ接続しています…')
        self.bar.grid(row=1,column=0,columnspan=3,sticky='ew',pady=(4,0));self.bar.start(12)
        def worker():
            try:
                result=run_sync(self.repository,lambda detail:self.events.put(('progress',detail)))
            except Exception as error:result=dict(status='failed',uploaded=False,detail=str(error))
            self.events.put(('result',result))
        threading.Thread(target=worker,daemon=True).start()

    def poll(self):
        try:
            while True:
                kind,value=self.events.get_nowait()
                if kind=='progress':self.text.set('更新・同期中：'+value);continue
                self.result=value;self.active=False;self.bar.stop();self.bar.grid_remove()
                self.thaw();self.button.state(['!disabled'])
                detail=value.get('detail','同期状態を確認できません。').splitlines()[0][:220]
                if value.get('status') in ('failed','skipped'):
                    self.details_button.grid(row=0,column=3,padx=(6,0))
                if value.get('restart_required'):
                    self.freeze();self.button.state(['disabled'])
                    self.restart_button.grid(row=0,column=2,padx=(6,0))
                    detail+=' 更新を反映するため再起動してください（入力は保存済み）。'
                self.text.set('GitHub同期：'+detail)
                self.on_finish(value)
        except queue.Empty:pass
        if self.winfo_exists():self.root.after(100,self.poll)

    def details(self):
        messagebox.showinfo('更新・同期の詳細',(self.result or {}).get('detail','状態未確認'),parent=self.root)

    def restart(self):
        if self.active:return
        try:
            env=dict(os.environ)
            if (self.result or {}).get('uploaded'):env['FLY_UPDATE_DONE']='1'
            else:env.pop('FLY_UPDATE_DONE',None)
            options=dict(creationflags=subprocess.CREATE_NO_WINDOW) if os.name=='nt' else {}
            subprocess.Popen([sys.executable,str(self.repository/'launch.py'),self.app,*self.restart_args()],
                             cwd=self.repository,env=env,**options)
        except OSError as error:self.text.set('再起動できませんでした：'+str(error));return
        self.on_close()
