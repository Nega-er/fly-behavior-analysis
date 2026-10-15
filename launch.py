#!/usr/bin/env python3
"""Check dependencies, install missing requirements in .venv, then launch an app."""
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
APPS={'analysis':'app.py','manual':'manual_behavior_app.py','analyze':'analyze_flies.py','annotate':'annotate_heads.py'}
PROBE=r'''
import importlib,importlib.metadata,json,re,sys
result={'supported':sys.version_info>=(3,10),'tk':False,'missing':[]}
try:
    import tkinter
    result['tk']=True
except ImportError:
    pass
modules={'numpy':'numpy','opencv-python':'cv2','scipy':'scipy','matplotlib':'matplotlib','Pillow':'PIL'}
for line in open(sys.argv[1],encoding='utf-8'):
    line=line.strip()
    if not line or line.startswith('#'):continue
    match=re.fullmatch(r'([A-Za-z0-9_-]+)>=(\d+(?:\.\d+)*)',line)
    if not match:raise ValueError('Unsupported requirement: '+line)
    package,minimum=match.groups()
    try:
        version=importlib.metadata.version(package)
        parts=tuple(int(n) for n in re.match(r'\d+(?:\.\d+)*',version)[0].split('.'))
        required=tuple(int(n) for n in minimum.split('.'))
        length=max(len(parts),len(required))
        if parts+(0,)*(length-len(parts))<required+(0,)*(length-len(required)):raise ValueError('outdated')
        importlib.import_module(modules[package])
    except Exception:
        result['missing'].append(line)
print(json.dumps(result))
'''


def probe(python,requirements):
    run=subprocess.run([str(python),'-c',PROBE,str(requirements)],capture_output=True,text=True,
                       encoding='utf-8',env=dict(os.environ,PYTHONUTF8='1'))
    if run.returncode:raise RuntimeError('Python環境を確認できません。\n'+run.stderr[-2000:])
    return json.loads(run.stdout.strip().splitlines()[-1])


def ensure_environment(root=HERE,python=None):
    root=Path(root);requirements=root/'fly_activity/requirements.txt'
    local=root/'.venv'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    selected=local if local.is_file() else Path(python or sys.executable)
    state=probe(selected,requirements)
    if not state['supported']:raise RuntimeError('Python 3.10以上をインストールしてから起動してください。')
    if not state['tk']:raise RuntimeError('Tkinterが使えません。Tcl/Tkを含むPythonをインストールしてください（pipでは追加できません）。')
    if not state['missing']:return selected
    if selected!=local:
        print('必要なライブラリが不足しています。このフォルダ内に専用環境 .venv を作成します。',flush=True)
        subprocess.run([str(selected),'-m','venv',str(root/'.venv')],check=True)
        selected=local
    print('requirements.txt のライブラリをインストールしています。初回はインターネット接続が必要です。',flush=True)
    subprocess.run([str(selected),'-m','pip','install','--disable-pip-version-check','-r',str(requirements)],check=True)
    state=probe(selected,requirements)
    if not state['supported'] or not state['tk'] or state['missing']:
        raise RuntimeError('インストール後の環境を確認できません。表示されたエラーとネットワーク接続を確認してください。')
    return selected


def main():
    app=sys.argv[1] if len(sys.argv)>1 else 'analysis'
    if app not in APPS:
        print('起動先が不正です。analysis / manual / analyze / annotate を指定してください。',file=sys.stderr);return 2
    try:
        python=ensure_environment()
        return subprocess.call([str(python),str(HERE/'fly_activity'/APPS[app]),*sys.argv[2:]],
                               env=dict(os.environ,PYTHONUTF8='1'),cwd=HERE)
    except (OSError,RuntimeError,subprocess.CalledProcessError) as error:
        print('起動準備に失敗しました：'+str(error),file=sys.stderr);return 1


if __name__=='__main__':sys.exit(main())
