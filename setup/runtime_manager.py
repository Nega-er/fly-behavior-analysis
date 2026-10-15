"""Build and verify an isolated, pinned runtime without touching Conda or system Python."""
import argparse
from contextlib import contextmanager
from datetime import datetime,timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import uuid

ROOT=Path(__file__).resolve().parent.parent
MODULES={'numpy':'numpy','opencv-python':'cv2','scipy':'scipy','matplotlib':'matplotlib','Pillow':'PIL',
         'contourpy':'contourpy','cycler':'cycler','fonttools':'fontTools','kiwisolver':'kiwisolver',
         'packaging':'packaging','pyparsing':'pyparsing','python-dateutil':'dateutil','six':'six'}
APPS={'manual':'manual_behavior_app.py','analysis':'app.py','analyze':'analyze_flies.py','annotate':'annotate_heads.py'}


def clean_environment(root=ROOT,source=None):
    env=dict(os.environ if source is None else source)
    prefixes=('CONDA','MAMBA','_CE_','PYTHON','VIRTUAL_ENV','TCL_LIBRARY','TK_LIBRARY','UV_','DYLD_','LD_LIBRARY_PATH')
    for key in list(env):
        if key.startswith(prefixes):env.pop(key,None)
    runtime=Path(root)/'.runtime'
    env.update(UV_PYTHON_INSTALL_DIR=str(runtime/'python'),UV_CACHE_DIR=str(runtime/'cache'),
               UV_PYTHON_BIN_DIR=str(runtime/'tools'),UV_NO_MODIFY_PATH='1',UV_SYSTEM_CERTS='true',
               PYTHONUTF8='1',PYTHONNOUSERSITE='1',MPLCONFIGDIR=str(runtime/'matplotlib'))
    if os.name=='nt':
        system=env.get('SystemRoot',r'C:\Windows');env['PATH']=system+r'\System32;'+system
    else:env['PATH']='/usr/bin:/bin:/usr/sbin:/sbin'
    return env


def expected_versions(root=ROOT):
    result={}
    for line in (Path(root)/'setup/requirements.in').read_text().splitlines():
        line=line.strip()
        if line and not line.startswith('#'):
            name,version=line.split('==');result[name]=version
    return result


def configure_tk(env,base):
    """Standalone Python keeps Tcl/Tk data outside a newly created venv."""
    base=Path(base)
    for variable,pattern,filename in (('TCL_LIBRARY','tcl*','init.tcl'),('TK_LIBRARY','tk*','tk.tcl')):
        for parent in (base/'lib',base/'tcl',base):
            match=next((p for p in sorted(parent.glob(pattern)) if (p/filename).is_file()),None)
            if match:
                env[variable]=str(match);break
        else:raise RuntimeError(f'専用Pythonに {filename} が見つかりません。')
    return env


def dependency_report(root=ROOT):
    missing=[];versions={}
    for name,expected in expected_versions(root).items():
        try:
            actual=importlib.metadata.version(name);versions[name]=actual
            if actual!=expected:missing.append(f'{name}: {actual} != {expected}')
            importlib.import_module(MODULES.get(name,name.replace('-','_')))
        except Exception as error:missing.append(f'{name}: {error}')
    return dict(python=platform.python_version(),executable=sys.executable,base_prefix=sys.base_prefix,
                conda=bool((Path(sys.base_prefix)/'conda-meta').exists() or os.environ.get('CONDA_PREFIX')),
                versions=versions,missing=missing)


def probe(python,root,env):
    run=subprocess.run([str(python),str(Path(root)/'setup/runtime_manager.py'),'--probe'],
                       env=env,cwd=root,capture_output=True,text=True,encoding='utf-8')
    if run.returncode:raise RuntimeError('venvの検証に失敗しました。\n'+run.stderr[-3000:])
    return json.loads(run.stdout.strip().splitlines()[-1])


@contextmanager
def setup_lock(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as stream:
        if os.name=='nt':
            import msvcrt
            stream.seek(0);stream.write(b'0');stream.flush();stream.seek(0)
            try:msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            except OSError:raise RuntimeError('別のセットアップが実行中です。終了後に起動してください。')
        else:
            import fcntl
            try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise RuntimeError('別のセットアップが実行中です。終了後に起動してください。')
        try:yield
        finally:
            if os.name=='nt':stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(stream,fcntl.LOCK_UN)


def ensure_environment(uv,root=ROOT):
    root=Path(root).resolve();runtime=root/'.runtime';config=json.loads((root/'setup/runtime-lock.json').read_text())
    version=config['python_version'];env=clean_environment(root)
    if platform.python_version()!=version:raise RuntimeError(f'Python {version} が必要です。OS別の起動ファイルから実行してください。')
    if (Path(sys.base_prefix)/'conda-meta').exists() or not Path(sys.base_prefix).resolve().is_relative_to((runtime/'python').resolve()):
        raise RuntimeError('Condaまたは外部Pythonが選択されました。専用Pythonから実行してください。')
    configure_tk(env,sys.base_prefix)
    requirements=root/'setup/requirements.lock'
    fingerprint=hashlib.sha256(requirements.read_bytes()).hexdigest()[:12]
    directory=runtime/'venvs'/f'python-{version}-{fingerprint}'
    python=directory/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if python.exists():
        report=probe(python,root,env)
        if report['python']!=version or report['conda'] or not Path(report['base_prefix']).resolve().is_relative_to((runtime/'python').resolve()):
            directory=directory.with_name(directory.name+'-'+uuid.uuid4().hex[:8])
            python=directory/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    elif directory.exists():
        directory=directory.with_name(directory.name+'-'+uuid.uuid4().hex[:8])
        python=directory/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if not python.exists():
        print('Creating dedicated venv: '+str(directory),flush=True)
        subprocess.run([str(uv),'venv','--python',sys.executable,str(directory)],check=True,env=env,cwd=root)
    report=probe(python,root,env)
    if report['missing']:
        print('Installing/synchronizing all locked packages (including dependencies)…',flush=True)
        subprocess.run([str(uv),'pip','sync','--python',str(python),'--require-hashes',str(requirements)],check=True,env=env,cwd=root)
    subprocess.run([str(uv),'pip','check','--python',str(python)],check=True,env=env,cwd=root)
    report=probe(python,root,env)
    if report['missing'] or report['conda']:raise RuntimeError('Package/Conda validation failed: '+json.dumps(report,ensure_ascii=False))
    env['VIRTUAL_ENV']=str(directory);env['PATH']=str(python.parent)+os.pathsep+env['PATH']
    return python,env,report


def health_check(report,root=ROOT):
    import cv2
    import numpy as np
    import tkinter as tk
    report.update(checked_utc=datetime.now(timezone.utc).isoformat(),platform=platform.platform(),
                  architecture=platform.machine(),conda_inherited=os.environ.get('FLY_CONDA_DETECTED','0'))
    tkroot=tk.Tk();tkroot.withdraw()
    sys.path.insert(0,str(Path(root)/'fly_activity'))
    from tk_images import photo_image
    from PIL import Image
    picture=photo_image(Image.new('RGB',(2,2),(10,20,30)),master=tkroot)
    if picture.get(0,0)!=(10,20,30):raise RuntimeError('Tk画像表示の検証に失敗しました。')
    tkroot.update_idletasks();tkroot.destroy()
    report['tk_image_rendering']='ok'
    report['tkinter']='ok'
    runtime=Path(root)/'.runtime';runtime.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='video_check_',dir=runtime) as temporary:
        path=Path(temporary)/'確認動画.mp4'
        writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),30,(64,64))
        report['mp4_writer_opened']=writer.isOpened()
        if writer.isOpened():
            report['video_backend']=writer.getBackendName()
            for _ in range(3):writer.write(np.zeros((64,64,3),np.uint8))
        writer.release()
        cap=cv2.VideoCapture(str(path));report['mp4_readable']=bool(cap.read()[0]);cap.release()
    report['opencv_video_build']=[line.strip() for line in cv2.getBuildInformation().splitlines() if any(k in line for k in ('Video I/O:','FFMPEG:','avcodec:','AVFoundation:'))]
    (runtime/'diagnostics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    if not report['mp4_writer_opened'] or not report['mp4_readable']:
        raise RuntimeError('確認動画のMP4保存・再読込みテストに失敗しました。.runtime/diagnostics.json を確認してください。')
    print('Validated: Python '+report['python']+' / Conda inactive / Tkinter / all pinned packages / MP4 write-read',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--uv',type=Path);parser.add_argument('--app',choices=(*APPS,'check'),default='manual')
    parser.add_argument('--probe',action='store_true');parser.add_argument('--compile-lock',action='store_true');parser.add_argument('--verified',action='store_true')
    args,extra=parser.parse_known_args()
    if args.probe:print(json.dumps(dependency_report()));return 0
    env=clean_environment()
    if args.compile_lock:
        subprocess.run([str(args.uv),'pip','compile','--universal','--generate-hashes',str(ROOT/'setup/requirements.in'),'-o',str(ROOT/'setup/requirements.lock')],check=True,env=env,cwd=ROOT);return 0
    if args.verified:
        report=dependency_report()
        required=json.loads((ROOT/'setup/runtime-lock.json').read_text())['python_version']
        if report['missing'] or report['conda'] or report['python']!=required or not Path(sys.prefix).resolve().is_relative_to((ROOT/'.runtime/venvs').resolve()):
            raise RuntimeError('The dedicated runtime no longer matches the lock.')
        try:health_check(report)
        except Exception as error:
            report['error']=str(error)
            (ROOT/'.runtime/diagnostics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            raise
        if args.app=='check':return 0
        return subprocess.call([sys.executable,str(ROOT/'fly_activity'/APPS[args.app]),*extra],env=os.environ.copy(),cwd=ROOT)
    if not args.uv:raise RuntimeError('OS別の起動ファイルから実行してください。')
    with setup_lock(ROOT/'.runtime/setup.lock'):
        python,env,_=ensure_environment(args.uv)
    return subprocess.call([str(python),str(Path(__file__).resolve()),'--verified','--app',args.app,*extra],env=env,cwd=ROOT)


if __name__=='__main__':
    try:sys.exit(main())
    except Exception as error:
        print('Setup/startup failed: '+str(error),file=sys.stderr);sys.exit(1)
