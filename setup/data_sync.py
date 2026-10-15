"""Startup synchronization restricted to saved experiment data, with conflict archives."""
from datetime import datetime,timezone
from contextlib import ExitStack,closing
import json
import os
from pathlib import Path
import sqlite3
import subprocess

from updater import find_git

GRAPHS={'report.png','direction_report.png','movement_rate.png','manual_behavior.png'}


def shared_path(name):
    path=Path(name)
    parts=path.parts
    if len(parts)<3 or parts[0]!='fly_activity':return False
    if any(p.startswith('.') for p in parts):return False
    if parts[1]=='learning':return path.suffix=='.json' or path.name=='examples.sqlite3'
    result=parts[1]=='outputs' or parts[1].startswith(('results_','run_','direction_confirmed_','direction_fixed_','direction_reanalysis_'))
    return result and (path.suffix in ('.csv','.json') or path.name in GRAPHS)


def synchronize_repository(root,python=None,app_env=None,git=None):
    root=Path(root).resolve();runtime=root/'.runtime';runtime.mkdir(exist_ok=True)
    result=dict(checked_utc=datetime.now(timezone.utc).isoformat(),status='skipped',detail='',uploaded=False)
    def finish(status,detail):
        result.update(status=status,detail=detail)
        print('Git sync: '+detail,flush=True)
        (runtime/'update_status.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        return result
    if not (root/'.git').exists():return finish('skipped','ZIP版は自動同期できません。Git clone版を使用してください。')
    git=git or find_git()
    if not git:return finish('skipped','Gitが見つかりません。ローカルで起動します。')
    env={k:v for k,v in os.environ.items() if not k.startswith('GIT_')}
    env.update(GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='Never',GIT_MERGE_AUTOEDIT='no')
    command=[git,'-c','credential.interactive=false','-c','merge.autoStash=false',
             '-c','rebase.autoStash=false','-c','user.name=Fly Behavior Sync',
             '-c','user.email=fly-behavior-sync@users.noreply.github.com','-C',str(root)]
    def run(*args,check=True,binary=False,timeout=120):
        p=subprocess.run([*command,*args],env=env,capture_output=True,timeout=timeout,
                         **({} if binary else dict(text=True,encoding='utf-8',errors='replace')))
        if check and p.returncode:raise RuntimeError((p.stderr+p.stdout).strip()[-3000:])
        return p
    def commit_data(message):
        names=run('ls-files','-m','-o','--exclude-standard','-z').stdout.split('\0')
        candidates=sorted({s for s in names if s and shared_path(s) and (root/s).is_file() and not (root/s).is_symlink()})
        oversized=[s for s in candidates if (root/s).stat().st_size>=95*1024*1024]
        if oversized:raise RuntimeError('GitHubの上限に近いファイルは送信できません：'+', '.join(oversized))
        with ExitStack() as stack:
            for name in candidates:
                if Path(name).name!='examples.sqlite3':continue
                db=stack.enter_context(closing(sqlite3.connect(root/name,timeout=10)))
                if db.execute('PRAGMA wal_checkpoint(FULL)').fetchone()[0]:raise RuntimeError('学習DBが使用中です。')
                db.execute('BEGIN IMMEDIATE')
            try:
                for start in range(0,len(candidates),80):run('add','--',*candidates[start:start+80])
                if run('diff','--cached','--quiet',check=False).returncode:run('commit','-m',message)
            except Exception:
                for start in range(0,len(candidates),80):run('restore','--staged','--',*candidates[start:start+80],check=False)
                raise

    def archive_remote(remote,path):
        blob=run('show',f'{remote}:{path}',check=False,binary=True)
        if blob.returncode:return
        parts=Path(path).parts
        if parts[1]=='learning':target=root/'fly_activity/learning/imports'/remote/Path(*parts[2:])
        else:target=root/'fly_activity/outputs/shared_conflicts'/remote/Path(path)
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and target.read_bytes()!=blob.stdout:raise RuntimeError('既存の競合履歴と一致しません：'+str(target))
        if not target.exists():target.write_bytes(blob.stdout)

    merging=False
    try:
        top=run('rev-parse','--show-toplevel').stdout.strip()
        if Path(top).resolve()!=root:return finish('skipped','アプリ自身のGitリポジトリではありません。')
        if run('symbolic-ref','--quiet','HEAD',check=False).returncode:return finish('skipped','ブランチ未選択のため同期を省略します。')
        upstream=run('rev-parse','--abbrev-ref','--symbolic-full-name','@{upstream}',check=False)
        if upstream.returncode:return finish('skipped','追跡先がありません。')
        # Do not include the user's staged work or interfere with an ongoing merge.
        if run('diff','--cached','--quiet',check=False).returncode or run('ls-files','-u').stdout:
            return finish('skipped','ステージ済みの変更・解決中の競合があるため同期を省略します。')
        result['previous_head']=run('rev-parse','HEAD').stdout.strip()
        before_remote=run('rev-parse','@{upstream}').stdout.strip()
        commit_data('Save fly analysis, manual annotations and learning data')
        remote_name=run('config','--get',f'branch.{run("branch","--show-current").stdout.strip()}.remote').stdout.strip()
        run('fetch',remote_name)
        remote=run('rev-parse','@{upstream}').stdout.strip()
        merge=run('merge','--no-edit','--no-commit','@{upstream}',check=False)
        if merge.returncode:
            conflicts=run('diff','--name-only','--diff-filter=U','-z').stdout.split('\0')
            conflicts=[p for p in conflicts if p]
            merging=bool(conflicts)
            if not conflicts or any(not shared_path(p) for p in conflicts):raise RuntimeError(merge.stderr+merge.stdout)
            for path in conflicts:
                if run('cat-file','-e','HEAD:'+path,check=False).returncode:raise RuntimeError('削除を含む競合は手動確認が必要です：'+path)
            if any(p.startswith('fly_activity/learning/') for p in conflicts):
                for name in ('examples.sqlite3','current.json','seed_model.json'):
                    archive_remote(remote,'fly_activity/learning/'+name)
            for path in conflicts:
                archive_remote(remote,path)
                # Both committed histories are retained; keep the local active file.
                run('checkout','--ours','--',path);run('add','--',path)
        merging=bool(run('rev-parse','-q','--verify','MERGE_HEAD',check=False).returncode==0)
        if merging:
            run('commit','-m','Merge shared fly data; preserve conflicting remote histories');merging=False
        result['restart_required']=before_remote!=remote or run('merge-base','--is-ancestor',remote,result['previous_head'],check=False).returncode!=0
        if python and (root/'setup/import_shared_learning.py').exists():
            imported=subprocess.run([str(python),str(root/'setup/import_shared_learning.py')],cwd=root,env=app_env,timeout=120,capture_output=True,text=True,encoding='utf-8')
            if imported.returncode:raise RuntimeError(imported.stderr[-3000:])
            if imported.stdout:print(imported.stdout,flush=True)
        commit_data('Import shared fly learning history')
        run('push',remote_name,'HEAD:'+run('config','--get',f'branch.{run("branch","--show-current").stdout.strip()}.merge').stdout.strip())
        result['uploaded']=True;result['current_head']=run('rev-parse','HEAD').stdout.strip()
        # Restart only when incoming history changed, not for a local data commit.
        incoming=result['restart_required']
        return finish('updated' if incoming else 'current','解析結果・手入力・学習データを同期しました。')
    except (RuntimeError,OSError,sqlite3.Error,subprocess.TimeoutExpired) as error:
        if merging:
            try:run('merge','--abort')
            except Exception:pass
        return finish('failed','同期できませんでした。ローカルデータとコミットを保持して起動します。 '+str(error))
