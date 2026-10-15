"""Best-effort startup pull. Never stash, reset, merge or switch branches."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def find_git():
    candidates=[]
    if sys.platform=='darwin':
        candidates.extend(['/opt/homebrew/bin/git','/usr/local/bin/git',
                           '/Applications/GitHub Desktop.app/Contents/Resources/app/git/bin/git',
                           str(Path.home()/'Applications/GitHub Desktop.app/Contents/Resources/app/git/bin/git')])
        # The system Git stub can open an installer when Command Line Tools are absent.
        try:
            if subprocess.run(['/usr/bin/xcode-select','-p'],capture_output=True,timeout=5).returncode==0:
                candidates.append('/usr/bin/git')
        except (OSError,subprocess.TimeoutExpired):pass
    elif os.name=='nt':
        for base,relative in (('ProgramFiles','Git/cmd/git.exe'),
                              ('ProgramFiles(x86)','Git/cmd/git.exe'),
                              ('LOCALAPPDATA','Programs/Git/cmd/git.exe'),
                              ('USERPROFILE','scoop/apps/git/current/cmd/git.exe')):
            if os.environ.get(base):candidates.append(str(Path(os.environ[base])/relative))
        candidates.append(shutil.which('git'))
        if os.environ.get('LOCALAPPDATA'):
            desktop=Path(os.environ['LOCALAPPDATA'])/'GitHubDesktop'
            candidates.extend(str(p) for p in sorted(desktop.glob('app-*/resources/app/git/cmd/git.exe'),reverse=True))
    else:candidates.append(shutil.which('git'))
    return next((str(p) for p in candidates if p and Path(p).is_file()),None)


def update_repository(root,git=None):
    root=Path(root).resolve()
    result=dict(checked_utc=datetime.now(timezone.utc).isoformat(),status='skipped',detail='')

    def finish(status,detail):
        result.update(status=status,detail=detail)
        print('Git update: '+detail,flush=True)
        try:
            directory=root/'.runtime';directory.mkdir(parents=True,exist_ok=True)
            temporary=directory/'update_status.json.tmp'
            temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
            temporary.replace(directory/'update_status.json')
        except OSError as error:print('Git update status: '+str(error),flush=True)
        return result

    if not (root/'.git').exists():
        return finish('skipped','Git管理情報がありません。ZIP版の自動更新にはGit cloneでの取得が必要です。')
    git=git or find_git()
    if not git:return finish('skipped','Gitが見つかりません。現在の版で起動します。')
    env=os.environ.copy()
    # Do not let an inherited Git environment select a different repository or index.
    for key in list(env):
        if key.startswith('GIT_'):env.pop(key,None)
    env.update(GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='Never',GIT_MERGE_AUTOEDIT='no')
    command=[git,'-c','credential.interactive=false','-c','pull.rebase=false',
             '-c','rebase.autoStash=false','-c','merge.autoStash=false',
             '-c','http.lowSpeedLimit=1000','-c','http.lowSpeedTime=15','-C',str(root)]

    def run(*args):
        return subprocess.run([*command,*args],env=env,capture_output=True,text=True,
                              encoding='utf-8',errors='replace',timeout=30)

    try:
        top=run('rev-parse','--show-toplevel')
        if top.returncode or Path(top.stdout.strip()).resolve()!=root:
            return finish('skipped','アプリ自身のGitリポジトリを確認できません。現在の版で起動します。')
        branch=run('symbolic-ref','--quiet','--short','HEAD')
        upstream=run('rev-parse','--abbrev-ref','--symbolic-full-name','@{upstream}')
        if branch.returncode or upstream.returncode:
            return finish('skipped','現在のブランチに追跡先がありません。現在の版で起動します。')
        result['previous_head']=run('rev-parse','HEAD').stdout.strip()
        pull=run('pull','--ff-only','--no-rebase','--no-autostash')
        result['current_head']=run('rev-parse','HEAD').stdout.strip()
        if pull.returncode:
            result['git_output']=(pull.stdout+'\n'+pull.stderr).strip()[-4000:]
            return finish('failed','更新できませんでした。手元の変更を保持して現在の版で起動します。\n'+result['git_output'])
        if result['current_head']!=result['previous_head']:
            return finish('updated','GitHubの更新を取得しました。更新後の起動処理を読み直します。')
        return finish('current','最新版です。')
    except (OSError,subprocess.TimeoutExpired) as error:
        return finish('failed','更新の確認に失敗しました。現在の版で起動します。 '+str(error))
