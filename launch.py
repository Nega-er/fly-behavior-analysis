#!/usr/bin/env python3
"""Compatibility entry: delegate to the OS-specific isolated runtime setup."""
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
APPS=('analysis','manual','analyze','annotate','check')


def command(app,extra=(),root=HERE,platform=None):
    if app not in APPS:raise ValueError('起動先を確認してください。')
    system=platform or sys.platform;root=Path(root)
    if system=='darwin':return ['/bin/zsh','-f',str(root/'setup/mac.sh'),'--app',app,*extra]
    if system=='win32':return ['powershell.exe','-NoLogo','-NoProfile','-ExecutionPolicy','Bypass','-File',str(root/'setup/windows.ps1'),'-App',app,*extra]
    raise RuntimeError('MacまたはWindows用の起動ファイルを使用してください。')


def main():
    try:return subprocess.call(command(sys.argv[1] if len(sys.argv)>1 else 'manual',sys.argv[2:]),cwd=HERE)
    except (ValueError,RuntimeError,OSError) as error:print(str(error),file=sys.stderr);return 1


if __name__=='__main__':sys.exit(main())
