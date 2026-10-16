"""GUI-requested fetch/pull, followed by saved-data commit/push."""
import argparse
import json
import os
from pathlib import Path
import sys

from data_sync import synchronize_repository
from runtime_manager import setup_lock


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent.parent)
    args=parser.parse_args()
    def report(stage,detail):
        print('FLY_SYNC_PROGRESS '+json.dumps(dict(stage=stage,detail=detail),ensure_ascii=False),flush=True)
    try:
        with setup_lock(args.root/'.runtime/setup.lock'):
            result=synchronize_repository(args.root,python=sys.executable,app_env=os.environ.copy(),
                                          pull_first=True,progress=report)
    except (RuntimeError,OSError) as error:
        result=dict(status='failed',uploaded=False,detail='同期を開始できません。別のアプリの同期・起動処理が終わってから再試行してください。 '+str(error))
    print('FLY_SYNC_RESULT '+json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
