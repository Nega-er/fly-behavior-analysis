"""Run with the verified venv, after downloading and archiving another PC's history."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'fly_activity'))
from cumulative_learning import LearningStore


def main():
    store=ROOT/'fly_activity/learning'
    histories=sorted((store/'imports').glob('*/examples.sqlite3'))
    if not histories:return
    learner=LearningStore(store)
    for source in histories:
        count=learner.import_history(source)
        print(f'Imported shared teachers: {count} ({source.parent.name})',flush=True)


if __name__=='__main__':main()
