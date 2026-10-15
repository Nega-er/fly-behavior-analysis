"""Parse explicit, stable display IDs without renumbering remaining flies."""
import re


def parse_excluded_slots(text):
    text=text.strip()
    if not text:return []
    parts=re.split(r'[,、，\s]+',text)
    if any(not p.isdecimal() or int(p)<1 for p in parts):
        raise ValueError('除外IDは1以上の整数をカンマで区切ってください（例：2,5）。')
    return sorted({int(p) for p in parts})
