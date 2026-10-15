#!/bin/zsh
cd "${0:A:h}" || exit 1
fly_python="${FLY_ACTIVITY_PYTHON:-}"
if [[ -z "$fly_python" ]]; then
  if [[ -x .venv/bin/python ]]; then
    fly_python="$PWD/.venv/bin/python"
  elif [[ -x "$HOME/Applications/AMADEUS/.venv/bin/python" ]]; then
    fly_python="$HOME/Applications/AMADEUS/.venv/bin/python"
  else
    fly_python=python3
  fi
fi
fly_candidates=(fly_activity/head_anchors_*.csv(Nom))
if (( ${#fly_candidates} == 0 )); then
  print '頭側入力がありません。「前進後退の頭側確認.command」を先に実行してください。'
  read '?Enterキーで閉じます。'
  exit 1
fi
fly_heads="$fly_candidates[1]"
fly_head_tracks=$("$fly_python" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]+".json"); print(json.loads(p.read_text())["tracks_source"] if p.exists() else "fly_activity/results_direction_px/tracks.csv")' "$fly_heads")
fly_output="fly_activity/direction_reanalysis_$(date +%Y%m%d_%H%M%S)"
"$fly_python" launch.py analyze \
  --tracks fly_activity/results_retracked_px/tracks.csv \
  --config fly_activity/example_config.json \
  --head-anchors "$fly_heads" \
  --head-anchor-tracks "$fly_head_tracks" \
  --direction-video 'R8実習用_5-10sON_MDN-Chrimson.mp4' \
  --out "$fly_output"
fly_status=$?
if [[ $fly_status -eq 0 ]]; then
  open "$fly_output/direction_report.png"
fi
print "終了コード: $fly_status / 頭側入力: $fly_heads / 結果: $fly_output"
read '?Enterキーで閉じます。'
exit "$fly_status"
