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
fly_stamp=$(date +%Y%m%d_%H%M%S)
fly_heads="fly_activity/head_anchors_${fly_stamp}.csv"
fly_output="fly_activity/direction_confirmed_${fly_stamp}"
"$fly_python" launch.py annotate \
  --video 'R8実習用_5-10sON_MDN-Chrimson.mp4' \
  --tracks fly_activity/results_retracked_px/tracks.csv --out "$fly_heads"
fly_status=$?
if [[ $fly_status -eq 0 ]]; then
  "$fly_python" launch.py analyze \
    --tracks fly_activity/results_retracked_px/tracks.csv \
    --direction-video 'R8実習用_5-10sON_MDN-Chrimson.mp4' \
    --config fly_activity/example_config.json --head-anchors "$fly_heads" --out "$fly_output"
  fly_status=$?
  if [[ $fly_status -eq 0 ]]; then
    open "$fly_output/direction_report.png"
  fi
fi
print "終了コード: $fly_status"
read '?Enterキーで閉じます。'
exit "$fly_status"
