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
fly_output="fly_activity/run_$(date +%Y%m%d_%H%M%S)"
"$fly_python" launch.py analyze \
  --video 'R8実習用_5-10sON_MDN-Chrimson.mp4' \
  --config fly_activity/example_config.json \
  --out "$fly_output"
fly_status=$?
if [[ $fly_status -eq 0 ]]; then
  open "$fly_output/report.png"
fi
print "終了コード: $fly_status / 出力先: $fly_output"
read '?Enterキーで閉じます。'
exit "$fly_status"
