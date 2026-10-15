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
"$fly_python" launch.py analysis "$@"
fly_status=$?
if [[ $fly_status -ne 0 ]]; then
  print "終了コード: $fly_status"
  read '?Enterキーで閉じます。'
fi
exit "$fly_status"
