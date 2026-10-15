#!/bin/zsh
cd /Users/ruvi/Applications/AMADEUS || exit 1
print 'AMADEUSを修正済みの設定で実行します。GUI側の処理は停止してから使用してください。'
/Users/ruvi/Applications/AMADEUS/.venv/bin/python -u \
  /Users/ruvi/Applications/AMADEUS/main/batch.py \
  /Users/ruvi/repo_tex/Exp_3A/amadeus_R8実習用_5-10sON_MDN-Chrimson/config_direct.yaml
result=$?
print "終了コード: $result"
read '?Enterキーで閉じます。'
exit "$result"
