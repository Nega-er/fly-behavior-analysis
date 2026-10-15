# ハエの行動解析・人手記録アプリ

動画から複数のハエを検出・追跡し、移動と前進・後退を解析するアプリと、個体ごとの行動を人の手で1秒ずつ記録するアプリです。

## 起動（管理者権限・既存Python・Condaは不要）

ZIPを展開し、自分が書き込めるフォルダで使ってください。Mac用とWindows用を分けています。

| OS | 起動するもの |
|---|---|
| Mac | **`FlyBehavior.app`** をダブルクリックし、「個体行動の手入力」「自動解析」「環境確認のみ」から選択 |
| Windows | **`Windows_個体行動の手入力.bat`** または **`Windows_自動解析.bat`** をダブルクリック |
| Windowsの診断のみ | `Windows_環境確認.bat` |

`.command` やCondaのターミナルから起動する必要はありません。初回はインターネットに接続してください。
Macでダウンロードした未署名アプリの確認が出た場合は、OSの「開く」操作で許可してください。このアプリは署名・公証済みではありません。

起動処理は次を順に実行します。

1. 継承されたConda・別venv・PYTHONPATH等を検出し、起動する子プロセスから除外。
2. SHA256を確認した固定版uvと、アプリ専用の **CPython 3.12.10** をローカルに用意。
3. 専用venvを作成し、Mac・Windows共通のハッシュ付きロックファイルから依存パッケージをインストール。
4. バージョン・依存関係・Conda非使用・Tkinter・MP4書き出しと再読込みを確認。
5. 検証したvenvのPythonでアプリを起動。

不足・異なる版・importに失敗するパッケージがあれば、専用venv内で再同期します。二度目以降は作成済みの環境を再利用します。
`sudo`、管理者として実行、システムPythonやCondaへのpipインストール、シェルの初期設定の書換えは行いません。
別のターミナルで動いているConda自体を終了・変更することもありません。

ランタイム・venv・キャッシュはこのフォルダの `.runtime/`、学習データは従来どおり `fly_activity/learning/` に置きます。
以前の `.venv` は削除・上書きせず、そのまま保持します。

固定版は `setup/runtime-lock.json` と `setup/requirements.in`、全依存の版と配布ファイルのハッシュは `setup/requirements.lock` で管理します。
NumPy 1.26.4 / OpenCV 4.10.0.84 / SciPy 1.11.4 / Matplotlib 3.10.6 / Pillow 11.3.0を使用します。
MacはApple Silicon・Intel、Windowsは64bit Pythonを使う構成です。

## 起動に失敗したとき

「環境確認のみ」または `Windows_環境確認.bat` を実行し、次のファイルを確認してください。

- `.runtime/diagnostics.json`：実際のPython・パッケージ版・Conda分離・OpenCV・MP4確認結果。
- `.runtime/logs/setup-mac.log` または `setup-windows.log`：ダウンロード・インストール・アプリのエラー全文。

管理されたPythonはAstralの[python-build-standalone](https://github.com/astral-sh/python-build-standalone)、環境管理は[uv](https://docs.astral.sh/uv/)を使用します。
Python 3.12.10付属のTcl/Tkデータを指定し、venvからもGUIを読み込めるようにしています。

## 人手で行動を記録する

- 新しい動画を検出するか、既存の `tracks.csv` を読み込みます。
- プルダウンで個体を選び、1秒区間の全体映像と拡大映像を確認します。
- 区間は自動で繰り返し再生され、行動を選ぶと保存して次の区間を自動再生します。
- グルーミング頭・グルーミング体・停止・前進・後退・飛んでいるを記録できます。Unknownと未入力は別に保持します。
- 自動保存・途中再開・修正履歴に対応し、CSVと行動時系列・合計時間のグラフを書き出せます。

[人手記録の詳しい操作説明](fly_activity/README_個体行動の手入力.md)

## 自動解析・人手学習

[自動解析と人手評価の操作説明](fly_activity/README_自動解析と精度評価.md)

接触や重なりの後まで、追跡番号が同じ個体に対応することを保証するものではありません。個体をまたぐ結論には確認動画を使ってください。

## データの保存と共有

アプリは `fly_activity`、累積教師・モデル履歴・最新版参照は `fly_activity/learning` にあります。
Mac・Windowsとも、この相対配置を使用します。累積学習データはこのリポジトリで共有します。
動画・生成画像・解析出力・`.venv` は公開対象に含めません。試す動画は各自で用意してください。
元動画は自動アップロードしません。

他PCで学習した場合は、作業前にGitHubの更新を取得し、既存の学習履歴を消さずに統合してください。
人手による行動記録は自動学習データへ登録しません。

## 開発・検証

```sh
# 開発者向け：構築済み専用venvのPythonで実行
python -m unittest discover -s fly_activity -v
python -m unittest discover -s . -p test_launch.py -v
```

画面操作を確認するテストがあるため、デスクトップ環境で実行してください。
旧 `.command` は互換用に残しています。`7個体移動解析.command` などは以前の実習動画専用の補助スクリプトです。公開していない動画・背景・解析結果を必要とするため、新しい動画には上記2アプリを使用してください。
`AMADEUS直接実行.command` は別途インストールしたAMADEUSと、その動画に合った設定が必要です。
