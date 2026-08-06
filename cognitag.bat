@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"

:MENU
cls
echo ==========================================
echo   CogniTag v2  辞書構築パイプライン
echo ==========================================
echo.
echo   [ 準備 ]
echo     0. 初期セットアップ (必要ライブラリの導入)
echo     9. 環境チェック (Python / Sudachi / Ollama)
echo.
echo   [ 検証 ]
echo     C. 較正テスト (プロンプト妥当性の確認 / 約5分)
echo     R. 隔離語の救済 (1回目の試行を除外して再判定)
echo.
echo   [ お試し実行 ]
echo     1. コーパス生成   (5カテゴリだけ)
echo     2. 候補語抽出
echo     3. Facet注釈      (30語だけ)
echo.
echo   [ 本番実行 ]
echo     4. コーパス生成   (全295カテゴリ / 約2時間)
echo     5. Facet注釈      (全候補語 / 長時間)
echo        ※4の後は必ず 2 を実行してから 5 へ
echo     6. 一括実行       (4 -^> 2 -^> 5 を連続)
echo.
echo   [ 確認 ]
echo     7. 現状レポート
echo     8. 生成物フォルダを開く
echo.
echo     Q. 終了
echo.
set "SEL="
set /p SEL="番号を入力してEnter: "

if /i "%SEL%"=="C" goto CALIBRATE
if /i "%SEL%"=="R" goto RESCUE
if /i "%SEL%"=="0" goto SETUP
if /i "%SEL%"=="9" goto CHECK
if /i "%SEL%"=="1" goto CORPUS_TEST
if /i "%SEL%"=="2" goto EXTRACT
if /i "%SEL%"=="3" goto ANNOTATE_TEST
if /i "%SEL%"=="4" goto CORPUS_FULL
if /i "%SEL%"=="5" goto ANNOTATE_FULL
if /i "%SEL%"=="6" goto RUN_ALL
if /i "%SEL%"=="7" goto STATS
if /i "%SEL%"=="8" goto OPENDIR
if /i "%SEL%"=="Q" goto END
goto MENU

:SETUP
cls
echo === 必要ライブラリを導入します ===
echo.
python -m pip install sudachipy sudachidict_core ollama
echo.
echo 導入処理が終わりました。
pause
goto MENU

:CHECK
cls
echo === 環境チェック ===
echo.
echo [Python]
python --version
if errorlevel 1 (
  echo   Python が見つかりません。PATH を確認してください。
  pause
  goto MENU
)
echo.
echo [SudachiPy]
python -c "from sudachipy import Dictionary; d=Dictionary().create(); print('  OK ->', d.tokenize(chr(30690)+chr(30462))[0].dictionary_form())" 2>nul
if errorlevel 1 echo   未導入です。メニュー 0 を実行してください。
echo.
echo [Ollama モデル一覧]
ollama list
if errorlevel 1 echo   ollama コマンドが見つかりません。
echo.
echo [GPU 使用状況] ※生成中に確認すると有効です
ollama ps
echo.
pause
goto MENU

:CALIBRATE
cls
echo === 較正テスト ===
echo.
echo facets.py のアンカー語(各軸15語)を正解ラベルとして、
echo 4軸が正しく分離できているかを測定します。
echo 本番注釈の前に必ず実行してください。
echo.
set "MDL=gemma3:4b"
set /p MDL="使用モデル [既定 gemma3:4b]: "
echo.
python run_pipeline.py calibrate --model %MDL%
echo.
pause
goto MENU

:RESCUE
cls
echo === 隔離語の救済 (1回目の試行を除外して再判定) ===
echo.
echo quarantine.json に保存済みの試行データから、1回目を除いた
echo 分散を再計算し、閾値以下になった語を辞書へ移します。
echo LLM への再問い合わせは行いません。
echo.
set "THR=0.5"
set /p THR="分散閾値 [既定 0.5]: "
echo.
echo --- まず dry-run で結果だけ確認します ---
echo.
python run_pipeline.py rescue --variance-threshold %THR% --dry-run
echo.
set "YN=N"
set /p YN="実際に反映しますか? (Y/N) [既定 N]: "
if /i not "%YN%"=="Y" goto MENU
echo.
python run_pipeline.py rescue --variance-threshold %THR%
echo.
pause
goto MENU

:CORPUS_TEST
cls
echo === コーパス生成 (5カテゴリのお試し) ===
echo.
set "MDL=gemma3:4b"
set /p MDL="使用モデル [既定 gemma3:4b]: "
echo.
python run_pipeline.py corpus --limit 5 --model %MDL%
echo.
echo 生成された corpus フォルダの中身を目視で確認してください。
echo 対話が自然か / 専門用語が文中に出ているかが判断ポイントです。
pause
goto MENU

:EXTRACT
cls
echo === 候補語の抽出と頻度フィルタ ===
echo.
set "MINF=2"
set /p MINF="最低出現回数 [既定 2]: "
echo.
python run_pipeline.py extract --min-frequency %MINF%
echo.
echo hapax率(出現1回語の割合)が高すぎる場合は
echo 最低出現回数を 3 に上げて再実行してください。
pause
goto MENU

:ANNOTATE_TEST
cls
echo === Facet注釈 (高頻度30語のお試し) ===
echo.
set "MDL=gemma3:4b"
set /p MDL="使用モデル [既定 gemma3:4b]: "
echo.
python run_pipeline.py annotate --limit 30 --model %MDL%
echo.
echo 最後に表示された「分散分布」の数値を控えてください。
echo 閾値決定の判断材料になります。
pause
goto MENU

:CORPUS_FULL
cls
echo === コーパス生成 (全295カテゴリ) ===
echo.
echo 注意: 数時間かかる場合があります。
echo       中断しても、同じ操作で続きから再開できます。
echo.
set "YN=N"
set /p YN="実行しますか? (Y/N) [既定 N]: "
if /i not "%YN%"=="Y" goto MENU
set "MDL=gemma3:4b"
set /p MDL="使用モデル [既定 gemma3:4b]: "
echo.
echo 開始時刻: %date% %time%
python run_pipeline.py corpus --model %MDL%
echo 終了時刻: %date% %time%
pause
goto MENU

:ANNOTATE_FULL
cls
echo === Facet注釈 (全候補語) ===
echo.
echo 確認: コーパス生成(4)の後に候補語抽出(2)を実行しましたか?
echo       していない場合、古い候補リストに対して注釈してしまいます。
echo.
set "THR=0.5"
set /p THR="分散閾値 [既定 0.5]: "
set "MDL=gemma3:4b"
set /p MDL="使用モデル [既定 gemma3:4b]: "
set "ITV=0"
set /p ITV="待機秒数 [PCなら0推奨/既定 0]: "
echo.
echo 開始時刻: %date% %time%
python run_pipeline.py annotate --variance-threshold %THR% --model %MDL% --interval %ITV%
echo 終了時刻: %date% %time%
pause
goto MENU

:RUN_ALL
cls
echo === 一括実行 (コーパス生成 -^> 抽出 -^> 注釈) ===
echo.
echo 注意: 全工程で数時間から半日かかる可能性があります。
echo       就寝前などの実行を推奨します。
echo.
set "YN=N"
set /p YN="実行しますか? (Y/N) [既定 N]: "
if /i not "%YN%"=="Y" goto MENU
set "MDL=gemma3:4b"
set /p MDL="使用モデル [既定 gemma3:4b]: "
set "THR=0.5"
set /p THR="分散閾値 [既定 0.5]: "
echo.
echo ############ 開始: %date% %time% ############
echo.
echo --- [1/3] コーパス生成 ---
python run_pipeline.py corpus --model %MDL%
echo.
echo --- [2/3] 候補語抽出 ---
python run_pipeline.py extract
echo.
echo --- [3/3] Facet注釈 ---
python run_pipeline.py annotate --variance-threshold %THR% --model %MDL% --interval 0
echo.
echo ############ 完了: %date% %time% ############
python run_pipeline.py stats
pause
goto MENU

:STATS
cls
python run_pipeline.py stats
echo.
pause
goto MENU

:OPENDIR
start "" "%~dp0"
goto MENU

:END
endlocal
exit /b 0