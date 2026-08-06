@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ==========================================
echo  CogniTag v2 - Facet API サーバーを起動
echo ==========================================
echo.

if not exist "server.py" (
  echo [エラー] server.py が見つかりません。
  echo このバッチは CogniTag_V2 フォルダ直下に置いてください。
  pause
  exit /b 1
)

if not exist "dictionary_v2.json" (
  echo [警告] dictionary_v2.json が見つかりません。
  echo 辞書なしでも起動しますが、coverage は 0 になります。
  echo.
)

echo サーバーを起動します。ポート 8010 を使用します。
echo このウィンドウを閉じるとサーバーも停止します。
echo.
echo   API ドキュメント : http://127.0.0.1:8010/docs
echo   ヘルスチェック   : http://127.0.0.1:8010/api/health
echo.
echo 辞書を更新した後は、必ずこのサーバーを再起動してください。
echo 起動時に一度だけ辞書を読み込む設計のためです。
echo.

start "" http://127.0.0.1:8010/docs

uvicorn server:app --host 127.0.0.1 --port 8010 --reload

echo.
echo サーバーが停止しました。
pause