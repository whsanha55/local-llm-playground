@echo off
chcp 65001 >nul
rem gemma4 채팅 서버(API+웹UI) 한방에 실행 — Ollama 백엔드
rem 사용: start.bat [포트]   (기본 8300, 모델은 MODEL_ID env로 변경 가능)
setlocal
cd /d "%~dp0"
set "PORT=%1"
if "%PORT%"=="" set "PORT=8300"
if "%MODEL_ID%"=="" set "MODEL_ID=gemma4-srt:latest"

where python >nul 2>nul || (echo python이 없습니다 - PATH를 확인하세요 & exit /b 1)

python -c "import fastapi, uvicorn, multipart" >nul 2>nul || (
  echo 의존성 설치 중...
  python -m pip install -r requirements.txt || exit /b 1
)

ollama list >nul 2>nul || (
  echo Ollama가 실행 중이 아닙니다 - Ollama 앱 또는 "ollama serve"를 먼저 띄워주세요.
  exit /b 1
)

rem 서버 뜨면 브라우저로 열기 (모델 프리로드에 시간이 걸릴 수 있음 - 실패하면 새로고침)
start "" /b cmd /c "timeout /t 8 /nobreak >nul & start http://127.0.0.1:%PORT%/"

echo 기동 중 (기본 모델: %MODEL_ID%) - http://127.0.0.1:%PORT%/  (종료: Ctrl+C)
echo 같은 와이파이 기기 접속: http://^(이 PC의 IP^):%PORT%/
python -m uvicorn gemma4_api:app --host 0.0.0.0 --port %PORT%
