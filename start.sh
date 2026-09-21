#!/bin/bash
# gemma-4 채팅 서버(API+웹UI) 한방에 실행
# 사용: start.sh   (모델: supergemma 고정, 포트는 PORT env로 변경 가능)
# 이미 실행 중이면 강제로 끄고 다시 시작한다.
set -eu
DIR="$(cd "$(dirname "$0")" && pwd)"
PY="$HOME/.local/share/uv/tools/mlx-lm/bin/python"
PORT="${PORT:-8300}"

[ -x "$PY" ] || { echo "mlx-lm 툴 파이썬이 없습니다: $PY" >&2; exit 1; }
[ -f "$DIR/gemma4_api.py" ] || { echo "gemma4_api.py가 없습니다: $DIR" >&2; exit 1; }

# 실행 중 인스턴스 종료 (graceful → 5초 후 강제)
if lsof -ti tcp:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  OLD_PID=$(lsof -ti tcp:"$PORT" -sTCP:LISTEN)
  echo "실행 중 인스턴스 종료 (PID: $OLD_PID)"
  kill $OLD_PID 2>/dev/null || true
  for _ in $(seq 1 5); do
    lsof -ti tcp:"$PORT" -sTCP:LISTEN >/dev/null 2>&1 || break
    sleep 1
  done
  if lsof -ti tcp:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "강제 종료 (kill -9)"
    kill -9 $(lsof -ti tcp:"$PORT" -sTCP:LISTEN) 2>/dev/null || true
    sleep 1
  fi
fi

case "${1:-supergemma}" in
  supergemma) MODEL_ID=supergemma-abliterated ;;
  *) echo "사용법: $0" >&2; exit 1 ;;
esac
export MODEL_ID

# 서버 준비되면 브라우저로 열기 (모델 프리로드에 수십 초 걸릴 수 있음)
( for _ in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { open "http://127.0.0.1:$PORT/"; exit 0; }
    sleep 1
  done ) &

echo "기동 중 (기본 모델: $MODEL_ID) — http://127.0.0.1:$PORT/  (종료: Ctrl+C)"
exec "$PY" -m uvicorn gemma4_api:app --host 127.0.0.1 --port "$PORT" --app-dir "$DIR"
