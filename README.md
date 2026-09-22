# local-llm-playground

[Ollama](https://ollama.com)로 로컬 LLM을 구동하는 채팅 서버 — 윈도우/맥 공용.
FastAPI 백엔드 + 싱글파일 웹 UI + 웹검색 MCP 연동으로, 외부 API 키 없이 모든 추론을 내 머신에서만 실행합니다.
추론은 Ollama 서버(REST)에 맡기고, 모델 목록은 설치된 것에서 동적으로 읽습니다.

## 주요 기능

- **로컬 채팅** — Ollama로 Gemma 모델 추론, NDJSON 토큰 스트리밍 + tok/s 표시
- **추론(thinking) 채널** — 추론 과정을 별도 접이식 블록으로 분리 표시, 토글로 on/off
- **URL 요약** — 웹페이지 본문을 추출해 로컬 모델로 요약
- **웹 검색** — 검색 결과 페이지들을 읽어 로컬 모델로 종합 답변 생성
- **원커맨드 실행** — `start.bat`(윈도우) / `./start.sh`(맥) 한 번으로 서버 기동 + 모델 프리로드 + 브라우저 오픈

## 기본 모델

`gemma4-srt:latest` — SRT 자막 번역에 튜닝된 Gemma-4 E4B(Q4_K_M). 채팅·번역 모두 이 모델로 돌아가며, `MODEL_ID` env 또는 웹 UI 드롭다운으로 설치된 다른 Ollama 모델로 교체할 수 있습니다.

## 요구사항

- Windows / macOS / Linux
- Python ≥ 3.10 + [Ollama](https://ollama.com) 실행 중(기본 모델 설치: `ollama pull gemma4-srt:latest`)
- Node.js ≥ 20 (web-search-mcp 빌드용 — 없으면 요약·검색 기능만 비활성)
- 인터넷 연결 (검색·요약 기능)

## 설치

```bash
# 1) 파이썬 의존성
pip install -r requirements.txt

# 2) web-search-mcp — 웹검색·본문추출용 MCP 서버 (mrkrsl/web-search-mcp, MIT)
git clone https://github.com/mrkrsl/web-search-mcp
cd web-search-mcp
npm install
npx playwright install
npm run build
cd ..
```

## 실행

윈도우:

```bat
start.bat            # 기본 포트 8300, 의존성 자동 설치 + 브라우저 오픈
start.bat 9000       # 포트 변경
```

맥/리눅스:

```bash
./start.sh           # 기본 포트 8300, 서버 준비되면 브라우저 자동 오픈
PORT=9000 ./start.sh # 포트 변경
```

수동 실행:

```bash
python -m uvicorn gemma4_api:app --host 0.0.0.0 --port 8300
```

## API

| 엔드포인트 | 설명 |
|---|---|
| `GET /` | 채팅 웹 UI |
| `GET /models` | 선택 가능 모델 목록 |
| `GET /health` | 헬스체크 |
| `POST /chat` | 완결 응답 `{messages, model, enable_thinking, max_tokens}` |
| `POST /chat/stream` | NDJSON 스트리밍 `{"t": 토큰}... {"done": true, "seconds", "chunks"}` |
| `POST /summarize/stream` | URL 본문 추출 → 요약 스트리밍. `{"phase":"fetch"}` 후 토큰 스트리밍 |
| `POST /search/stream` | 웹 검색 → 종합 답변 스트리밍. `{"phase":"search"}` 후 토큰 스트리밍 |
| `POST /translate` | `.srt`/`.txt` 업로드 → 번역 작업 등록(202 `{job_id}`). 대기목록에 쌓여 순차 실행, 청크 병렬. SRT 구조 유지 |
| `GET /translate` | 번역 전용 웹 UI(업로드·대시보드·작업 리스트·조회/취소/삭제) |
| `GET /translate/jobs` | 번역 작업 목록(상태·진행률, 대기순번) |
| `GET /translate/jobs/{id}` | 작업 상태 조회. 완료 시 번역 결과 포함 |
| `POST /translate/jobs/{id}/cancel` | 작업 취소. 대기 중은 즉시, 실행 중은 청크 경계에서 정지 |
| `DELETE /translate/jobs/{id}` | 작업 목록에서 제거(활성 작업은 취소 후 제거) |

파일 번역 원본/결과는 `translations/<타임스탬프>_<원본명>` / `...<원본명>.ko.<확장자>`로 저장됩니다.

### 원격(PC) 클라이언트

다른 PC에서 폴링 방식으로 전송·수신(표준라이브러리만 사용, pip 불필요):

```bash
python translate_client.py --server http://<Mac의 LAN IP>:8300 자막.srt
# 또는 GEMMA_SERVER=http://<Mac IP>:8300 설정 후 python translate_client.py 자막.srt
```

업로드 → 진행률 표시(5초 폴링) → 완료 시 `<이름>.ko.<확장자>` 저장.

## 환경변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `PORT` | `8300` | 서버 포트 (start.bat / start.sh) |
| `MODEL_ID` | `gemma4-srt:latest` | 기본 모델 (Ollama name:tag) |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama 서버 주소 |
| `OLLAMA_TIMEOUT` | `600` | Ollama 요청 청크 간 타임아웃(초) |
| `SUMMARY_MAX_CHARS` | `20000` | URL 요약 시 본문 최대 길이 |
| `SEARCH_MAX_CHARS` | `4000` | 검색 시 페이지당 본문 최대 길이 |
| `SEARCH_LIMIT` | `4` | 검색 결과 페이지 수 |
| `TRANSLATE_CHUNK_BLOCKS` | `50` | 파일 번역 청크 최대 블록 수 |
| `TRANSLATE_WORKERS` | `3` | 파일 번역 병렬 워커 수 (Ollama `OLLAMA_NUM_PARALLEL`에 따라 실병렬 결정) |
| `TRANSLATE_GEN_RETRIES` | `3` | 생성 일시 실패·마커 드리프트 재시도 수 |
| `WEB_SEARCH_MCP_JS` | `./web-search-mcp/dist/index.js` | MCP 서버 진입점 경로 |
| `WEB_SEARCH_MCP_TIMEOUT` | `120` | MCP 호출 타임아웃(초) |

## 구조

```
gemma4_api.py    # FastAPI 서버 — 채팅/요약/검색/번역 API + Ollama REST 클라이언트
gemma4_ui.html   # 싱글파일 웹 UI (마크다운 렌더링, 스트리밍, 추론 토글)
websearch_mcp.py # web-search-mcp(stdio MCP) 최소 클라이언트 (JSON-RPC 직접 구현)
start.bat        # 윈도우 실행 스크립트 — 의존성 확인 후 기동
start.sh         # 맥 실행 스크립트 — 기존 인스턴스 정리 후 기동
requirements.txt # 파이썬 의존성 (fastapi, uvicorn, python-multipart)
```
