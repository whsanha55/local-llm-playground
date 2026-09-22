# local-llm-playground

Apple Silicon Mac에서 [MLX](https://github.com/ml-explore/mlx)로 로컬 LLM을 구동하는 채팅 서버.
FastAPI 백엔드 + 싱글파일 웹 UI + 웹검색 MCP 연동으로, 외부 API 키 없이 모든 추론을 내 Mac에서만 실행합니다.

## 주요 기능

- **로컬 채팅** — mlx-lm으로 Gemma 모델 추론, NDJSON 토큰 스트리밍 + tok/s 표시
- **추론(thinking) 채널** — 추론 과정을 별도 접이식 블록으로 분리 표시, 토글로 on/off
- **URL 요약** — 웹페이지 본문을 추출해 로컬 모델로 요약
- **웹 검색** — 검색 결과 페이지들을 읽어 로컬 모델로 종합 답변 생성
- **원커맨드 실행** — `./start.sh` 한 번으로 서버 기동 + 모델 프리로드 + 브라우저 오픈

## 기본 모델

[`Jiunsong/supergemma4-e4b-abliterated-mlx`](https://huggingface.co/Jiunsong/supergemma4-e4b-abliterated-mlx) — 수위 제한이 완화된(abliterated) 모델입니다. 출력 수위에 주의하세요. `MODELS` dict를 수정하면 다른 MLX 모델로 교체할 수 있습니다.

## 요구사항

- macOS (Apple Silicon)
- [uv](https://docs.astral.sh/uv/) + mlx-lm 툴
- Node.js ≥ 20 (web-search-mcp 빌드용)
- 인터넷 연결 (첫 실행 시 모델 다운로드, 검색·요약 기능)

## 설치

```bash
# 1) mlx-lm (uv 툴 환경에 FastAPI/uvicorn 포함)
uv tool install mlx-lm

# 2) web-search-mcp — 웹검색·본문추출용 MCP 서버 (mrkrsl/web-search-mcp, MIT)
git clone https://github.com/mrkrsl/web-search-mcp
cd web-search-mcp
npm install
npx playwright install
npm run build
cd ..
```

## 실행

```bash
./start.sh          # 기본 포트 8300, 서버 준비되면 브라우저 자동 오픈
PORT=9000 ./start.sh  # 포트 변경
```

수동 실행:

```bash
~/.local/share/uv/tools/mlx-lm/bin/python -m uvicorn gemma4_api:app --host 127.0.0.1 --port 8300
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
| `GET /translate/jobs` | 번역 작업 목록(상태·진행률, 대기순번) |
| `GET /translate/jobs/{id}` | 작업 상태 조회. 완료 시 번역 결과 포함 |

파일 번역 원본/결과는 `translations/<타임스탬프>_<원본명>` / `...<원본명>.ko.<확장자>`로 저장됩니다.

## 환경변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `PORT` | `8300` | 서버 포트 (start.sh) |
| `MODEL_ID` | `supergemma-abliterated` | 기본 모델 |
| `SUMMARY_MAX_CHARS` | `20000` | URL 요약 시 본문 최대 길이 |
| `SEARCH_MAX_CHARS` | `4000` | 검색 시 페이지당 본문 최대 길이 |
| `SEARCH_LIMIT` | `4` | 검색 결과 페이지 수 |
| `TRANSLATE_CHUNK_CHARS` | `2000` | 파일 번역 청크 목표 글자수 |
| `TRANSLATE_WORKERS` | `3` | 파일 번역 병렬 워커 수 |
| `WEB_SEARCH_MCP_JS` | `./web-search-mcp/dist/index.js` | MCP 서버 진입점 경로 |
| `WEB_SEARCH_MCP_TIMEOUT` | `120` | MCP 호출 타임아웃(초) |

## 구조

```
gemma4_api.py    # FastAPI 서버 — 채팅/요약/검색 API + 모델 로딩
gemma4_ui.html   # 싱글파일 웹 UI (마크다운 렌더링, 스트리밍, 추론 토글)
websearch_mcp.py # web-search-mcp(stdio MCP) 최소 클라이언트 (JSON-RPC 직접 구현)
start.sh         # 실행 스크립트 — 기존 인스턴스 정리 후 기동
```
