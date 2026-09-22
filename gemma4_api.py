"""gemma-4 대화 API + 웹 UI — Ollama 백엔드(윈도우/맥 공용).

모델: Ollama로 설치된 모델(기본 gemma4-srt:latest — SRT 번역 튜닝).
추론은 Ollama 서버(http://127.0.0.1:11434, OLLAMA_HOST env로 변경)에 맡기고
이 서버는 프롬프트 조립·스트리밍 브리지만 담당한다. 모델 목록은 설치된
것에서 동적으로 읽는다. 추론 채널(message.thinking)은 스트림에 <channel|>
마커를 끼워 구분해 기존 웹 UI의 분리 로직을 그대로 쓴다.

웹 UI는 gemma4_ui.html 로 분리 — GET / 에서 매번 디스크에서 읽는다(수정 즉시 반영).
웹 본문 추출·검색은 web-search-mcp(stdio MCP 서버, websearch_mcp.py 경유)를 사용
— node로 web-search-mcp 빌드가 있어야 한다(없으면 요약/검색만 오류 남).

API:
  GET  /models           — 선택 가능 모델 목록(설치된 Ollama 모델)
  POST /chat             — 완결 응답 {messages, model, enable_thinking, max_tokens}
  POST /chat/stream      — NDJSON 스트리밍 {"t":토큰}... {"done":true,"seconds","chunks"}
  POST /summarize/stream — URL 본문 추출 → 요약 스트리밍 {url,...}
                           {"phase":"fetch"} 후 {"t":...}... {"done":...}
  POST /search/stream    — 웹 검색 → 종합 답변 스트리밍 {query,limit,...}
                           {"phase":"search"} 후 {"t":...}... {"done":...}
  POST /translate        — .srt/.txt 파일 업로드 → 한국어 번역(완결 응답, 청크 병렬)
  GET  /                 — 채팅 웹페이지(마크다운·스트리밍·리셋·추론 토글·URL 요약·검색·파일 번역)

실행: pip install -r requirements.txt
      python -m uvicorn gemma4_api:app --port 8300   (또는 start.bat)
"""
import asyncio
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from websearch_mcp import McpError
from websearch_mcp import client as mcp

DEFAULT_MODEL = os.environ.get("MODEL_ID", "gemma4-srt:latest")
CLOSER = "<channel|>"  # 추론→답변 전환 마커 — 웹 UI가 이걸로 분리한다
UI_PATH = Path(__file__).with_name("gemma4_ui.html")
TRANSLATE_UI_PATH = Path(__file__).with_name("translate.html")
SUMMARY_MAX_CHARS = int(os.environ.get("SUMMARY_MAX_CHARS", "20000"))
SEARCH_MAX_CHARS = int(os.environ.get("SEARCH_MAX_CHARS", "4000"))  # 페이지당
SEARCH_LIMIT = int(os.environ.get("SEARCH_LIMIT", "4"))
TRANSLATE_MAX_BYTES = int(os.environ.get("TRANSLATE_MAX_BYTES", "2000000"))
TRANSLATE_CHUNK_BLOCKS = int(os.environ.get("TRANSLATE_CHUNK_BLOCKS", "50"))  # 청크 최대 블록 수
TRANSLATE_WORKERS = int(os.environ.get("TRANSLATE_WORKERS", "3"))  # 병렬 효과는 Ollama의 OLLAMA_NUM_PARALLEL 설정을 따름
GEN_RETRIES = int(os.environ.get("TRANSLATE_GEN_RETRIES", "3"))  # 생성 일시 실패·마커 드리프트 재시도 수
TRANSLATE_DIR = Path(__file__).parent / "translations"

class OllamaError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


# ollama 파이썬 패키지 대신 REST를 직접 쓴다(버전에 따라 think 지원이 갈리므로).
# 로컬 서비스라 시스템 프록시를 우회한다.
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
if "://" not in OLLAMA_URL:
    OLLAMA_URL = "http://" + OLLAMA_URL  # "127.0.0.1:11434" 관례 지원
READ_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "600"))  # 청크 간 격리 타임아웃
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _request(path: str, payload=None, timeout=READ_TIMEOUT):
    """Ollama REST 호출 → 파싱된 JSON. HTTP 오류는 OllamaError 로."""
    if payload is None:
        req = urllib.request.Request(OLLAMA_URL + path)
    else:
        req = urllib.request.Request(
            OLLAMA_URL + path, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
    try:
        with _opener.open(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:500]
        try:
            body = json.loads(body).get("error", body)
        except json.JSONDecodeError:
            pass
        raise OllamaError(e.code, body) from None


def _chat_payload(model_id, messages, think, max_tokens, stream):
    return dict(model=model_id, messages=messages, think=think, stream=stream,
                options={"num_predict": max_tokens})


def _installed() -> List[str]:
    """설치된 Ollama 모델명(name:tag) 목록."""
    tags = _request("/api/tags", timeout=15)
    return sorted(m.get("model") or m.get("name") or "" for m in tags.get("models", []))


def ensure_model(model_id: str):
    if model_id not in _installed():
        raise HTTPException(404, f"설치되지 않은 모델: {model_id} — ollama pull {model_id}")


def _unsupported_thinking(e) -> bool:
    return "does not support thinking" in str(e)


def chat_once(model_id, messages, think, max_tokens):
    """완결 응답 — {message:{content,thinking},...} dict. 추론 미지원 모델에
    think=True 면 끄고 재시도."""
    try:
        return _request("/api/chat", _chat_payload(model_id, messages, think, max_tokens, False))
    except OllamaError as e:
        if think and _unsupported_thinking(e):
            return _request("/api/chat", _chat_payload(model_id, messages, False, max_tokens, False))
        raise


def stream_parts(model_id, messages, think, max_tokens):
    """("think"|"text", 조각) NDJSON 스트리밍 — 동기 제너레이터(스레드에서 소비한다)."""
    def _parts(think_flag):
        req = urllib.request.Request(
            OLLAMA_URL + "/api/chat",
            data=json.dumps(_chat_payload(model_id, messages, think_flag, max_tokens, True)).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            r = _opener.open(req, timeout=READ_TIMEOUT)
        except urllib.error.HTTPError as e:
            raise OllamaError(e.code, e.read().decode("utf-8", errors="replace")[:500]) from None
        with r:
            for line in r:
                if not line.strip():
                    continue
                part = json.loads(line)
                if part.get("error"):
                    raise OllamaError(500, part["error"])
                msg = part.get("message") or {}
                if msg.get("thinking"):
                    yield "think", msg["thinking"]
                if msg.get("content"):
                    yield "text", msg["content"]
                if part.get("done"):
                    return

    try:
        yield from _parts(think)
    except OllamaError as e:  # 첫 토큰 전에만 난다 — 재시도 안전
        if not (think and _unsupported_thinking(e)):
            raise
        yield from _parts(False)


def _warmup():
    """기동 검문 — Ollama 실행·기본 모델 설치 확인 후 모델을 메모리에 올린다."""
    try:
        names = _installed()
    except Exception as e:
        raise RuntimeError(
            f"Ollama에 연결할 수 없습니다 — Ollama가 실행 중인지 확인: {e}"
        ) from e
    if DEFAULT_MODEL not in names:
        raise RuntimeError(
            f"모델이 설치되어 있지 않습니다: {DEFAULT_MODEL} — ollama pull {DEFAULT_MODEL}"
        )
    t0 = time.time()
    chat_once(DEFAULT_MODEL, [{"role": "user", "content": "hi"}], False, 1)
    print(f"{DEFAULT_MODEL} 로드 완료 ({time.time() - t0:.1f}s)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(_warmup)  # 기본 모델은 미리 로드
    worker = asyncio.create_task(translate_worker())
    try:
        yield
    finally:
        worker.cancel()
        await mcp.close()  # web-search-mcp 서브프로세스 정리


app = FastAPI(title="gemma-4-e4b API", lifespan=lifespan)


class Msg(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Msg]
    model: str = DEFAULT_MODEL
    enable_thinking: bool = False
    max_tokens: int = 1024


class SummarizeRequest(BaseModel):
    url: str
    model: str = DEFAULT_MODEL
    enable_thinking: bool = False
    max_tokens: int = 1024


class SearchRequest(BaseModel):
    query: str
    limit: int = SEARCH_LIMIT
    model: str = DEFAULT_MODEL
    enable_thinking: bool = False
    max_tokens: int = 1024


SUMMARIZE_PROMPT = (
    "아래는 웹페이지에서 추출한 본문이야. 이 내용을 한국어로 요약해줘.\n"
    "핵심 주제를 한 줄로 먼저 말하고, 중요한 포인트를 불릿으로 정리해줘.\n"
    "본문에 없는 내용은 만들지 말고, 본문이 비어 있으면 그렇게 말해줘.\n\n"
    "URL: {url}\n\n--- 본문 ---\n{page}\n--- 본문 끝 ---"
)

SEARCH_PROMPT = (
    "'{query}'에 대한 웹 검색 결과가 아래에 있어. 이를 바탕으로 한국어로 답변해줘.\n"
    "정보를 종합해 정리하고, 근거가 되는 출처를 [번호]로 인용해줘.\n"
    "검색 결과에 없는 내용은 만들지 마.\n\n"
    "--- 검색 결과 ---\n{results}\n--- 검색 결과 끝 ---"
)


@app.get("/models")
def models():
    names = _installed()
    if DEFAULT_MODEL in names:  # 기본 모델을 맨 위로
        names.remove(DEFAULT_MODEL)
        names.insert(0, DEFAULT_MODEL)
    return {
        "default": DEFAULT_MODEL,
        "models": [
            {"id": n, "label": n + (" (기본)" if n == DEFAULT_MODEL else "")}
            for n in names
        ],
    }


@app.get("/health")
def health():
    return {"status": "ok", "default": DEFAULT_MODEL, "installed": len(_installed())}


@app.post("/chat")
async def chat(req: ChatRequest):
    ensure_model(req.model)
    t0 = time.time()
    try:
        resp = await asyncio.to_thread(
            chat_once, req.model, [m.model_dump() for m in req.messages],
            req.enable_thinking, req.max_tokens,
        )
    except OllamaError as e:
        raise HTTPException(e.status or 500, str(e)) from e
    msg = resp.get("message") or {}
    reasoning = (msg.get("thinking") or "").strip()
    answer = (msg.get("content") or "").strip()
    return {
        "answer": answer,
        "reasoning": reasoning,
        "model": req.model,
        "enable_thinking_param": req.enable_thinking,
        "model_thought": bool(reasoning),
        "seconds": round(time.time() - t0, 2),
    }


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """NDJSON 스트리밍. Ollama 동기 스트림은 스레드에서 소비해 큐로 넘긴다.
    추론 조각 뒤에는 <channel|> 마커를 끼워 넣으므로 클라이언트는
    기존처럼 마커로 추론/답변을 분리한다."""
    return StreamingResponse(
        gen_ndjson(req),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def gen_ndjson(req: ChatRequest):
    """생성 NDJSON 라인 제너레이터 — 채팅/요약/검색 엔드포인트가 공유.

    동기 이터레이터를 이벤트 루프에서 바로 돌면 토큰 대기 중 다른 요청
    (번역 작업 상태 폴링 등)이 굶으므로 생산은 스레드에 맡긴다.
    """
    ensure_model(req.model)
    messages = [m.model_dump() for m in req.messages]
    t0 = time.time()
    chunks = 0
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    DONE = object()

    def produce():
        try:
            for item in stream_parts(req.model, messages,
                                     req.enable_thinking, req.max_tokens):
                loop.call_soon_threadsafe(q.put_nowait, item)
        except BaseException as e:  # 스레드 안 예외는 큐로 전달
            loop.call_soon_threadsafe(q.put_nowait, e)
        finally:
            loop.call_soon_threadsafe(q.put_nowait, DONE)

    threading.Thread(target=produce, daemon=True).start()

    closed = False  # 추론→본문 전환 마커는 첫 본문 조각 앞에 한 번
    while True:
        item = await q.get()
        if item is DONE:
            break
        if isinstance(item, BaseException):
            yield json.dumps({"error": str(item)}, ensure_ascii=False) + "\n"
            return
        kind, text = item
        if not text:
            continue
        if kind == "text" and not closed:
            closed = True
            chunks += 1
            yield json.dumps({"t": CLOSER}, ensure_ascii=False) + "\n"
        chunks += 1
        yield json.dumps({"t": text}, ensure_ascii=False) + "\n"
    yield json.dumps(
        {"done": True, "seconds": round(time.time() - t0, 2), "chunks": chunks}
    ) + "\n"


async def gen_tool_stream(phase: str, req, fetch, to_messages):
    """요약/검색 공통: 진행 알림 → MCP 호출 → 결과로 ChatRequest 를 만들어 스트리밍.

    req: model/enable_thinking/max_tokens 필드를 가진 요약·검색 요청
    fetch: await 시 MCP 결과 텍스트를 반환하는 coroutine
    to_messages: 결과 텍스트로 messages 리스트를 만드는 함수
    """
    async def gen():
        yield json.dumps({"phase": phase}, ensure_ascii=False) + "\n"
        try:
            result = await fetch()
        except McpError as e:
            yield json.dumps({"error": str(e)}, ensure_ascii=False) + "\n"
            return
        chat = ChatRequest(
            messages=to_messages(result),
            model=req.model,
            enable_thinking=req.enable_thinking,
            max_tokens=req.max_tokens,
        )
        async for line in gen_ndjson(chat):
            yield line

    return StreamingResponse(
        gen(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/summarize/stream")
async def summarize_stream(req: SummarizeRequest):
    """URL 본문을 web-search-mcp 로 추출해 한국어 요약 스트리밍."""
    if not req.url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="http(s):// URL이 필요합니다")

    def to_messages(page: str):
        return [Msg(role="user", content=SUMMARIZE_PROMPT.format(
            url=req.url, page=page or "(본문을 추출하지 못했습니다)"
        ))]

    return await gen_tool_stream(
        "fetch",
        req,
        lambda: mcp.fetch_page(req.url, SUMMARY_MAX_CHARS),
        to_messages,
    )


@app.post("/search/stream")
async def search_stream(req: SearchRequest):
    """웹 검색(full-web-search, 본문 포함) 결과로 종합 답변 스트리밍."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="검색어가 비었습니다")

    def to_messages(results: str):
        return [Msg(role="user", content=SEARCH_PROMPT.format(
            query=req.query, results=results or "(검색 결과가 없습니다)"
        ))]

    return await gen_tool_stream(
        "search",
        req,
        lambda: mcp.search(req.query, req.limit, SEARCH_MAX_CHARS),
        to_messages,
    )


# --- 파일 번역(/translate): 파싱·청킹·마커 재조립 순수 함수 + 병렬 실행 ---

TS_RE = re.compile(r"\d{1,2}:\d{2}:\d{2}[,.]\d{3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[,.]\d{3}")
MARKER_RE = re.compile(r"^\[(\d+)\]\s*(.*)$")


def decode_upload(data: bytes) -> str:
    """utf-8(BOM 포함) → cp949 폴백. 윈도우 생성 파일 대비."""
    for enc in ("utf-8-sig", "cp949"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    raise HTTPException(400, "인코딩을 인식할 수 없습니다 (utf-8/cp949 지원)")


def parse_srt(text: str):
    """SRT → [(번호줄, 타임스탬프줄, 본문줄리스트)]. 형식이 아니면 None.

    절반 이상 덩어리가 타임스탬프 블록이면 SRT로 본다.
    """
    parts = re.split(r"\n\s*\n", text.replace("\r\n", "\n").strip())
    blocks = []
    for raw in parts:
        lines = [l for l in raw.split("\n") if l.strip()]
        ts_at = next((i for i, l in enumerate(lines) if TS_RE.search(l)), None)
        if ts_at is None or ts_at + 1 >= len(lines):
            continue
        header = " ".join(lines[:ts_at]).strip() or str(len(blocks) + 1)
        blocks.append((header, lines[ts_at], [l.strip() for l in lines[ts_at + 1:]]))
    return blocks if blocks and len(blocks) >= len(parts) * 0.5 else None


def chunk_items(items, limit=TRANSLATE_CHUNK_BLOCKS):
    """[(번호, 한줄텍스트)] → 최대 limit 블록 크기 청크 리스트."""
    return [items[i:i + limit] for i in range(0, len(items), limit)]


def translate_prompt(lines):
    body = "\n".join(f"[{no}] {text}" for no, text in lines)
    return (
        "아래 [번호]로 시작하는 각 줄을 한국어로 자연스럽게 번역해줘.\n"
        "번역문만 [번호] 접두와 함께 한 줄씩 출력하고, 번호를 건너뛰거나 "
        "설명을 덧붙이지 마.\n\n" + body
    )


def parse_markers(text, expected):
    """응답 → {번호: 번역문}. expected 의 번호가 하나라도 빠지면 None."""
    got = {}
    for line in text.splitlines():
        m = MARKER_RE.match(line.strip())
        if m:
            got[m.group(1)] = m.group(2).strip()
    return got if all(no in got for no, _ in expected) else None


def _generate(model_id: str, text: str, max_tokens: int) -> str:
    """워커 스레드용 생성 래퍼 — 추론 끄고 본문만 받는다.

    일시 오류(연결 끊김·타임아웃·5xx)는 짧게 쉬었다가 재시도한다.
    400/404는 재시도로 안 풀리는 요청 문제라 즉시 던진다.
    """
    for attempt in range(GEN_RETRIES):
        try:
            resp = chat_once(model_id, [{"role": "user", "content": text}], False, max_tokens)
            return ((resp.get("message") or {}).get("content") or "").strip()
        except (OllamaError, OSError) as e:
            if isinstance(e, OllamaError) and e.status in (400, 404):
                raise
            if attempt == GEN_RETRIES - 1:
                raise
            time.sleep(1.5 ** attempt)


def translate_all(model_id, chunks, on_progress=None, should_cancel=None):
    """ThreadPoolExecutor 로 청크 병렬 번역 → ({번호: 번역문}, [미번역 번호]).

    Ollama는 요청을 큐에 쌓아 처리하므로 실제 병렬도는 서버 설정
    (OLLAMA_NUM_PARALLEL)을 따른다 — 직렬이어도 동작에는 문제 없다.
    대청크 마커 드리프트(모델이 긴 리스트에서 블록을 생략) 시 같은 크기
    재시도는 무의미하므로 반분할해 재귀 재시도한다. 8블록 이하 잎 청크는
    같은 크기로 GEN_RETRIES 번 반복 재시도한 뒤 한 블록씩 개별 번역까지
    폴백한다 — 원문 유지는 최후의 수단.
    on_progress(완료 청크 수)는 청크가 끝날 때마다, should_cancel()이 True면
    남은 청크를 건너뛴다(진행 중 청크는 마저 끝나고 정지 — 협력 취소).
    """
    def work(chunk):
        if should_cancel and should_cancel():
            return {}, []
        prompt = translate_prompt(chunk)
        max_tokens = min(8192, max(1024, sum(len(t) for _, t in chunk)))
        if len(chunk) <= 8:  # 잎 청크 — 같은 크기 반복 재시도(반복검사)
            for _ in range(GEN_RETRIES):
                got = parse_markers(_generate(model_id, prompt, max_tokens), chunk)
                if got is not None:
                    return got, []
            if len(chunk) > 1:  # 그래도 드리프트 — 한 블록씩 개별 번역
                by, miss = {}, []
                for item in chunk:
                    g, m = work([item])
                    by.update(g)
                    miss.extend(m)
                return by, miss
            return dict(chunk), [no for no, _ in chunk]  # 최후: 원문 유지
        got = parse_markers(_generate(model_id, prompt, max_tokens), chunk)
        if got is not None:
            return got, []
        mid = len(chunk) // 2
        lg, lu = work(chunk[:mid])
        rg, ru = work(chunk[mid:])
        return {**lg, **rg}, lu + ru

    by_no, untranslated, done = {}, [], 0
    with ThreadPoolExecutor(max_workers=TRANSLATE_WORKERS) as pool:
        futs = [pool.submit(work, c) for c in chunks]
        for fut in as_completed(futs):
            got, missing = fut.result()
            by_no.update(got)
            untranslated.extend(missing)
            done += 1
            if on_progress:
                on_progress(done)
    return by_no, untranslated


# --- 번역 작업 큐: 요청은 대기목록에 쌓고 한 번에 하나씩 실행 ---

jobs = {}  # job_id -> 상태 dict (완료 결과 포함. 로컬 도구라 만료 정리 없음)
translate_queue: asyncio.Queue = asyncio.Queue()


async def translate_worker():
    """큐 소비 루프 — 번역 작업을 순차 실행(청크 병렬은 내부 유지)."""
    while True:
        payload = await translate_queue.get()
        job = jobs.get(payload["job_id"])
        if job is None:  # 대기 중 목록에서 삭제됨 — 폐기
            continue
        if job.get("cancel"):  # 대기 중 취소 — 실행 없이 폐기
            job["state"] = "cancelled"
            continue
        job["state"], job["t0"] = "running", time.time()
        try:
            by_no, untranslated = await asyncio.get_running_loop().run_in_executor(
                None, translate_all, payload["model"], payload["chunks"],
                lambda n: job.update(done_chunks=n),
                lambda: job.get("cancel"),
            )
            if job.get("cancel"):  # 실행 중 취소 — 청크 경계에서 정지, 결과 폐기
                job.update(state="cancelled")
                continue
            if payload["is_srt"]:
                translation = "\n\n".join(
                    f"{no}\n{ts}\n{by_no[no]}" for no, ts, _ in payload["blocks"]
                )
            else:
                translation = "\n\n".join(by_no[no] for no, _ in payload["items"])
            job["seconds"] = round(time.time() - job["t0"], 1)
            out_path = TRANSLATE_DIR / (payload["stem"] + ".ko" + payload["ext"])
            out_path.write_text(translation, encoding="utf-8")
            job.update(state="done", translation=translation,
                       untranslated=untranslated, out_file=out_path.name)
        except Exception as e:  # 작업 하나 실패가 워커를 죽이지 않게
            job.update(state="error", error=str(e))


def job_view(job, with_translation=False):
    """job dict → API 응답용 뷰. wait_ahead 는 대기 순번, seconds 는 진행 경과."""
    view = {k: job[k] for k in
            ("id", "name", "state", "blocks", "chunks", "done_chunks")}
    view["seconds"] = (round(time.time() - job["t0"], 1)
                       if job["state"] == "running" else job.get("seconds"))
    if job["state"] == "queued":
        ahead = [j["id"] for j in jobs.values() if j["state"] == "queued"]
        view["wait_ahead"] = ahead.index(job["id"])
    if with_translation:
        for k in ("translation", "untranslated", "error", "out_file"):
            if k in job:
                view[k] = job[k]
    return view


@app.post("/translate", status_code=202)
async def translate(
    file: UploadFile = File(...), model: str = Form(DEFAULT_MODEL)
):
    """.srt/.txt 업로드 → 번역 작업 등록(즉시 job_id 반환). SRT 구조는 서버가 유지."""
    data = await file.read()
    if len(data) > TRANSLATE_MAX_BYTES:
        raise HTTPException(413, "파일이 2MB 제한을 초과합니다")
    orig = file.filename or "untitled.txt"
    name = orig.lower()
    text = decode_upload(data)

    if name.endswith(".srt"):
        blocks = parse_srt(text)
        if not blocks:
            raise HTTPException(400, "SRT 형식을 인식하지 못했습니다")
        items = [(no, " ".join(body)) for no, _, body in blocks]
        is_srt = True
    elif name.endswith(".txt") or not name:
        paras = [p.strip().replace("\n", " ")
                 for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
        items = [(str(i), p) for i, p in enumerate(paras)]
        blocks, is_srt = None, False
    else:
        raise HTTPException(400, "지원하지 않는 형식입니다 (.srt/.txt)")
    if not items:
        raise HTTPException(400, "번역할 텍스트가 없습니다")

    # 원본 저장: translations/<타임스탬프>_<원본명>, 결과는 .ko 확장자로 같은 폴더
    TRANSLATE_DIR.mkdir(exist_ok=True)
    stem, ext = os.path.splitext(orig)
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    in_path = TRANSLATE_DIR / f"{ts}_{stem}{ext}"
    in_path.write_bytes(data)

    job_id = uuid4().hex[:12]
    jobs[job_id] = {"id": job_id, "name": orig, "state": "queued",
                    "blocks": len(items), "chunks": len(chunk_items(items)),
                    "done_chunks": 0, "in_file": in_path.name}
    await translate_queue.put({
        "job_id": job_id, "model": model, "items": items, "blocks": blocks,
        "is_srt": is_srt, "chunks": chunk_items(items),
        "stem": f"{ts}_{stem}", "ext": ext,
    })
    return {"job_id": job_id, "state": "queued", "chunks": jobs[job_id]["chunks"]}


@app.get("/translate/jobs")
def translate_jobs():
    """번역 작업 목록(결과 본문 제외)."""
    return {"jobs": [job_view(j) for j in jobs.values()]}


@app.get("/translate/jobs/{job_id}")
def translate_job(job_id: str):
    """개별 작업 상태. 완료 시 translation 포함."""
    if job_id not in jobs:
        raise HTTPException(404, "unknown job")
    return job_view(jobs[job_id], with_translation=True)


@app.post("/translate/jobs/{job_id}/cancel")
def cancel_translate_job(job_id: str):
    """작업 취소. queued 는 즉시, running 은 청크 경계에서 정지(협력 취소)."""
    if job_id not in jobs:
        raise HTTPException(404, "unknown job")
    job = jobs[job_id]
    if job["state"] not in ("queued", "running"):
        raise HTTPException(409, f"이미 끝난 작업입니다 ({job['state']})")
    job["cancel"] = True
    if job["state"] == "queued":
        job["state"] = "cancelled"
    return job_view(job)


@app.delete("/translate/jobs/{job_id}")
def delete_translate_job(job_id: str):
    """작업 목록에서 제거. 활성(queued/running) 작업이면 취소 후 제거."""
    if job_id not in jobs:
        raise HTTPException(404, "unknown job")
    job = jobs[job_id]
    if job["state"] in ("queued", "running"):
        job["cancel"] = True
        if job["state"] == "queued":
            job["state"] = "cancelled"
    del jobs[job_id]
    return {"deleted": job_id}


def _self_test():
    """파서·청커·재조립 assert 자체점검 — `python -m gemma4_api`."""
    srt = ("1\n00:00:01,000 --> 00:00:02,000\nHello\nworld\n\n"
           "2\n00:00:02,000 --> 00:00:03,500\nBye\n\n"
           "3\n00:00:03,500 --> 00:00:04,000\nEnd")
    blocks = parse_srt(srt)
    assert blocks and len(blocks) == 3, blocks
    assert blocks[0][2] == ["Hello", "world"], blocks[0]
    assert parse_srt("그냥 텍스트\n두 줄") is None

    items = [(no, " ".join(b)) for no, _, b in blocks]
    chunks = chunk_items(items, limit=2)
    # 최대 2블록 청크 → [1,2] + [3] 의 2청크
    assert len(chunks) == 2 and sum(len(c) for c in chunks) == 3, chunks

    got = parse_markers("[1] 안녕\n[2] 잘가\n[3] 끝", items)
    assert got == {"1": "안녕", "2": "잘가", "3": "끝"}
    assert parse_markers("[1] 안녕\n[3] 끝", items) is None  # 2 누락

    out = "\n\n".join(f"{no}\n{ts}\n{got[no]}" for no, ts, _ in blocks)
    assert out.count("-->") == 3 and "안녕" in out and "Hello" not in out

    sample = Path(__file__).parent / "srt-sample/The_Odyssey.srt"
    if sample.exists():  # 실샘플: 파싱 개수 + 전체 타임스탬프 보존 확인
        sb = parse_srt(decode_upload(sample.read_bytes()))
        assert sb and len(sb) == 1405, len(sb)
        assert all(len(b[2]) and TS_RE.search(b[1]) for b in sb)
    print("self-test OK")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(UI_PATH)


@app.get("/translate", include_in_schema=False)
def translate_page():
    return FileResponse(TRANSLATE_UI_PATH)


if __name__ == "__main__":
    _self_test()
