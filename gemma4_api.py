"""gemma-4 E4B 대화 API + 웹 UI.

모델: supergemma4-e4b-abliterated (HF 저장소 — 캐시에 있으면 그대로 로드).
첫 요청 시 로딩(약 15초), 이후 캐시.

웹 UI는 gemma4_ui.html 로 분리 — GET / 에서 매번 디스크에서 읽는다(수정 즉시 반영).
웹 본문 추출·검색은 web-search-mcp(stdio MCP 서버, websearch_mcp.py 경유)를 사용.

API:
  GET  /models           — 선택 가능 모델 목록
  POST /chat             — 완결 응답 {messages, model, enable_thinking, max_tokens}
  POST /chat/stream      — NDJSON 스트리밍 {"t":토큰}... {"done":true,"seconds","chunks"}
  POST /summarize/stream — URL 본문 추출 → 요약 스트리밍 {url,...}
                           {"phase":"fetch"} 후 {"t":...}... {"done":...}
  POST /search/stream    — 웹 검색 → 종합 답변 스트리밍 {query,limit,...}
                           {"phase":"search"} 후 {"t":...}... {"done":...}
  POST /translate        — .srt/.txt 파일 업로드 → 한국어 번역(완결 응답, 청크 병렬)
  GET  /                 — 채팅 웹페이지(마크다운·스트리밍·리셋·추론 토글·URL 요약·검색·파일 번역)

실행: ~/.local/share/uv/tools/mlx-vlm/bin/python -m uvicorn gemma4_api:app --port 8300
"""
import asyncio
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from mlx_vlm import apply_chat_template, generate, load, stream_generate

from websearch_mcp import McpError
from websearch_mcp import client as mcp

MODELS = {
    "supergemma-abliterated": {
        "label": "supergemma4-e4b-abliterated (수위 높음)",
        "path": "Jiunsong/supergemma4-e4b-abliterated-mlx",
    },
}
DEFAULT_MODEL = os.environ.get("MODEL_ID", "supergemma-abliterated")
CLOSER = "<channel|>"
UI_PATH = Path(__file__).with_name("gemma4_ui.html")
SUMMARY_MAX_CHARS = int(os.environ.get("SUMMARY_MAX_CHARS", "20000"))
SEARCH_MAX_CHARS = int(os.environ.get("SEARCH_MAX_CHARS", "4000"))  # 페이지당
SEARCH_LIMIT = int(os.environ.get("SEARCH_LIMIT", "4"))
TRANSLATE_MAX_BYTES = int(os.environ.get("TRANSLATE_MAX_BYTES", "2000000"))
TRANSLATE_CHUNK_CHARS = int(os.environ.get("TRANSLATE_CHUNK_CHARS", "2000"))  # 4000서 115블록 마커 드리프트로 하향
TRANSLATE_WORKERS = int(os.environ.get("TRANSLATE_WORKERS", "3"))  # 스파이크 1.43x 확인

state = {"models": {}}  # model_id -> (model, tokenizer)


def get_model(model_id: str):
    if model_id not in MODELS:
        raise HTTPException(status_code=404, detail=f"unknown model: {model_id}")
    if model_id not in state["models"]:
        state["models"][model_id] = load(MODELS[model_id]["path"])
    return state["models"][model_id]


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model(DEFAULT_MODEL)  # 기본 모델은 미리 로드
    try:
        yield
    finally:
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


def build_prompt(req: ChatRequest):
    model, processor = get_model(req.model)
    return apply_chat_template(
        processor,
        model.config,
        [m.model_dump() for m in req.messages],
        add_generation_prompt=True,
        enable_thinking=req.enable_thinking,
    )


@app.get("/models")
def models():
    return {
        "default": DEFAULT_MODEL,
        "models": [
            {"id": mid, "label": m["label"]} for mid, m in MODELS.items()
        ],
    }


@app.get("/health")
def health():
    loaded = list(state["models"].keys())
    return {"status": "ok", "default": DEFAULT_MODEL, "loaded": loaded}


@app.post("/chat")
async def chat(req: ChatRequest):
    # MLX는 GPU 스트림이 스레드 종속이라 async(메인 스레드 실행)여야 한다.
    model, processor = get_model(req.model)
    prompt = build_prompt(req)
    t0 = time.time()
    text = generate(model, processor, prompt=prompt, max_tokens=req.max_tokens).text
    seconds = round(time.time() - t0, 2)

    if CLOSER in text:
        reasoning, answer = text.split(CLOSER, 1)
        reasoning = reasoning.replace("<|channel>thought", "").strip()
        answer = answer.strip()
    else:
        reasoning, answer = "", text.strip()

    return {
        "answer": answer,
        "reasoning": reasoning,
        "model": req.model,
        "enable_thinking_param": req.enable_thinking,
        "model_thought": bool(reasoning),
        "seconds": seconds,
    }


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """NDJSON 스트리밍. MLX 생성은 메인 스레드(async 제너레이터 본문)에서 돌고,
    yield 사이사이 청크가 flush 된다. 추론 채널은 그대로 흘려보내므로
    클라이언트가 <channel|>로 분리한다."""
    return StreamingResponse(
        gen_ndjson(req),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def gen_ndjson(req: ChatRequest):
    """생성 NDJSON 라인 제너레이터 — 채팅/요약/검색 엔드포인트가 공유."""
    model, processor = get_model(req.model)
    prompt = build_prompt(req)
    t0 = time.time()
    chunks = 0
    for resp in stream_generate(
        model, processor, prompt=prompt, max_tokens=req.max_tokens
    ):
        if resp.text:
            chunks += 1
            yield json.dumps({"t": resp.text}, ensure_ascii=False) + "\n"
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


def chunk_items(items, target=TRANSLATE_CHUNK_CHARS):
    """[(번호, 한줄텍스트)] → 번호 경계에서만 분할한 청크 리스트."""
    chunks, cur, size = [], [], 0
    for item in items:
        cur.append(item)
        size += len(item[1])
        if size >= target:
            chunks.append(cur)
            cur, size = [], 0
    if cur:
        chunks.append(cur)
    return chunks


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
    """워커 스레드용 generate 래퍼 — 챗 템플릿 적용 후 생성, 추론 채널 분리."""
    model, processor = get_model(model_id)
    prompt = apply_chat_template(
        processor, model.config,
        [{"role": "user", "content": text}],
        add_generation_prompt=True, enable_thinking=False,
    )
    out = generate(model, processor, prompt=prompt, max_tokens=max_tokens).text
    if CLOSER in out:
        out = out.split(CLOSER, 1)[1]
    return out.strip()


def translate_all(model_id, chunks):
    """ThreadPoolExecutor 로 청크 병렬 번역 → ({번호: 번역문}, [미번역 번호]).

    마커 드리프트(모델이 긴 리스트에서 블록을 생략) 시 같은 크기 재시도는
    무의미하므로 청크를 반분할해 재귀 재시도한다.
    """
    def work(chunk):
        prompt = translate_prompt(chunk)
        max_tokens = min(4096, max(1024, sum(len(t) for _, t in chunk)))
        got = parse_markers(_generate(model_id, prompt, max_tokens), chunk)
        if got is not None:
            return got, []
        if len(chunk) > 8:
            mid = len(chunk) // 2
            lg, lu = work(chunk[:mid])
            rg, ru = work(chunk[mid:])
            return {**lg, **rg}, lu + ru
        # ponytail: 8블록 이하 재실패는 원문 유지 — 전체 500보다 낫다
        return dict(chunk), [no for no, _ in chunk]

    with ThreadPoolExecutor(max_workers=TRANSLATE_WORKERS) as pool:
        results = list(pool.map(work, chunks))
    by_no, untranslated = {}, []
    for got, missing in results:
        by_no.update(got)
        untranslated.extend(missing)
    return by_no, untranslated


@app.post("/translate")
async def translate(
    file: UploadFile = File(...), model: str = Form(DEFAULT_MODEL)
):
    """.srt/.txt 업로드 → 한국어 번역. SRT는 번호·타임스탬프 구조를 서버가 재조립해 유지."""
    data = await file.read()
    if len(data) > TRANSLATE_MAX_BYTES:
        raise HTTPException(413, "파일이 2MB 제한을 초과합니다")
    name = (file.filename or "").lower()
    text = decode_upload(data)
    t0 = time.time()

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
        is_srt = False
    else:
        raise HTTPException(400, "지원하지 않는 형식입니다 (.srt/.txt)")
    if not items:
        raise HTTPException(400, "번역할 텍스트가 없습니다")

    chunks = chunk_items(items)
    by_no, untranslated = await asyncio.get_running_loop().run_in_executor(
        None, translate_all, model, chunks
    )

    if is_srt:
        translation = "\n\n".join(
            f"{no}\n{ts}\n{by_no[no]}" for no, ts, _ in blocks
        )
    else:
        translation = "\n\n".join(by_no[no] for no, _ in items)

    return {
        "translation": translation,
        "blocks": len(items),
        "chunks": len(chunks),
        "untranslated": untranslated,
        "seconds": round(time.time() - t0, 1),
        "model": model,
    }


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
    chunks = chunk_items(items, target=10)
    # "Hello world"(11자)는 단독 청크, "Bye"+"End"는 묶여 2청크
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


if __name__ == "__main__":
    _self_test()
