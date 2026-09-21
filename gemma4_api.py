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
  GET  /                 — 채팅 웹페이지(마크다운·스트리밍·리셋·추론 토글·URL 요약·검색)

실행: ~/.local/share/uv/tools/mlx-lm/bin/python -m uvicorn gemma4_api:app --port 8300
"""
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from mlx_lm import load, generate, stream_generate

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
    tokenizer = get_model(req.model)[1]
    return tokenizer.apply_chat_template(
        [m.model_dump() for m in req.messages],
        add_generation_prompt=True,
        tokenize=False,
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
    model, tokenizer = get_model(req.model)
    prompt = build_prompt(req)
    t0 = time.time()
    text = generate(model, tokenizer, prompt=prompt, max_tokens=req.max_tokens)
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
    model, tokenizer = get_model(req.model)
    prompt = build_prompt(req)
    t0 = time.time()
    chunks = 0
    for resp in stream_generate(
        model, tokenizer, prompt=prompt, max_tokens=req.max_tokens
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




@app.get("/", include_in_schema=False)
def index():
    return FileResponse(UI_PATH)
