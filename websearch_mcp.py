"""web-search-mcp(stdio MCP 서버) 최소 클라이언트.

mrkrsl/web-search-mcp 는 MCP stdio 서버 — 줄 단위 JSON-RPC 2.0으로 통신한다.
공식 mcp 파이썬 패키지를 mlx-lm 툴 env에 추가하지 않고, 필요한 최소
(initialize 핸드셰이크 + tools/call)만 직접 구현한다.

단독 스모크 테스트:
  python websearch_mcp.py https://example.com
"""
import asyncio
import json
import os
import sys
from pathlib import Path

SERVER_JS = Path(os.environ.get(
    "WEB_SEARCH_MCP_JS",
    str(Path(__file__).with_name("web-search-mcp") / "dist" / "index.js"),
))
CALL_TIMEOUT = float(os.environ.get("WEB_SEARCH_MCP_TIMEOUT", "120"))


class McpError(RuntimeError):
    pass


class McpClient:
    """node 서브프로세스 하나를 띄워 두고 재사용. 죽었으면 다음 호출에서 재시작."""

    def __init__(self):
        self._proc = None
        self._id = 0
        self._lock = asyncio.Lock()

    async def _ensure(self):
        if self._proc and self._proc.returncode is None:
            return
        if not SERVER_JS.exists():
            raise McpError(
                f"MCP 서버가 없습니다: {SERVER_JS} — web-search-mcp 빌드가 필요합니다"
            )
        self._proc = await asyncio.create_subprocess_exec(
            "node", str(SERVER_JS),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            # stderr는 서버 콘솔로 흘러간다(playwright 에러 진단용)
        )
        try:
            await asyncio.wait_for(self._request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "gemma4-api", "version": "1.0"},
            }), 30)
        except asyncio.TimeoutError:
            raise McpError("MCP 서버 initialize 시간 초과") from None
        await self._notify("notifications/initialized")

    async def _send(self, obj):
        self._proc.stdin.write((json.dumps(obj) + "\n").encode())
        await self._proc.stdin.drain()

    async def _notify(self, method):
        await self._send({"jsonrpc": "2.0", "method": method})

    async def _request(self, method, params):
        """요청 하나를 보내고 응답 id가 일치하는 줄을 기다린다."""
        self._id += 1
        rid = self._id
        await self._send(
            {"jsonrpc": "2.0", "id": rid, "method": method, "params": params}
        )
        while True:
            line = await self._proc.stdout.readline()
            if not line:
                raise McpError("MCP 서버가 응답을 끊었습니다")
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue  # 프로토콜 밖 잡음은 건너뛴다
            if msg.get("id") != rid:
                continue  # 서버 발신 알림 등 — 이 요청의 응답이 아님
            if "error" in msg:
                raise McpError(msg["error"].get("message", str(msg["error"])))
            return msg.get("result", {})

    async def call(self, tool, args):
        """tools/call — 결과 텍스트를 합쳐 반환. 실패/타임아웃 시 서버를 정리한다."""
        async with self._lock:
            try:
                await self._ensure()
                result = await asyncio.wait_for(
                    self._request("tools/call", {"name": tool, "arguments": args}),
                    CALL_TIMEOUT,
                )
            except (McpError, asyncio.TimeoutError, OSError) as e:
                await self._kill()
                raise McpError(f"MCP 호출 실패 ({tool}): {e}") from e
        if result.get("isError"):
            raise McpError(f"MCP 툴 오류 ({tool}): {_text(result)}")
        return _text(result)

    async def _kill(self):
        if self._proc:
            try:
                self._proc.kill()
                await self._proc.wait()
            except (ProcessLookupError, OSError):
                pass
            self._proc = None

    async def close(self):
        """정상 종료 — stdin을 닫아 node가 빠져나가게 한다. 안 되면 강제."""
        async with self._lock:
            if self._proc and self._proc.returncode is None:
                try:
                    self._proc.stdin.close()
                    try:
                        await asyncio.wait_for(self._proc.wait(), 5)
                    except asyncio.TimeoutError:
                        await self._kill()
                except OSError:
                    await self._kill()
            elif self._proc:
                await self._kill()
            self._proc = None

    # 편의 메서드
    async def fetch_page(self, url, max_chars):
        return await self.call("get-single-web-page-content", {
            "url": url, "maxContentLength": int(max_chars),
        })

    async def search(self, query, limit, max_chars):
        # 파라미터는 숫자/불리언 타입으로 넘긴다 — 문자열로 넘기면 서버의
        # "Llama 감지" 로직이 maxContentLength를 2000자로 강제한다.
        return await self.call("full-web-search", {
            "query": query, "limit": int(limit),
            "includeContent": True, "maxContentLength": int(max_chars),
        })


def _text(result):
    return "\n".join(
        c.get("text", "") for c in result.get("content", []) if c.get("type") == "text"
    ).strip()


client = McpClient()  # gemma4_api 가 쓰는 공유 인스턴스


if __name__ == "__main__":
    async def _main():
        url = sys.argv[1] if len(sys.argv) > 1 else "https://example.com"
        out = await client.fetch_page(url, 20000)
        print(out[:2000] or "(빈 응답)")
        await client.close()
    asyncio.run(_main())
