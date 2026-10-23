"""mcp_client_tester.py - A Minimal MCP Client That Tests minimal_mcp_server.py Over Real Pipes.

1. Spawns minimal_mcp_server.py as a subprocess with piped stdin/stdout, as an MCP host does.
2. Runs the lifecycle: 'initialize' (version negotiation, capabilities), the 'initialized'
   notification, 'ping', 'tools/list', 'tools/call'.
3. Checks the error contract: parse error, invalid request, unknown method, unknown tool, invalid
   arguments, tool execution errors (isError), access control, and that notifications get no reply.
4. Shuts down the way the specification describes: close the server's stdin and wait for it to exit.

Every read has a timeout, so a server that fails to answer makes the test fail instead of hang, and
stderr is drained on a thread so a chatty server cannot block on a full pipe.

Zero external dependencies. Pure Python standard library.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from typing import Any

TIMEOUT_S = 5.0


class MCPTestClient:
    """A minimal MCP client speaking newline-delimited JSON-RPC to a server subprocess."""

    def __init__(self, server_path: str, root: str) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, server_path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1, cwd=root,
        )
        self.msg_id = 0
        self.stdout_lines: "queue.Queue[str]" = queue.Queue()
        self.stderr_lines: list[str] = []
        threading.Thread(target=self._pump, args=(self.proc.stdout, self.stdout_lines.put), daemon=True).start()
        threading.Thread(target=self._pump, args=(self.proc.stderr, self.stderr_lines.append), daemon=True).start()

    @staticmethod
    def _pump(stream: Any, sink: Any) -> None:
        for line in stream:
            sink(line)

    def write_raw(self, line: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def read_message(self) -> dict:
        try:
            return json.loads(self.stdout_lines.get(timeout=TIMEOUT_S))
        except queue.Empty:
            raise AssertionError(f"no response within {TIMEOUT_S}s; server stderr: {''.join(self.stderr_lines)}")

    def request(self, method: str, params: dict | None = None) -> dict:
        self.msg_id += 1
        self.write_raw(json.dumps({"jsonrpc": "2.0", "id": self.msg_id, "method": method, "params": params or {}}))
        response = self.read_message()
        assert response.get("id") == self.msg_id, f"response id {response.get('id')} != request id {self.msg_id}"
        return response

    def notify(self, method: str, params: dict | None = None) -> None:
        self.write_raw(json.dumps({"jsonrpc": "2.0", "method": method, "params": params or {}}))

    def call_tool(self, name: str, arguments: dict) -> dict:
        return self.request("tools/call", {"name": name, "arguments": arguments})

    def close(self) -> int:
        """Shut down as the spec describes: close stdin, wait, then terminate if needed."""
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        try:
            return self.proc.wait(timeout=TIMEOUT_S)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            return self.proc.wait(timeout=TIMEOUT_S)


def run_verification_suite() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    server_script = os.path.join(here, "minimal_mcp_server.py")
    client = MCPTestClient(server_script, root=here)
    exit_code = None
    try:
        # 1. initialize: a supported version is echoed back; capabilities declare tools
        init = client.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                             "clientInfo": {"name": "test-harness", "version": "1.0"}})
        assert init["result"]["protocolVersion"] == "2025-06-18"
        assert init["result"]["serverInfo"]["name"] == "baremetal-mcp-server"
        assert "tools" in init["result"]["capabilities"]

        # 2. the initialized notification gets no reply: the next message on stdout answers the ping
        client.notify("notifications/initialized")
        assert client.request("ping")["result"] == {}

        # 3. an unsupported version gets the server's newest version instead (the client may then disconnect)
        assert client.request("initialize", {"protocolVersion": "1999-01-01"})["result"]["protocolVersion"] == "2025-06-18"

        # 4. discovery
        tools = client.request("tools/list")["result"]["tools"]
        assert {t["name"] for t in tools} == {"calculate_token_budget", "read_file_head"}
        assert all(t["inputSchema"]["type"] == "object" for t in tools)

        # 5. a successful call returns text content plus structured content
        budget = client.call_tool("calculate_token_budget", {"total_context_limit": 100_000, "used_tokens": 85_000})["result"]
        assert budget["isError"] is False
        assert json.loads(budget["content"][0]["text"]) == budget["structuredContent"]
        assert budget["structuredContent"]["remaining_tokens"] == 15_000 and budget["structuredContent"]["status"] == "warning"

        head = client.call_tool("read_file_head", {"filepath": "minimal_mcp_server.py", "max_lines": 3})["result"]
        assert head["structuredContent"]["lines_read"] == 3
        assert "minimal_mcp_server.py" in head["structuredContent"]["content"][0]

        # 6. protocol errors (-32602): unknown tool, missing or mistyped arguments
        assert client.call_tool("no_such_tool", {})["error"]["code"] == -32602
        assert client.call_tool("calculate_token_budget", {"total_context_limit": 100_000})["error"]["code"] == -32602
        assert client.call_tool("calculate_token_budget", {"total_context_limit": "lots", "used_tokens": 1})["error"]["code"] == -32602

        # 7. tool execution errors are results with isError, so the model can read them
        outside = client.call_tool("read_file_head", {"filepath": os.path.join("..", "README.md")})["result"]
        assert outside["isError"] is True and "outside" in outside["content"][0]["text"]
        missing = client.call_tool("read_file_head", {"filepath": "does_not_exist.txt"})["result"]
        assert missing["isError"] is True

        # 8. JSON-RPC errors: unknown method, malformed JSON, invalid request
        assert client.request("non_existent_method")["error"]["code"] == -32601
        client.write_raw("{not json")
        assert client.read_message()["error"]["code"] == -32700
        client.write_raw(json.dumps([{"jsonrpc": "2.0", "id": 99, "method": "ping"}]))
        assert client.read_message()["error"]["code"] == -32600

        # 9. stdout carried only JSON-RPC messages; the logs went to stderr
        assert client.stdout_lines.empty()
    finally:
        exit_code = client.close()
    assert exit_code == 0, f"server exited with {exit_code}"
    assert any("stdin closed" in line for line in client.stderr_lines)

    print("[OK] MCP lifecycle, discovery, tool calls, error contract, access control, and stdio shutdown verified.")


if __name__ == "__main__":
    print("Executing stdio MCP Client-Server Verification...")
    run_verification_suite()
