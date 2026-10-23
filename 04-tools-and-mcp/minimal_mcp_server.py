"""minimal_mcp_server.py - A Minimal stdio MCP Server in Pure Python.

This script implements the parts of the Model Context Protocol (specification 2025-06-18) that a tool
server needs, over the stdio transport:

1. Transport: newline-delimited JSON-RPC 2.0 on stdin/stdout. Messages never contain embedded
   newlines. stdout carries MCP messages only; every log line goes to stderr.
2. Lifecycle: 'initialize' with protocol-version negotiation and capability declaration, then the
   'notifications/initialized' notification, then normal operation. Shutdown is the client closing
   stdin; the server exits on EOF.
3. Tools: 'tools/list' and 'tools/call'. Unknown tools and invalid arguments are protocol errors
   (-32602); failures while running a tool are tool results with "isError": true.
4. Utilities: 'ping' answers with an empty result.
5. JSON-RPC errors: -32700 parse error, -32600 invalid request, -32601 method not found,
   -32602 invalid params, -32603 internal error. Every request (a message with an id) gets exactly
   one response; notifications (no id) never get one.
6. Security basics the specification requires of servers: validate inputs and control access. The
   file tool reads only inside one root directory and caps how much it returns.

Not implemented: resources, prompts, pagination, progress, cancellation, the Streamable HTTP transport,
rate limiting. mcp_client_tester.py exercises this server over real pipes.

Zero external dependencies. Pure Python standard library.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Callable, Dict, Optional

SUPPORTED_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")  # newest first
SERVER_NAME = "baremetal-mcp-server"
SERVER_VERSION = "1.1.0"
ROOT = os.path.realpath(os.environ.get("MCP_ROOT") or os.getcwd())
MAX_LINES = 200


class InvalidParams(Exception):
    """Arguments that do not match the tool's input schema (JSON-RPC -32602)."""


# --------------------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------------------
def _int_arg(arguments: dict, name: str, default: Optional[int] = None, minimum: int = 0) -> int:
    value = arguments.get(name, default)
    if value is None:
        raise InvalidParams(f"'{name}' is required")
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidParams(f"'{name}' must be an integer")
    if value < minimum:
        raise InvalidParams(f"'{name}' must be at least {minimum}")
    return value


def tool_calculate_token_budget(arguments: dict) -> dict[str, Any]:
    limit = _int_arg(arguments, "total_context_limit", minimum=1)
    used = _int_arg(arguments, "used_tokens")
    if used > limit:
        return {"total_context_limit": limit, "used_tokens": used, "remaining_tokens": 0,
                "utilization_pct": 100.0, "status": "context_exhausted"}
    pct = round(used / limit * 100.0, 2)
    return {"total_context_limit": limit, "used_tokens": used, "remaining_tokens": limit - used,
            "utilization_pct": pct, "status": "warning" if pct > 80.0 else "nominal"}


def tool_read_file_head(arguments: dict) -> dict[str, Any]:
    filepath = arguments.get("filepath")
    if not isinstance(filepath, str) or not filepath:
        raise InvalidParams("'filepath' must be a non-empty string")
    max_lines = min(_int_arg(arguments, "max_lines", default=10, minimum=1), MAX_LINES)

    # Access control: resolve symlinks and '..' first, then require the path to stay under ROOT.
    real = os.path.realpath(os.path.join(ROOT, filepath))
    if os.path.commonpath([real, ROOT]) != ROOT:
        raise PermissionError(f"'{filepath}' is outside the server's root directory")
    if not os.path.isfile(real):
        raise FileNotFoundError(f"File not found: {filepath}")

    lines: list[str] = []
    with open(real, "r", encoding="utf-8", errors="replace") as f:
        for idx, line in enumerate(f):
            if idx >= max_lines:
                break
            lines.append(line.rstrip("\r\n"))
    return {"filepath": os.path.relpath(real, ROOT), "lines_read": len(lines), "content": lines}


TOOLS: Dict[str, Callable[[dict], dict]] = {
    "calculate_token_budget": tool_calculate_token_budget,
    "read_file_head": tool_read_file_head,
}

TOOL_DEFINITIONS = [
    {
        "name": "calculate_token_budget",
        "description": "Calculate remaining token capacity and utilization status for a context window.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "total_context_limit": {"type": "integer", "minimum": 1,
                                        "description": "Maximum token capacity of the model context window."},
                "used_tokens": {"type": "integer", "minimum": 0, "description": "Number of tokens already consumed."},
            },
            "required": ["total_context_limit", "used_tokens"],
        },
    },
    {
        "name": "read_file_head",
        "description": "Read the first N lines of a file under the server's root directory, to inspect it without flooding the context.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path relative to the server's root directory."},
                "max_lines": {"type": "integer", "minimum": 1, "maximum": MAX_LINES,
                              "description": "Maximum number of lines to read (default 10)."},
            },
            "required": ["filepath"],
        },
    },
]


# --------------------------------------------------------------------------------------
# JSON-RPC message handling (pure: message in, optional response out)
# --------------------------------------------------------------------------------------
def _error(req_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def _result(req_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def negotiate_version(requested: Any) -> str:
    """Echo the client's version if supported, otherwise offer the newest one this server supports."""
    return requested if requested in SUPPORTED_VERSIONS else SUPPORTED_VERSIONS[0]


def _call_tool(req_id: Any, params: dict) -> dict[str, Any]:
    name = params.get("name")
    if name not in TOOLS:
        return _error(req_id, -32602, f"Unknown tool: {name}")
    arguments = params.get("arguments") or {}
    if not isinstance(arguments, dict):
        return _error(req_id, -32602, "'arguments' must be an object")
    try:
        output = TOOLS[name](arguments)
    except InvalidParams as exc:
        return _error(req_id, -32602, f"Invalid params: {exc}")
    except (OSError, ValueError) as exc:  # a failure while running the tool, reported to the model
        return _result(req_id, {"content": [{"type": "text", "text": f"Error: {exc}"}], "isError": True})
    text = json.dumps(output)
    return _result(req_id, {"content": [{"type": "text", "text": text}], "structuredContent": output, "isError": False})


def handle_message(msg: Any) -> Optional[dict[str, Any]]:
    """Return the response for one decoded JSON-RPC message, or None for a notification."""
    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" or not isinstance(msg.get("method"), str):
        return _error(msg.get("id") if isinstance(msg, dict) else None, -32600, "Invalid Request")
    req_id, method = msg.get("id"), msg["method"]
    params = msg.get("params") or {}
    is_notification = "id" not in msg
    if not isinstance(params, dict):
        return None if is_notification else _error(req_id, -32602, "'params' must be an object")

    try:
        if is_notification:
            log_debug(f"notification: {method}")
            return None
        if method == "initialize":
            return _result(req_id, {
                "protocolVersion": negotiate_version(params.get("protocolVersion")),
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            })
        if method == "ping":
            return _result(req_id, {})
        if method == "tools/list":
            return _result(req_id, {"tools": TOOL_DEFINITIONS})
        if method == "tools/call":
            return _call_tool(req_id, params)
        return _error(req_id, -32601, f"Method not found: {method}")
    except Exception as exc:  # noqa: BLE001 - a request must never go unanswered
        log_debug(f"internal error on {method}: {exc!r}")
        return _error(req_id, -32603, "Internal error")


# --------------------------------------------------------------------------------------
# stdio transport
# --------------------------------------------------------------------------------------
def send_message(message: dict[str, Any]) -> None:
    """Write one JSON-RPC message to stdout. json.dumps without indent never emits a newline."""
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def log_debug(msg: str) -> None:
    """Log to stderr only; anything that is not an MCP message on stdout would corrupt the stream."""
    sys.stderr.write(f"[{SERVER_NAME}] {msg}\n")
    sys.stderr.flush()


def main() -> None:
    log_debug(f"starting on stdio; root={ROOT}")
    for line in sys.stdin:  # ends at EOF, which is how the client shuts the server down
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as err:
            send_message(_error(None, -32700, f"Parse error: {err}"))
            continue
        response = handle_message(msg)
        if response is not None:
            send_message(response)
    log_debug("stdin closed; exiting")


if __name__ == "__main__":
    main()
