from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# MCP服务器配置
@dataclass(frozen=True)
class MCPServerSpec:
    name: str
    command: str
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    transport: str = "stdio"

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "MCPServerSpec | None":
        name = str(raw.get("name") or raw.get("server_name") or raw.get("id") or "").strip()
        command = str(raw.get("command") or raw.get("exec") or "").strip()
        if not name or not command:
            return None

        args = raw.get("args") or []
        if isinstance(args, str):
            args = [args]

        env = raw.get("env") or {}
        if not isinstance(env, dict):
            env = {}

        cwd = raw.get("cwd") or None
        transport = str(raw.get("transport") or "stdio").strip().lower()

        return cls(
            name=name,
            command=command,
            args=tuple(str(arg) for arg in args),
            env={str(key): str(value) for key, value in env.items()},
            cwd=str(cwd) if cwd else None,
            transport=transport,
        )


@dataclass(frozen=True)
class MCPToolInfo:
    server_name: str
    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)

    @property
    def qualified_name(self) -> str:
        return f"{self.server_name}.{self.name}"


def _load_json_from_path(path_text: str) -> Any:
    path = Path(path_text).expanduser()
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _default_config_candidates() -> list[Path]:
    base_dir = Path(__file__).resolve().parent.parent
    return [
        base_dir / "mcp_servers.json",
        base_dir / "config" / "mcp_servers.json",
    ]


def _load_server_spec_items() -> list[dict[str, Any]]:
    config_path = os.getenv("MCP_SERVER_CONFIG_PATH", "").strip()
    config_json = os.getenv("MCP_SERVERS_JSON", "").strip()

    raw: Any = None
    if config_path:
        try:
            raw = _load_json_from_path(config_path)
        except (OSError, json.JSONDecodeError):
            return []
    elif config_json:
        try:
            raw = json.loads(config_json)
        except json.JSONDecodeError:
            try:
                raw = _load_json_from_path(config_json)
            except (OSError, json.JSONDecodeError):
                return []
    else:
        for candidate in _default_config_candidates():
            if not candidate.exists():
                continue
            try:
                raw = _load_json_from_path(str(candidate))
                break
            except (OSError, json.JSONDecodeError):
                continue

    if raw is None:
        return []
    if isinstance(raw, dict) and isinstance(raw.get("servers"), list):
        return [item for item in raw["servers"] if isinstance(item, dict)]
    if isinstance(raw, dict) and ("name" in raw or "command" in raw):
        return [raw]
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    return []


def _require_mcp_sdk():
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError as exc:  # pragma: no cover - runtime dependency guard
        raise RuntimeError(
            "MCP SDK is not installed. Add `mcp` to requirements and install dependencies first."
        ) from exc

    return ClientSession, StdioServerParameters, stdio_client


def _tool_schema_text(schema: dict[str, Any]) -> str:
    if not schema:
        return "{}"
    try:
        return json.dumps(schema, ensure_ascii=False, indent=2)
    except TypeError:
        return str(schema)


def _stringify_tool_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, list):
        parts = [_stringify_tool_content(item) for item in content]
        return "\n".join(part for part in parts if part)
    text = getattr(content, "text", None)
    if text is not None:
        return str(text)
    data = getattr(content, "model_dump", None)
    if callable(data):
        try:
            return json.dumps(data(), ensure_ascii=False)
        except TypeError:
            pass
    return str(content)

# MCP客户端管理器
class MCPClientManager:
    def __init__(self, servers: list[MCPServerSpec] | None = None):
        self.servers = servers or []
        self._tool_cache: list[MCPToolInfo] | None = None

    @classmethod
    def from_environment(cls) -> "MCPClientManager":
        specs = []
        for item in _load_server_spec_items():
            spec = MCPServerSpec.from_mapping(item)
            if spec is not None:
                specs.append(spec)
        return cls(specs)

    def has_servers(self) -> bool:
        return bool(self.servers)

    def _server_map(self) -> dict[str, MCPServerSpec]:
        return {server.name: server for server in self.servers}

    async def list_tools(self, refresh: bool = False) -> list[MCPToolInfo]:
        if self._tool_cache is not None and not refresh:
            return list(self._tool_cache)

        if not self.servers:
            self._tool_cache = []
            return []

        ClientSession, StdioServerParameters, stdio_client = _require_mcp_sdk()
        tools: list[MCPToolInfo] = []

        for server in self.servers:
            if server.transport != "stdio":
                continue

            params = StdioServerParameters(
                command=server.command,
                args=list(server.args),
                env=server.env or None,
                cwd=server.cwd,
            )
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.list_tools()
                    for tool in getattr(result, "tools", []) or []:
                        input_schema = (
                            getattr(tool, "inputSchema", None)
                            or getattr(tool, "input_schema", None)
                            or {}
                        )
                        tools.append(
                            MCPToolInfo(
                                server_name=server.name,
                                name=str(getattr(tool, "name", "")).strip(),
                                description=str(getattr(tool, "description", "") or "").strip(),
                                input_schema=dict(input_schema) if isinstance(input_schema, dict) else {},
                            )
                        )

        self._tool_cache = tools
        return list(tools)

    def list_tools_sync(self, refresh: bool = False) -> list[MCPToolInfo]:
        return asyncio.run(self.list_tools(refresh=refresh))

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict[str, Any] | None = None) -> str:
        servers = self._server_map()
        if server_name not in servers:
            raise KeyError(f"Unknown MCP server: {server_name}")

        server = servers[server_name]
        if server.transport != "stdio":
            raise NotImplementedError(f"Unsupported MCP transport: {server.transport}")

        ClientSession, StdioServerParameters, stdio_client = _require_mcp_sdk()
        params = StdioServerParameters(
            command=server.command,
            args=list(server.args),
            env=server.env or None,
            cwd=server.cwd,
        )

        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments or {})
                text = _stringify_tool_content(getattr(result, "content", result))
                if not text:
                    text = json.dumps(
                        {
                            "server": server_name,
                            "tool": tool_name,
                            "result": getattr(result, "model_dump", lambda: str(result))(),
                        },
                        ensure_ascii=False,
                    )
                return text

    def call_tool_sync(self, server_name: str, tool_name: str, arguments: dict[str, Any] | None = None) -> str:
        return asyncio.run(self.call_tool(server_name, tool_name, arguments))

    def describe_tools_text(self, refresh: bool = False) -> str:
        try:
            tools = self.list_tools_sync(refresh=refresh)
        except Exception as exc:
            return f"当前无法加载外部 MCP 工具: {exc}"
        if not tools:
            return "当前没有配置可用的外部 MCP 工具。"

        lines = ["可用的外部 MCP 工具："]
        for tool in tools:
            lines.append(
                f"- {tool.qualified_name}: {tool.description or '无描述'}\n"
                f"  入参: {_tool_schema_text(tool.input_schema)}"
            )
        return "\n".join(lines)
