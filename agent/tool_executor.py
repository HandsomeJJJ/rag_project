from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from agent.mcp_client import MCPClientManager

# MCP工具执行器
class MCPToolExecutor:
    def __init__(self, client_manager: MCPClientManager):
        self.client_manager = client_manager

    def has_tools(self) -> bool:
        return self.client_manager.has_servers()

    def describe_tools_text(self, refresh: bool = False) -> str:
        return self.client_manager.describe_tools_text(refresh=refresh)

    def execute(self, server_name: str, tool_name: str, arguments: dict[str, Any] | None = None) -> str:
        return self.client_manager.call_tool_sync(server_name, tool_name, arguments or {})

# 构建MCP工具
def build_mcp_tool(executor: MCPToolExecutor):
    @tool("mcp_call_tool")
    def mcp_call_tool(
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> str:
        """Call one configured MCP tool by server name and tool name."""
        return executor.execute(server_name, tool_name, arguments or {})

    return mcp_call_tool

