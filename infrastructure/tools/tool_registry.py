from __future__ import annotations
from typing import Optional

from application.ports.tool_port import ToolPort, ToolRegistryPort


class ToolRegistry(ToolRegistryPort):
    def __init__(self, tools: list[ToolPort]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    def list_tools(self) -> list[ToolPort]:
        return list(self._tools.values())

    def get_tool(self, name: str) -> Optional[ToolPort]:
        return self._tools.get(name)
