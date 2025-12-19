"""Smoke tests for the MCP server shim."""
from __future__ import annotations

import importlib


def test_mcp_server_importable():
    module = importlib.import_module("mcp.dml_mcp_server")
    assert hasattr(module, "create_server")
    if getattr(module, "MCP_AVAILABLE", False):
        server = module.create_server(host="127.0.0.1", port=8765)
        assert server.name == "daystrom-dml"
