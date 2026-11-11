"""Server tools that need no container: figure opening and its path restriction."""

import asyncio

from fastmcp import Client

from hatchet_mcp import server
from hatchet_mcp.config import Config, save_config


def test_open_figure_only_opens_pdfs_inside_work_dir(tmp_path, monkeypatch):
    work = tmp_path / "work"
    inside = work / "runs" / "r1" / "plot-cn" / "S.1D.pdf"
    inside.parent.mkdir(parents=True)
    inside.write_bytes(b"%PDF")
    outside = tmp_path / "elsewhere.pdf"
    outside.write_bytes(b"%PDF")
    not_pdf = work / "runs" / "r1" / "run.json"
    not_pdf.write_text("{}")
    save_config(Config(runtime="docker", work_dir=str(work), configured=True))

    launched = []

    class Done:
        returncode = 0

    monkeypatch.setattr(
        server.subprocess, "run", lambda argv, **kw: launched.append(argv) or Done()
    )
    monkeypatch.setattr(server.sys, "platform", "darwin")

    async def call():
        async with Client(server.mcp) as client:
            paths = [str(inside), str(outside), str(not_pdf)]
            return (await client.call_tool("open_figure", {"pdf_paths": paths})).structured_content

    result = asyncio.run(call())
    assert result["opened"] == [str(inside.resolve())]
    assert sorted(result["rejected"]) == sorted([str(outside.resolve()), str(not_pdf.resolve())])
    assert launched == [["open", str(inside.resolve())]]
