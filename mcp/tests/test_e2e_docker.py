"""End-to-end run of the workflow through the stdio protocol against Docker.

Opt-in because it builds the image and runs HATCHet: set ``HATCHET_MCP_E2E=1`` and have a
running Docker daemon. The server is launched as a subprocess exactly as an MCP client
launches it, the input is simulated inside the image, and every workflow tool is called.
"""

import asyncio
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

pytestmark = pytest.mark.skipif(
    os.environ.get("HATCHET_MCP_E2E") != "1", reason="set HATCHET_MCP_E2E=1 to run"
)

REPO = Path(__file__).resolve().parents[2]
TIMEOUT_S = 3600


async def _call(client, name, **kwargs):
    result = await client.call_tool(name, kwargs, timeout=600)
    return result


def _report(result):
    """Structured report of check_job or show_results; images arrive as separate blocks."""
    return result.structured_content


async def _wait(client, job_id):
    for _ in range(TIMEOUT_S // 300):
        result = await _call(client, "check_job", job_id=job_id, wait_seconds=300)
        report = _report(result)
        if report["state"] != "running":
            return result, report
    raise TimeoutError(job_id)


def _simulate(image, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "-e",
            "HOME=/tmp",
            "-v",
            f"{out_dir}:/out",
            image,
            "python",
            "-c",
            "from hatchet.check.simulate import simulate_bb_dir; simulate_bb_dir('/out')",
        ],
        check=True,
        capture_output=True,
    )
    return out_dir / "bb_dir"


def test_full_workflow(tmp_path):
    env = {**os.environ, "HATCHET_MCP_CONFIG": str(tmp_path / "config.json")}
    exe = shutil.which("hatchet-mcp") or str(Path(sys.executable).with_name("hatchet-mcp"))
    transport = StdioTransport(command=exe, args=[], env=env)

    async def run():
        async with Client(transport) as client:
            status = (await _call(client, "setup_status")).structured_content
            assert not status["configured"] and status["config"]["runtime"] == "docker"

            configured = (
                await _call(
                    client,
                    "configure",
                    work_dir=str(tmp_path / "work"),
                    tmp_dir=str(tmp_path / "scratch"),
                    hatchet_local_source=str(REPO),
                    gurobi_license="none",
                )
            ).structured_content
            assert configured["solver"] == "cbc"

            build = (await _call(client, "build_container")).structured_content
            _, report = await _wait(client, build["job_id"])
            assert report["state"] == "succeeded", report.get("log_tail")
            assert report["hatchet_check"]["passed"] and report["hatchet_check"]["n_fail"] == 0

            bb = _simulate("hatchet-mcp:latest", tmp_path / "sim")
            inspected = (
                await _call(client, "inspect_input", bb_dir=str(bb), genome="hg38")
            ).structured_content
            assert inspected["valid"], inspected["errors"]

            started = (
                await _call(
                    client,
                    "start_analysis",
                    bb_dir=str(bb),
                    genome="hg38",
                    sample_id="SIM",
                    min_k=3,
                    max_k=5,
                    restarts=3,
                    max_tumor_clones=1,
                    ploidy="diploid",
                    cores=4,
                    compute_cn_extra={"reg_steps": 3},
                )
            ).structured_content
            assert started["started"]
            result, report = await _wait(client, started["job_id"])
            assert report["state"] == "succeeded", report.get("log_tail")
            assert report["results"]["selected_k"] >= 3
            assert any(c.type == "image" for c in result.content)

            cont = (
                await _call(client, "continue_analysis", run_id=started["run_id"])
            ).structured_content
            result, report = await _wait(client, cont["job_id"])
            assert report["state"] == "succeeded", report.get("log_tail")
            cn = report["results"]
            assert cn["selected_ploidy"] in ("diploid", "tetraploid")
            assert cn["n_tumor_clones"] >= 1
            assert abs(next(iter(cn["samples"].values()))["purity"] - 0.8) < 0.15
            assert sum(c.type == "image" for c in result.content) >= 3

            run_dir = tmp_path / "work" / "runs" / started["run_id"]
            preview_root = tmp_path / "scratch" / "previews" / started["run_id"]
            assert not list(run_dir.rglob("*.png")), "previews leaked into the run directory"
            assert list(preview_root.rglob("*.png"))
            inline = [f for f in report["figures"] if f["inline"]]
            assert inline and all(Path(f["pdf"]).is_file() for f in inline)
            assert all(p.startswith(str(preview_root)) for f in inline for p in f["preview_png"])

            runs = (await _call(client, "list_runs")).structured_content["result"]
            assert runs[0]["run_id"] == started["run_id"]

    asyncio.run(run())
