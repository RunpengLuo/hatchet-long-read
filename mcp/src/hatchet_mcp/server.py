"""hatchet-mcp tools.

The server never imports HATCHet. Each tool writes a bash script that invokes a
``hatchet`` subcommand and runs it inside the container (or, for the ``local`` runtime,
against an existing install), so the server itself only needs ``fastmcp``.
"""

import asyncio
import json
import re
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from fastmcp import Context, FastMCP
from fastmcp.tools import ToolResult
from fastmcp.utilities.types import Image
from mcp.types import TextContent

from hatchet_mcp import __version__, results, runtime, scripts
from hatchet_mcp.config import (
    GENOMES,
    RUNTIMES,
    Config,
    choose_solver,
    config_path,
    container_context_dir,
    detect_runtimes,
    inspect_gurobi_license,
    load_config,
    reference_dir,
    save_config,
)
from hatchet_mcp.inputs import inspect_bb_dir

MAX_WAIT_S = 540
POLL_S = 5
Genome = Literal["hg38", "hg19", "chm13"]

INSTRUCTIONS = """\
hatchet-mcp runs HATCHet (allele- and clone-specific copy-number inference from
multi-sample bulk tumor sequencing) inside a container. Drive the user through these
steps in order, one at a time, and stop to ask where indicated.

1. Call setup_status first, every session. If the setup is not configured, show the
   proposed runtime, work_dir and tmp_dir, then ASK the user: (a) the path to their
   Gurobi licence file, or "none" to use the open-source CBC solver; (b) whether to
   change the work or tmp directories.
2. Call configure with the answers. Relay the licence note: a node-locked licence
   cannot validate inside a container, so CBC will be used.
3. If the image is missing or out of date, call build_container, then poll check_job
   until it finishes (the first build takes several minutes) and report the
   hatchet check result.
4. ASK the user for the analysis: path to the bb_dir, genome build, a sample id, the
   expected range of TUMOR clones, ploidy (diploid, tetraploid meaning whole-genome
   duplication, or both), CPU cores, memory and a time limit. Anything left unset keeps
   HATCHet's default.
5. Call inspect_input and show the stats. Stop and explain if the input is invalid.
6. Call start_analysis to run cluster-bins, then poll check_job. When it succeeds, show
   the returned figures (model selection, selected clustering, QC) and summarise the
   selected K and the clusters.
7. Call continue_analysis to run compute-cn and plot-cn, then poll check_job. When it
   succeeds, show the model-selection result, report the selected ploidy, number of
   tumor clones, purity and clone proportions per sample, and show the 1D and 2D
   copy-number profiles.

Figures: the user sees HATCHet's PDFs. When a stage finishes, call open_figure with the
PDFs of its headline figures (those with "inline": true) to open them in the user's viewer,
and list every figure's PDF path as a link. The PNG previews attached to results are for
you to read and describe; do not present PNG paths to the user. HATCHet's figure titles
count the normal clone (n=2 means one tumor clone); report tumor clones to the user.

Jobs keep running if the session ends; list_runs and check_job recover them later.
"""

mcp = FastMCP(name="hatchet-mcp", instructions=INSTRUCTIONS)


def _host_paths(config: Config, job_dir: Path, bb_dir: str | None = None) -> dict:
    config.tmp_path.mkdir(parents=True, exist_ok=True)
    host = {"job": str(job_dir), "tmp": str(config.tmp_path)}
    if bb_dir:
        host["bb"] = str(Path(bb_dir).expanduser().resolve())
    return host


def _inner_paths(config: Config, host: dict) -> scripts.Paths:
    if config.runtime == "local":
        return scripts.Paths(
            bb=host.get("bb", ""), job=host["job"], ref=reference_dir(config), tmp=host["tmp"]
        )
    return scripts.CONTAINER_PATHS


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)
    return path


def _image_status(config: Config) -> dict:
    if config.runtime == "local":
        prefix = Path(config.local_env_prefix or "")
        ok = (prefix / "bin" / "hatchet").exists()
        return {"runtime": "local", "hatchet_found": ok, "ready": ok}
    labels = runtime.image_labels(config) if config.runtime else None
    source = runtime.source_commit(config)
    built = (labels or {}).get("hatchet.commit")
    return {
        "image": config.apptainer_image
        if config.runtime in ("apptainer", "singularity")
        else config.image,
        "exists": labels is not None,
        "built_from": built,
        "source_now": source,
        "up_to_date": labels is not None and source is not None and built == source,
        "ready": labels is not None,
    }


def _require_ready() -> Config:
    config = load_config()
    if not config.configured:
        raise RuntimeError(
            "hatchet-mcp is not configured yet; call setup_status and configure first"
        )
    if not _image_status(config)["ready"]:
        raise RuntimeError("the HATCHet container is not built yet; call build_container first")
    return config


def _preview_root(config: Config, run_id: str) -> str:
    return str(config.tmp_path / "previews" / run_id)


def _run_dir(config: Config, run_id: str) -> Path:
    path = config.work_path / "runs" / run_id
    if not path.is_dir():
        raise FileNotFoundError(f"no run '{run_id}' under {config.work_path / 'runs'}")
    return path


def _load_run(run_dir: Path) -> dict:
    return json.loads((run_dir / "run.json").read_text())


def _save_run(run_dir: Path, record: dict):
    (run_dir / "run.json").write_text(json.dumps(record, indent=2) + "\n")


def _launch_stage(
    config: Config, run_dir: Path, record: dict, kind: str, params: scripts.AnalysisParams
) -> runtime.Job:
    stage = kind.replace("-", "_")
    job_id = runtime.new_job_id(kind)
    status = run_dir / ".status" / f"{stage}.exit"
    status.parent.mkdir(parents=True, exist_ok=True)
    host = _host_paths(config, run_dir, record["bb_dir"])
    launcher = _write(
        run_dir / "scripts" / f"{stage}.launch.sh",
        scripts.render_launch(
            config,
            host,
            f"{stage}.sh",
            f"hatchet-mcp-{job_id}",
            str(status),
            cores=params.cores,
            memory_gb=params.memory_gb,
        ),
    )
    job = runtime.Job(
        job_id=job_id,
        kind=kind,
        run_id=record["run_id"],
        launch_script=str(launcher),
        log=str(run_dir / "logs" / f"{stage}.log"),
        status_file=str(status),
        container_name=f"hatchet-mcp-{job_id}",
        runtime=config.runtime,
    )
    runtime.start_job(config, job)
    record.setdefault("stages", {})[kind] = {"job_id": job_id, "started": job.started}
    _save_run(run_dir, record)
    return job


def _result(report: dict, figures: list[dict]) -> ToolResult:
    """Report as structured content plus one text block and one image block per figure.

    The shape is the same whether or not figures exist, so clients can always read the
    report from ``structured_content`` and agents always see it as the first text block.
    Only headline figures carry previews, which are for the agent; the user is shown
    the PDFs listed in the report.
    """
    content = [TextContent(type="text", text=json.dumps(report, indent=1, default=str))]
    content += [
        Image(path=png).to_image_content() for f in figures for png in f.get("preview_png", [])
    ]
    return ToolResult(content=content, structured_content=report)


@mcp.tool
def setup_status() -> dict:
    """Report the current setup and what the user still needs to decide.

    Call this first in every session. It detects installed container runtimes (Docker,
    Podman, Apptainer, Singularity), shows the stored or proposed configuration
    including the work and scratch directories, classifies the configured Gurobi
    licence, and reports whether the HATCHet image exists and matches its source.
    """
    config = load_config()
    runtimes = detect_runtimes()
    licence = inspect_gurobi_license(config.gurobi_license)
    image = _image_status(config) if config.runtime else {"ready": False}
    if not any(r["ready"] for r in runtimes) and config.runtime != "local":
        next_step = "No container runtime is ready. Install or start Docker (or Podman/Apptainer)."
    elif not config.configured:
        next_step = "Ask the user for the Gurobi licence path (or none) and any directory changes, then call configure."
    elif not image["ready"]:
        next_step = "Call build_container to build the HATCHet image and run hatchet check."
    elif image.get("up_to_date") is False and image.get("source_now"):
        next_step = "The image is behind its source; offer build_container to update, or proceed with the current image."
    else:
        next_step = "Ready. Ask the user for the analysis inputs, then call inspect_input."
    return {
        "hatchet_mcp_version": __version__,
        "configured": config.configured,
        "config_file": str(config_path()),
        "config": asdict(config),
        "runtimes": runtimes,
        "gurobi_license": licence,
        "image": image,
        "genomes_available": sorted(GENOMES),
        "next_step": next_step,
    }


@mcp.tool
def configure(
    gurobi_license: Annotated[
        str | None, "Path to a Gurobi licence file, 'none' to clear it and use CBC, or omit to keep"
    ] = None,
    work_dir: Annotated[
        str | None, "Root directory for runs, scripts, logs and job records"
    ] = None,
    tmp_dir: Annotated[
        str | None, "Scratch directory mounted into the container as HOME and TMPDIR"
    ] = None,
    runtime_name: Annotated[
        Literal["docker", "podman", "apptainer", "singularity", "local"] | None,
        "Container runtime; 'local' runs an existing HATCHet install on the host",
    ] = None,
    solver: Annotated[
        Literal["auto", "cbc", "gurobi"],
        "ILP solver; auto picks gurobi only if the licence works where HATCHet runs",
    ] = "auto",
    hatchet_repo: Annotated[str | None, "Git repository the image builds HATCHet from"] = None,
    hatchet_ref: Annotated[str | None, "Branch or tag of the repository to build"] = None,
    hatchet_local_source: Annotated[
        str | None, "Local HATCHet checkout to build from instead of cloning, or 'none' to clone"
    ] = None,
    local_env_prefix: Annotated[
        str | None, "Conda prefix with HATCHet installed, for runtime 'local'"
    ] = None,
    apptainer_image: Annotated[
        str | None, "Path of the .sif image, for apptainer/singularity"
    ] = None,
    registry_image: Annotated[str | None, "Registry image an apptainer .sif is built from"] = None,
) -> dict:
    """Save the configuration the user chose; omitted values keep their current setting.

    Creates the work and scratch directories, classifies the Gurobi licence by its key
    names only (values are never read out), and selects the solver.
    """
    config = load_config()
    notes = []
    if runtime_name is not None:
        config.runtime = runtime_name
    if config.runtime is None:
        raise RuntimeError("no container runtime available; install Docker, Podman or Apptainer")
    if config.runtime not in RUNTIMES:
        raise ValueError(f"runtime must be one of {RUNTIMES}")
    if config.runtime != "local":
        ready = {r["name"]: r for r in detect_runtimes()}
        if config.runtime not in ready:
            raise RuntimeError(f"{config.runtime} is not installed")
        if not ready[config.runtime]["ready"]:
            notes.append(
                f"{config.runtime} is installed but not running: {ready[config.runtime]['detail']}"
            )

    paths = {
        "work_dir": work_dir,
        "tmp_dir": tmp_dir,
        "local_env_prefix": local_env_prefix,
        "apptainer_image": apptainer_image,
    }
    for attr, value in paths.items():
        if value is not None:
            setattr(config, attr, str(Path(value).expanduser()))
    strings = {
        "hatchet_repo": hatchet_repo,
        "hatchet_ref": hatchet_ref,
        "registry_image": registry_image,
    }
    for attr, value in strings.items():
        if value is not None:
            setattr(config, attr, value)
    if gurobi_license is not None:
        config.gurobi_license = (
            None
            if gurobi_license.lower() == "none"
            else str(Path(gurobi_license).expanduser().resolve())
        )
    if hatchet_local_source is not None:
        config.hatchet_local_source = (
            None
            if hatchet_local_source.lower() == "none"
            else str(Path(hatchet_local_source).expanduser().resolve())
        )
    if config.runtime == "local":
        if not (Path(config.local_env_prefix or "") / "bin" / "hatchet").exists():
            raise ValueError(
                "runtime 'local' needs local_env_prefix pointing at a conda env with HATCHet installed"
            )
        if not config.hatchet_local_source:
            raise ValueError(
                "runtime 'local' needs hatchet_local_source, whose data/reference HATCHet uses"
            )

    licence = inspect_gurobi_license(config.gurobi_license)
    auto = choose_solver(licence, config.runtime)
    if solver == "auto":
        config.solver = auto
    else:
        config.solver = solver
        if solver == "gurobi" and auto != "gurobi":
            notes.append(
                "gurobi requested but the licence will not validate where HATCHet runs; compute-cn may fail"
            )

    config.work_path.mkdir(parents=True, exist_ok=True)
    config.tmp_path.mkdir(parents=True, exist_ok=True)
    config.configured = True
    path = save_config(config)
    return {
        "saved_to": str(path),
        "config": asdict(config),
        "gurobi_license": licence,
        "solver": config.solver,
        "notes": notes,
        "next_step": "Call build_container."
        if not _image_status(config)["ready"]
        else "Setup complete; ask for analysis inputs.",
    }


@mcp.tool
def build_container() -> dict:
    """Build the HATCHet image and verify it with ``hatchet check``, in the background.

    Docker/Podman: fetches HATCHet (or stages the configured local checkout), creates the
    conda environment from HATCHet's environment.yaml, pip-installs HATCHet, then runs
    ``hatchet check`` inside the new image. Apptainer: builds the .sif from
    ``registry_image`` if needed, then runs the check. Local: runs the check only.
    Returns a job id; poll it with check_job.
    """
    config = load_config()
    if not config.configured:
        raise RuntimeError("call configure first")
    setup_dir = config.work_path / "setup"
    job_id = runtime.new_job_id("build")
    status = setup_dir / ".status" / f"{job_id}.exit"
    status.parent.mkdir(parents=True, exist_ok=True)
    host = _host_paths(config, setup_dir)
    _write(
        setup_dir / "scripts" / "check.sh",
        scripts.render_check(_inner_paths(config, host), config.solver),
    )
    name = f"hatchet-mcp-{job_id}"

    if config.runtime in ("docker", "podman"):
        commit = runtime.source_commit(config)
        if commit is None:
            raise RuntimeError(
                f"cannot resolve HATCHet source ({config.hatchet_local_source or config.hatchet_repo + '@' + config.hatchet_ref})"
            )
        check_argv = scripts.container_command(
            config, host, f"{scripts.CONTAINER_PATHS.job}/scripts/check.sh", name
        )
        text = scripts.render_build(
            config,
            str(container_context_dir()),
            str(config.work_path / "build"),
            commit,
            str(status),
            check_argv,
        )
    elif config.runtime in ("apptainer", "singularity"):
        sif = config.apptainer_image or str(config.work_path / "images" / "hatchet-mcp.sif")
        config.apptainer_image = sif
        save_config(config)
        commands = []
        if not Path(sif).exists():
            if not config.registry_image:
                raise RuntimeError(
                    "set registry_image (a published hatchet-mcp image) so the .sif can be built"
                )
            Path(sif).parent.mkdir(parents=True, exist_ok=True)
            commands.append([config.runtime, "build", sif, f"docker://{config.registry_image}"])
        commands.append(
            scripts.container_command(
                config, host, f"{scripts.CONTAINER_PATHS.job}/scripts/check.sh", name
            )
        )
        text = scripts.render_steps(f"prepare {sif} and run hatchet check", str(status), commands)
    else:
        text = scripts.render_launch(config, host, "check.sh", name, str(status))

    launcher = _write(setup_dir / "scripts" / f"{job_id}.sh", text)
    job = runtime.Job(
        job_id=job_id,
        kind="build",
        run_id=None,
        launch_script=str(launcher),
        log=str(setup_dir / "logs" / f"{job_id}.log"),
        status_file=str(status),
        container_name=name,
        runtime=config.runtime,
    )
    runtime.start_job(config, job)
    return {
        "job_id": job_id,
        "script": str(launcher),
        "log": job.log,
        "message": "Build started. Poll check_job with wait_seconds to follow it; a first build takes several minutes.",
    }


@mcp.tool
async def check_job(
    job_id: Annotated[
        str, "Job id returned by build_container, start_analysis or continue_analysis"
    ],
    wait_seconds: Annotated[
        int, f"Block up to this many seconds (max {MAX_WAIT_S}) for the job to finish"
    ] = 0,
    ctx: Context = None,
) -> ToolResult:
    """Report a job's progress; when it has finished, return its results and figures.

    For a finished build this includes the hatchet check assertions. For cluster-bins it
    includes the selected K, per-K model scores, the cluster table and the model-
    selection, selected-clustering and QC figures. For compute-cn it includes the model-
    selection table, selected ploidy, tumor clone count, purity and clone proportions,
    and the 1D and 2D copy-number profiles.
    """
    config = load_config()
    job = runtime.load_job(config, job_id)
    wait = max(0, min(wait_seconds, MAX_WAIT_S))
    waited = 0
    state = runtime.job_state(job)
    while state["state"] == "running" and waited < wait:
        await asyncio.sleep(POLL_S)
        waited += POLL_S
        if ctx is not None:
            await ctx.report_progress(progress=waited, total=wait)
        state = runtime.job_state(job)

    report = {"job_id": job.job_id, "kind": job.kind, "run_id": job.run_id, **state, "log": job.log}
    if state["state"] != "succeeded":
        report["log_tail"] = runtime.tail(job.log, 60 if state["state"] == "failed" else 15)
        if state["state"] == "running":
            report["message"] = "Still running. Call check_job again, optionally with wait_seconds."
        return _result(report, [])

    figures = []
    if job.kind == "build":
        report["hatchet_check"] = results.check_results(Path(job.log).read_text(errors="replace"))
        report["image"] = _image_status(config)
    else:
        run_dir = _run_dir(config, job.run_id)
        record = _load_run(run_dir)
        if job.kind == "cluster-bins":
            summary, figures = results.cluster_bins_results(
                str(run_dir), _preview_root(config, run_dir.name)
            )
            report["next_step"] = (
                "Show these results, then call continue_analysis to run compute-cn."
            )
        else:
            summary, figures = results.compute_cn_results(
                str(run_dir), _preview_root(config, run_dir.name)
            )
            report["next_step"] = "Analysis complete."
        report["results"] = summary
        record["stages"][job.kind].update({"state": "succeeded", "elapsed_s": state["elapsed_s"]})
        _save_run(run_dir, record)
    report["figures"] = figures
    return _result(report, figures)


@mcp.tool
def cancel_job(job_id: Annotated[str, "Job id to stop"]) -> dict:
    """Stop a running job and its container."""
    config = load_config()
    return runtime.cancel_job(runtime.load_job(config, job_id))


@mcp.tool
def inspect_input(
    bb_dir: Annotated[str, "Path to the bb_dir produced by upstream preprocessing"],
    genome: Annotated[
        Genome | None, "Genome build to validate against; omit to report compatible builds"
    ] = None,
) -> dict:
    """Validate a bb_dir against HATCHet's input format and summarise it.

    Checks that every required file is present, that sample_ids.tsv and bb.tsv.gz carry
    the required columns, that the matrix shapes agree with the bin and sample counts,
    and that bin coordinates fit the chosen genome build. Reports the number of bins,
    chromosomes and regions, SNP coverage, tumor and normal samples, and which genome
    builds the coordinates are compatible with.
    """
    return inspect_bb_dir(bb_dir, genome, runtime.read_chrom_sizes(load_config()))


@mcp.tool
def start_analysis(
    bb_dir: Annotated[str, "Path to the bb_dir"],
    genome: Annotated[Genome, "Genome build of the bin coordinates"],
    sample_id: Annotated[str, "Patient identifier used to label outputs"],
    min_tumor_clones: Annotated[
        int, "Fewest TUMOR clones to fit (the normal clone is added automatically)"
    ] = 1,
    max_tumor_clones: Annotated[
        int, "Most TUMOR clones to fit; consider more when several tumor samples exist"
    ] = 3,
    ploidy: Annotated[
        Literal["diploid", "tetraploid", "both"],
        "Hypotheses to test; tetraploid means whole-genome duplication",
    ] = "both",
    cores: Annotated[int, "CPU cores for the container and HATCHet's worker pools"] = 4,
    memory_gb: Annotated[int | None, "Container memory cap in GB; omit for no cap"] = None,
    time_limit_hours: Annotated[
        float | None, "Wall-clock limit per stage in hours; omit for none"
    ] = None,
    min_k: Annotated[
        int | None, "Smallest K tried by cluster-bins; omit for HATCHet's default"
    ] = None,
    max_k: Annotated[
        int | None, "Largest K tried by cluster-bins; omit for HATCHet's default (30, slow)"
    ] = None,
    restarts: Annotated[int | None, "EM restarts per K; omit for HATCHet's default"] = None,
    solver_time_limit_s: Annotated[int | None, "Per-solve compute-cn time limit in seconds"] = None,
    cluster_bins_extra: Annotated[dict | None, "Extra cluster-bins flags, e.g. {'seed': 7}"] = None,
    compute_cn_extra: Annotated[
        dict | None,
        "Extra compute-cn flags. Runtime scales with ploidies x clone counts x reg_steps "
        "(default 15) x cd_nseeds (default 400); {'reg_steps': 3} gives a quick first pass",
    ] = None,
) -> dict:
    """Validate the input, write both stage scripts, and start cluster-bins.

    Parameters are frozen into run.json and into the generated scripts under
    ``<work_dir>/runs/<run_id>/scripts/`` so the run can be inspected or repeated. Only
    values the user set are passed to HATCHet. Returns the run id and the job id; poll
    the job with check_job.
    """
    config = _require_ready()
    report = inspect_bb_dir(bb_dir, genome, runtime.read_chrom_sizes(config))
    if not report["valid"]:
        return {"started": False, "reason": "input failed validation", "input": report}

    params = scripts.AnalysisParams(
        sample_id=sample_id,
        genome=genome,
        min_k=min_k,
        max_k=max_k,
        restarts=restarts,
        min_tumor_clones=min_tumor_clones,
        max_tumor_clones=max_tumor_clones,
        ploidy=ploidy,
        cores=cores,
        memory_gb=memory_gb,
        time_limit_hours=time_limit_hours,
        solver=config.solver,
        solver_time_limit_s=solver_time_limit_s,
        cluster_bins_extra=cluster_bins_extra or {},
        compute_cn_extra=compute_cn_extra or {},
    )
    params.validate()

    safe = re.sub(r"[^A-Za-z0-9._-]", "_", sample_id)
    run_id = f"{safe}-{datetime.now():%Y%m%d-%H%M%S}"
    run_dir = config.work_path / "runs" / run_id
    host = _host_paths(config, run_dir, bb_dir)
    inner = _inner_paths(config, host)
    _write(
        run_dir / "scripts" / "cluster_bins.sh", scripts.render_cluster_bins(inner, params, run_id)
    )
    _write(run_dir / "scripts" / "compute_cn.sh", scripts.render_compute_cn(inner, params, run_id))
    record = {
        "run_id": run_id,
        "bb_dir": host["bb"],
        "created": datetime.now().isoformat(timespec="seconds"),
        "runtime": config.runtime,
        "image": config.image,
        "params": asdict(params),
        "hatchet_flags": {"minClone": params.min_clone_flag, "maxClone": params.max_clone_flag},
        "input": report,
        "stages": {},
    }
    _save_run(run_dir, record)
    job = _launch_stage(config, run_dir, record, "cluster-bins", params)
    return {
        "started": True,
        "run_id": run_id,
        "job_id": job.job_id,
        "run_dir": str(run_dir),
        "scripts": sorted(str(p) for p in (run_dir / "scripts").glob("*.sh")),
        "input": {k: report[k] for k in ("samples", "bins", "genome", "warnings")},
        "clone_translation": (
            f"{min_tumor_clones}-{max_tumor_clones} tumor clones -> HATCHet --minClone "
            f"{params.min_clone_flag} --maxClone {params.max_clone_flag} (HATCHet counts the normal clone)"
        ),
        "message": "cluster-bins started. Poll check_job with wait_seconds until it finishes.",
    }


@mcp.tool
def continue_analysis(
    run_id: Annotated[str, "Run id returned by start_analysis"],
    min_tumor_clones: Annotated[int | None, "Override the fewest tumor clones"] = None,
    max_tumor_clones: Annotated[int | None, "Override the most tumor clones"] = None,
    ploidy: Annotated[
        Literal["diploid", "tetraploid", "both"] | None, "Override the ploidy hypotheses"
    ] = None,
    cores: Annotated[int | None, "Override CPU cores"] = None,
    memory_gb: Annotated[int | None, "Override the memory cap in GB"] = None,
    time_limit_hours: Annotated[float | None, "Override the wall-clock limit in hours"] = None,
    solver_time_limit_s: Annotated[
        int | None, "Override the per-solve time limit in seconds"
    ] = None,
) -> dict:
    """Start compute-cn and plot-cn on a run whose cluster-bins stage has succeeded.

    Overrides regenerate the compute-cn script before launching. Returns the job id;
    poll it with check_job.
    """
    config = _require_ready()
    run_dir = _run_dir(config, run_id)
    record = _load_run(run_dir)
    if not (run_dir / "cluster-bins" / "bulk.seg").exists():
        raise RuntimeError(
            "cluster-bins has not produced bulk.seg for this run yet; check its job first"
        )
    params = scripts.AnalysisParams(**record["params"])
    for attr, value in (
        ("min_tumor_clones", min_tumor_clones),
        ("max_tumor_clones", max_tumor_clones),
        ("ploidy", ploidy),
        ("cores", cores),
        ("memory_gb", memory_gb),
        ("time_limit_hours", time_limit_hours),
        ("solver_time_limit_s", solver_time_limit_s),
    ):
        if value is not None:
            setattr(params, attr, value)
    params.solver = config.solver
    params.validate()
    host = _host_paths(config, run_dir, record["bb_dir"])
    _write(
        run_dir / "scripts" / "compute_cn.sh",
        scripts.render_compute_cn(_inner_paths(config, host), params, run_id),
    )
    record["params"] = asdict(params)
    record["hatchet_flags"] = {"minClone": params.min_clone_flag, "maxClone": params.max_clone_flag}
    job = _launch_stage(config, run_dir, record, "compute-cn", params)
    return {
        "started": True,
        "run_id": run_id,
        "job_id": job.job_id,
        "script": str(run_dir / "scripts" / "compute_cn.sh"),
        "hatchet_flags": record["hatchet_flags"],
        "message": "compute-cn started. Poll check_job with wait_seconds until it finishes.",
    }


@mcp.tool
def show_results(
    run_id: Annotated[str, "Run id"],
    stage: Annotated[Literal["cluster-bins", "compute-cn"], "Stage whose results to show"],
) -> ToolResult:
    """Show the summary and figures of a finished stage again."""
    config = load_config()
    run_dir = _run_dir(config, run_id)
    if stage == "cluster-bins":
        summary, figures = results.cluster_bins_results(
            str(run_dir), _preview_root(config, run_dir.name)
        )
    else:
        summary, figures = results.compute_cn_results(
            str(run_dir), _preview_root(config, run_dir.name)
        )
    return _result(
        {"run_id": run_id, "stage": stage, "results": summary, "figures": figures}, figures
    )


@mcp.tool
def open_figure(
    pdf_paths: Annotated[list[str], "PDF paths from a stage result's figures"],
) -> dict:
    """Open HATCHet PDF figures in the user's default viewer.

    Only PDFs inside the configured work directory can be opened. Uses ``open`` on macOS
    and ``xdg-open`` on Linux; elsewhere the paths are returned for the user to open.
    """
    config = load_config()
    root = config.work_path
    valid, rejected = [], []
    for raw in pdf_paths:
        path = Path(raw).expanduser().resolve()
        inside = path.is_relative_to(root)
        (valid if inside and path.suffix == ".pdf" and path.is_file() else rejected).append(
            str(path)
        )
    opener = {"darwin": ["open"], "linux": ["xdg-open"]}.get(sys.platform)
    opened, failed = [], []
    if opener and valid:
        launches = [opener + valid] if sys.platform == "darwin" else [opener + [p] for p in valid]
        for argv in launches:
            ok = subprocess.run(argv, capture_output=True).returncode == 0
            (opened if ok else failed).extend(argv[len(opener) :])
    return {
        "opened": opened,
        "not_opened": failed or ([] if opener else valid),
        "rejected": rejected,
        "note": "rejected paths are not PDFs inside the work directory" if rejected else "",
    }


@mcp.tool
def list_runs() -> list[dict]:
    """List analysis runs and the state of their stages, newest first."""
    config = load_config()
    root = config.work_path / "runs"
    out = []
    for run_dir in sorted(root.glob("*"), reverse=True) if root.exists() else []:
        if not (run_dir / "run.json").exists():
            continue
        record = _load_run(run_dir)
        stages = {}
        for kind, info in record.get("stages", {}).items():
            try:
                stages[kind] = {
                    "job_id": info["job_id"],
                    **runtime.job_state(runtime.load_job(config, info["job_id"])),
                }
            except FileNotFoundError:
                stages[kind] = info
        out.append(
            {
                "run_id": record["run_id"],
                "bb_dir": record["bb_dir"],
                "created": record["created"],
                "stages": stages,
            }
        )
    return out
