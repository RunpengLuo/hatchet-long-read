"""Generate the bash scripts that run HATCHet subcommands.

Each stage is written as two scripts, both kept on disk so a run can be inspected or
repeated by hand:

- an inner script that calls ``hatchet <subcommand>``; it sees container paths (or host
  paths for the ``local`` runtime) and renders each PDF figure to PNG previews under
  ``<tmp>/previews/<run_id>/``, keeping the run directory to HATCHet's own outputs;
- a host launcher that wraps the inner script in ``docker run`` / ``podman run`` /
  ``apptainer exec``, or runs it directly for ``local``, and records the exit code.

Only flags whose value the user set are emitted, so every other parameter keeps the
default from HATCHet's ``hatchet.yaml``.
"""

import os
import shlex
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from hatchet_mcp import __version__
from hatchet_mcp.config import GENOMES, IMAGE_HATCHET_DIR, REFERENCE_SUBDIR, Config

LICENSE_IN_CONTAINER = "/opt/gurobi/gurobi.lic"
PLOIDIES = ("diploid", "tetraploid", "both")
PNG_SCALE_PX = 1600
RSYNC_EXCLUDES = (
    ".git/",
    "tmp/",
    "site/",
    "mcp/",
    "notebooks/",
    "build/",
    "dist/",
    "__pycache__/",
    ".pytest_cache/",
    ".ruff_cache/",
    ".snakemake/",
    ".ipynb_checkpoints/",
    "*.egg-info/",
)


@dataclass(frozen=True)
class Paths:
    """Locations as seen by the inner script.

    Attributes:
        bb: Input ``bb_dir``.
        job: Run directory holding scripts and stage outputs.
        ref: HATCHet's own data/reference directory.
        tmp: Scratch directory.
    """

    bb: str
    job: str
    ref: str
    tmp: str


CONTAINER_PATHS = Paths(
    bb="/data/bb", job="/job", ref=f"{IMAGE_HATCHET_DIR}/{REFERENCE_SUBDIR}", tmp="/scratch"
)


@dataclass
class AnalysisParams:
    """User-facing analysis parameters, translated to HATCHet flags on render.

    Attributes:
        sample_id: Patient identifier used to label compute-cn outputs.
        genome: Genome key, one of ``config.GENOMES``.
        min_k: Smallest number of cluster states tried by cluster-bins.
        max_k: Largest number of cluster states tried by cluster-bins.
        restarts: Random EM restarts per K in cluster-bins.
        min_tumor_clones: Fewest tumor clones compute-cn fits.
        max_tumor_clones: Most tumor clones compute-cn fits.
        ploidy: ``diploid`` (no WGD), ``tetraploid`` (WGD) or ``both``.
        cores: CPU cores for the container and for HATCHet's worker pools.
        memory_gb: Container memory cap, or None for no cap.
        time_limit_hours: Wall-clock cap per stage, or None for no cap.
        solver: ILP solver for compute-cn.
        solver_time_limit_s: Per-solve time limit passed as ``--timelimit``.
        cluster_bins_extra: Additional cluster-bins flags, name -> value.
        compute_cn_extra: Additional compute-cn flags, name -> value.
    """

    sample_id: str
    genome: str
    min_k: int | None = None
    max_k: int | None = None
    restarts: int | None = None
    min_tumor_clones: int = 1
    max_tumor_clones: int = 3
    ploidy: str = "both"
    cores: int = 4
    memory_gb: int | None = None
    time_limit_hours: float | None = None
    solver: str = "cbc"
    solver_time_limit_s: int | None = None
    cluster_bins_extra: dict = field(default_factory=dict)
    compute_cn_extra: dict = field(default_factory=dict)

    def validate(self):
        """Raise ValueError on inconsistent parameters."""
        if self.genome not in GENOMES:
            raise ValueError(f"unknown genome '{self.genome}'; choose one of {sorted(GENOMES)}")
        if self.ploidy not in PLOIDIES:
            raise ValueError(f"ploidy must be one of {PLOIDIES}")
        if not 1 <= self.min_tumor_clones <= self.max_tumor_clones:
            raise ValueError("need 1 <= min_tumor_clones <= max_tumor_clones")
        if self.min_k is not None and self.max_k is not None and self.min_k > self.max_k:
            raise ValueError("min_k must not exceed max_k")
        if self.cores < 1:
            raise ValueError("cores must be >= 1")
        if self.solver not in ("cbc", "gurobi"):
            raise ValueError("solver must be cbc or gurobi")

    @property
    def min_clone_flag(self) -> int:
        """HATCHet ``--minClone``; its clone count includes the normal clone."""
        return self.min_tumor_clones + 1

    @property
    def max_clone_flag(self) -> int:
        """HATCHet ``--maxClone``; its clone count includes the normal clone."""
        return self.max_tumor_clones + 1


def _flags(options: dict) -> list[str]:
    """Render ``{flag: value}`` as CLI tokens; None is dropped, True is a bare flag."""
    tokens = []
    for name, value in options.items():
        if value is None or value is False:
            continue
        flag = name if name.startswith("-") else f"--{name}"
        tokens.append(flag)
        if value is not True:
            tokens.append(str(value))
    return tokens


def host_timezone() -> str | None:
    """The host's timezone as a POSIX TZ rule the container can use without tzdata.

    Honours ``TZ`` when set; otherwise reads the rule from the footer that TZif version
    2+ files end with (``/etc/localtime``).

    Returns:
        A TZ string such as ``EST5EDT,M3.2.0,M11.1.0``, or None when unavailable.
    """
    if os.environ.get("TZ"):
        return os.environ["TZ"]
    try:
        data = Path("/etc/localtime").read_bytes()
    except OSError:
        return None
    if not data.startswith(b"TZif") or data[4:5] not in (b"2", b"3", b"4"):
        return None
    try:
        footer = data[data.rindex(b"\n", 0, len(data) - 1) + 1 : -1].decode("ascii")
    except (ValueError, UnicodeDecodeError):
        return None
    return footer or None


class Raw(str):
    """A token already quoted for the shell, emitted verbatim so ``$VAR`` expands."""


def _command(argv: list[str], indent: str = "  ") -> str:
    """Quote ``argv`` and lay it out one flag per line with its value.

    The program and subcommand share the first line; each flag keeps at most one
    following value; trailing positional arguments share the last line.
    """
    quoted = [a if isinstance(a, Raw) else shlex.quote(a) for a in argv]
    is_flag = [str(a).startswith("-") for a in argv]
    first = next((i for i, f in enumerate(is_flag) if f), len(argv))
    lines, i = [" ".join(quoted[:first])], first
    while i < len(argv):
        if is_flag[i] and i + 1 < len(argv) and not is_flag[i + 1]:
            lines.append(f"{quoted[i]} {quoted[i + 1]}")
            i += 2
        elif is_flag[i]:
            lines.append(quoted[i])
            i += 1
        else:
            lines.append(" ".join(quoted[i:]))
            break
    return (" \\\n" + indent).join(lines)


def _header(title: str) -> str:
    stamp = datetime.now().isoformat(timespec="seconds")
    return (
        "#!/usr/bin/env bash\n"
        f"# {title}\n"
        f"# generated by hatchet-mcp {__version__} on {stamp}\n"
        "set -euo pipefail\n"
    )


_HELPERS = f"""
render_pngs() {{
  local dir="$1" pdf rel out
  rm -rf "$PREVIEW/${{dir#"$JOB"/}}"
  while IFS= read -r -d '' pdf; do
    rel="${{pdf#"$JOB"/}}"
    out="$PREVIEW/${{rel%.pdf}}"
    mkdir -p "${{out%/*}}"
    if command -v pdftoppm >/dev/null 2>&1; then
      pdftoppm -png -scale-to {PNG_SCALE_PX} "$pdf" "$out" \\
        || echo "hatchet-mcp: could not render $pdf" >&2
    elif command -v sips >/dev/null 2>&1; then
      sips -s format png "$pdf" --out "$out-1.png" >/dev/null \\
        || echo "hatchet-mcp: could not render $pdf" >&2
    fi
  done < <(find "$dir" -name '*.pdf' -print0)
}}

run_limited() {{
  if [[ -n "${{STAGE_TIME_LIMIT_S:-}}" ]] && command -v timeout >/dev/null 2>&1; then
    timeout --signal=TERM --kill-after=60 "$STAGE_TIME_LIMIT_S" "$@"
  else
    "$@"
  fi
}}
"""


def _thread_env(params: AnalysisParams) -> str:
    n = params.cores
    limit = "" if params.time_limit_hours is None else str(int(params.time_limit_hours * 3600))
    return (
        f"export OMP_NUM_THREADS={n} NUMBA_NUM_THREADS={n} "
        f"MKL_NUM_THREADS={n} OPENBLAS_NUM_THREADS={n}\n"
        f"STAGE_TIME_LIMIT_S={shlex.quote(limit)}\n"
    )


def preview_dir(paths: Paths, run_id: str) -> str:
    """Directory receiving the PNG previews of a run, as seen by the inner script."""
    return f"{paths.tmp}/previews/{run_id}"


def _preview_env(paths: Paths, run_id: str) -> str:
    return f"JOB={shlex.quote(paths.job)}\nPREVIEW={shlex.quote(preview_dir(paths, run_id))}\n"


def _genome_args(paths: Paths, genome: str) -> dict:
    sizes, bed = GENOMES[genome]
    return {
        "genome_size": f"{paths.ref}/{sizes}",
        "region_bed": f"{paths.ref}/{bed}",
    }


def render_cluster_bins(paths: Paths, params: AnalysisParams, run_id: str) -> str:
    """Inner script for ``hatchet cluster-bins``.

    Args:
        paths: Paths as seen by the script.
        params: Analysis parameters.
        run_id: Run the previews are filed under.

    Returns:
        Bash script text.
    """
    out = f"{paths.job}/cluster-bins"
    options = {
        "bb_dir": paths.bb,
        "bbc_dir": out,
        **_genome_args(paths, params.genome),
        "minK": params.min_k,
        "maxK": params.max_k,
        "restarts": params.restarts,
        "verbosity": 1,
        **params.cluster_bins_extra,
    }
    return (
        _header(f"cluster-bins for {params.sample_id}")
        + _HELPERS
        + "\n"
        + _thread_env(params)
        + _preview_env(paths, run_id)
        + f"mkdir -p {shlex.quote(out)}\n\n"
        + "run_limited "
        + _command(["hatchet", "cluster-bins", *_flags(options)])
        + "\n\n"
        + f"render_pngs {shlex.quote(out)}\n"
    )


def render_compute_cn(paths: Paths, params: AnalysisParams, run_id: str) -> str:
    """Inner script for ``hatchet compute-cn`` followed by ``hatchet plot-cn``.

    The selected ploidy is the hypothesis whose ``chosen.<ploidy>.bbc.ucn`` is
    byte-identical to ``best.bbc.ucn``, which is how compute-cn records its choice.

    Args:
        paths: Paths as seen by the script.
        params: Analysis parameters.
        run_id: Run the previews are filed under.

    Returns:
        Bash script text.
    """
    bbc_dir = f"{paths.job}/cluster-bins"
    result = f"{paths.job}/compute-cn"
    plots = f"{paths.job}/plot-cn"
    genome = _genome_args(paths, params.genome)
    parallel = params.cores > 1
    options = {
        "bbc": f"{bbc_dir}/bulk.bbc",
        "seg": f"{bbc_dir}/bulk.seg",
        "result_dir": result,
        **genome,
        "sample_id": params.sample_id,
        "solver": params.solver,
        "minClone": params.min_clone_flag,
        "maxClone": params.max_clone_flag,
        "diploid": params.ploidy in ("diploid", "both"),
        "tetraploid": params.ploidy in ("tetraploid", "both"),
        "cd_njobs": params.cores,
        "solver_threads": 1 if parallel else None,
        "timelimit": params.solver_time_limit_s,
        "verbosity": 1,
        **params.compute_cn_extra,
    }
    plot_options = {
        "bbc": f"{result}/best.bbc.ucn",
        "seg": f"{result}/best.seg.ucn",
        "gamma_file": f"{result}/gammas.tsv",
        "plot_dir": plots,
        **genome,
        "sample_id": params.sample_id,
    }
    plot_argv = ["hatchet", "plot-cn", *_flags(plot_options), "--ploidy", Raw('"$selected"')]
    return (
        _header(f"compute-cn and plot-cn for {params.sample_id}")
        + _HELPERS
        + "\n"
        + _thread_env(params)
        + _preview_env(paths, run_id)
        + f"mkdir -p {shlex.quote(result)} {shlex.quote(plots)}\n\n"
        + "run_limited "
        + _command(["hatchet", "compute-cn", *_flags(options)])
        + "\n\n"
        + f'RESULT={shlex.quote(result)}\nselected=""\n'
        + "for p in diploid tetraploid; do\n"
        + '  if [[ -f "$RESULT/chosen.$p.bbc.ucn" ]] && '
        + 'cmp -s "$RESULT/chosen.$p.bbc.ucn" "$RESULT/best.bbc.ucn"; then\n'
        + '    selected="$p"\n  fi\ndone\n'
        + 'if [[ -z "$selected" ]]; then\n'
        + '  echo "hatchet-mcp: could not identify the selected ploidy in $RESULT" >&2\n'
        + "  exit 3\nfi\n"
        + 'echo "$selected" > "$RESULT/selected_ploidy.txt"\n\n'
        + _command(plot_argv)
        + "\n\n"
        + f"render_pngs {shlex.quote(result)}\n"
        + f"render_pngs {shlex.quote(plots)}\n"
    )


def render_check(paths: Paths, solver: str) -> str:
    """Inner script for ``hatchet check``, keeping its outputs for inspection.

    Args:
        paths: Paths as seen by the script.
        solver: Solver to verify.

    Returns:
        Bash script text.
    """
    out = f"{paths.job}/check"
    return (
        _header("hatchet check: verify the installation end to end")
        + f"rm -rf {shlex.quote(out)}\n"
        + "hatchet --version\n"
        + _command(["hatchet", "check", "--solver", solver, "--out_dir", out, "--verbosity", "1"])
        + "\n"
    )


def container_command(
    config: Config,
    host: dict,
    inner_script: str,
    container_name: str,
    cores: int | None = None,
    memory_gb: int | None = None,
) -> list[str]:
    """Build the argv that runs an inner script under the configured runtime.

    Args:
        config: User configuration.
        host: Host paths to mount, keyed ``bb``, ``job``, ``tmp``; ``bb`` may be absent.
            Reference files are read from the HATCHet source inside the image.
        inner_script: Path of the inner script as seen by the runtime.
        container_name: Name for the container, used to stop it on cancel.
        cores: CPU cap, or None.
        memory_gb: Memory cap in GB, or None.

    Returns:
        Command argv.

    Raises:
        ValueError: If the runtime is not a container runtime.
    """
    env = {
        "HOME": CONTAINER_PATHS.tmp,
        "TMPDIR": CONTAINER_PATHS.tmp,
        "MPLCONFIGDIR": f"{CONTAINER_PATHS.tmp}/mpl",
        "NUMBA_CACHE_DIR": f"{CONTAINER_PATHS.tmp}/numba",
        "MPLBACKEND": "Agg",
    }
    tz = host_timezone()
    if tz:
        env["TZ"] = tz
    binds = [
        (host["job"], CONTAINER_PATHS.job, "rw"),
        (host["tmp"], CONTAINER_PATHS.tmp, "rw"),
    ]
    if host.get("bb"):
        binds.insert(0, (host["bb"], CONTAINER_PATHS.bb, "ro"))
    if config.solver == "gurobi" and config.gurobi_license:
        binds.append((config.gurobi_license, LICENSE_IN_CONTAINER, "ro"))
        env["GRB_LICENSE_FILE"] = LICENSE_IN_CONTAINER

    runtime = config.runtime
    if runtime in ("docker", "podman"):
        argv = [runtime, "run", "--rm", "--init", "--name", container_name]
        argv += ["--user", f"{os.getuid()}:{os.getgid()}"]
        if runtime == "podman":
            argv += ["--userns=keep-id"]
        if cores:
            argv += ["--cpus", str(cores)]
        if memory_gb:
            argv += ["--memory", f"{memory_gb}g"]
        for src, dst, mode in binds:
            argv += ["-v", f"{src}:{dst}:{mode}"]
        for key, value in env.items():
            argv += ["-e", f"{key}={value}"]
        return argv + [config.image, "bash", inner_script]
    if runtime in ("apptainer", "singularity"):
        if not config.apptainer_image:
            raise ValueError("apptainer_image is not configured")
        bind = ",".join(f"{src}:{dst}:{mode}" for src, dst, mode in binds)
        envs = ",".join(f"{k}={v}" for k, v in env.items())
        return [
            runtime,
            "exec",
            "--cleanenv",
            "--containall",
            "--bind",
            bind,
            "--env",
            envs,
            config.apptainer_image,
            "bash",
            inner_script,
        ]
    raise ValueError(f"runtime '{runtime}' does not run containers")


def render_launch(
    config: Config,
    host: dict,
    stage_script_name: str,
    container_name: str,
    status_file: str,
    cores: int | None = None,
    memory_gb: int | None = None,
) -> str:
    """Host-side launcher that runs one inner script and records its exit code.

    Args:
        config: User configuration.
        host: Host paths keyed ``bb``, ``job``, ``tmp``.
        stage_script_name: File name of the inner script under ``<job>/scripts``.
        container_name: Container name, reused by cancellation.
        status_file: Host path that receives the exit code.
        cores: CPU cap.
        memory_gb: Memory cap in GB.

    Returns:
        Bash script text.
    """
    head = (
        _header(f"host launcher for {stage_script_name} ({config.runtime})")
        + f"trap 'echo $? > {shlex.quote(status_file)}' EXIT\n"
        + f"rm -f {shlex.quote(status_file)}\n"
        + f"mkdir -p {shlex.quote(host['tmp'])}\n\n"
    )
    if config.runtime == "local":
        prefix = config.local_env_prefix or ""
        lines = [
            f"export PATH={shlex.quote(prefix)}/bin:$PATH CONDA_PREFIX={shlex.quote(prefix)}",
            f"export TMPDIR={shlex.quote(host['tmp'])} MPLBACKEND=Agg",
            f"export MPLCONFIGDIR={shlex.quote(host['tmp'])}/mpl",
            f"export NUMBA_CACHE_DIR={shlex.quote(host['tmp'])}/numba",
        ]
        if config.solver == "gurobi" and config.gurobi_license:
            lines.append(f"export GRB_LICENSE_FILE={shlex.quote(config.gurobi_license)}")
        script = f"{host['job']}/scripts/{stage_script_name}"
        return head + "\n".join(lines) + f"\n\nbash {shlex.quote(script)}\n"
    argv = container_command(
        config,
        host,
        f"{CONTAINER_PATHS.job}/scripts/{stage_script_name}",
        container_name,
        cores=cores,
        memory_gb=memory_gb,
    )
    return head + _command(argv) + "\n"


def render_build(
    config: Config,
    context_dir: str,
    build_dir: str,
    commit: str,
    status_file: str,
    check_argv: list[str] | None = None,
) -> str:
    """Host-side script that builds the image and then runs ``hatchet check`` in it.

    The source identity is baked into the image as a label, so a later status call can
    tell whether the image is behind its source.

    Args:
        config: User configuration.
        context_dir: Directory holding the Dockerfile.
        build_dir: Scratch directory for the staged source.
        commit: Source identity recorded in the image label; see
            ``runtime.source_commit``.
        status_file: Host path that receives the exit code.
        check_argv: Command that runs ``hatchet check`` in the new image, appended after
            the build; None skips verification.

    Returns:
        Bash script text.
    """
    q = shlex.quote
    lines = [
        _header(f"build {config.image} and verify it with hatchet check"),
        f"trap 'echo $? > {q(status_file)}' EXIT",
        f"rm -f {q(status_file)}",
        f"mkdir -p {q(build_dir)}",
        f"COMMIT={q(commit)}",
        "",
    ]
    if config.hatchet_local_source:
        src = str(Path(config.hatchet_local_source).expanduser().resolve())
        excludes = " ".join(f"--exclude {q(e)}" for e in RSYNC_EXCLUDES)
        lines += [
            f'echo "hatchet-mcp: staging local source {src}"',
            f"SRC_CTX={q(build_dir)}/hatchet_src",
            f'rsync -a --delete {excludes} {q(src)}/ "$SRC_CTX"/',
            "SOURCE=local",
        ]
    else:
        lines += [f"SRC_CTX={q(build_dir)}/empty", 'mkdir -p "$SRC_CTX"', "SOURCE=git"]
    build_argv = [
        config.runtime or "docker",
        "build",
        "--progress=plain",
        "--build-context",
        Raw('"hatchet_src=$SRC_CTX"'),
        "--build-arg",
        Raw('"HATCHET_SOURCE=$SOURCE"'),
        "--build-arg",
        f"HATCHET_REPO={config.hatchet_repo}",
        "--build-arg",
        f"HATCHET_REF={config.hatchet_ref}",
        "--build-arg",
        Raw('"HATCHET_COMMIT=$COMMIT"'),
        "--build-arg",
        f"HATCHET_MCP_VERSION={__version__}",
        "-t",
        config.image,
        context_dir,
    ]
    lines += ['echo "hatchet-mcp: building from $COMMIT"', "", _command(build_argv), ""]
    if check_argv:
        lines += ['echo "hatchet-mcp: verifying the image with hatchet check"']
        lines += [_command(check_argv), ""]
    return "\n".join(lines) + "\n"


def render_steps(title: str, status_file: str, commands: list[list[str]]) -> str:
    """Host-side script that runs commands in order and records the exit code.

    Args:
        title: One-line description written to the script header.
        status_file: Host path that receives the exit code.
        commands: Argv lists run sequentially; the script stops at the first failure.

    Returns:
        Bash script text.
    """
    lines = [
        _header(title),
        f"trap 'echo $? > {shlex.quote(status_file)}' EXIT",
        f"rm -f {shlex.quote(status_file)}",
        "",
    ]
    for argv in commands:
        lines += [_command(argv), ""]
    return "\n".join(lines) + "\n"
