"""Background jobs, image state and source identity.

Every long-running operation (image build, cluster-bins, compute-cn) runs as a detached
host process executing a generated launcher script. Job records are written to
``<work_dir>/jobs/`` rather than held in memory: an MCP stdio server lives only as long as
the client session that spawned it, so a job started in one session must stay findable,
and keep running, after that session ends.
"""

import hashlib
import io
import json
import os
import secrets
import signal
import subprocess
import tarfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from hatchet_mcp.config import GENOMES, Config, reference_dir

_CHILDREN: dict[str, subprocess.Popen] = {}


@dataclass
class Job:
    """A background operation.

    Attributes:
        job_id: Unique identifier.
        kind: ``build``, ``cluster-bins`` or ``compute-cn``.
        run_id: Analysis run the job belongs to, if any.
        launch_script: Host script the job executes.
        log: Combined stdout/stderr of the launcher.
        status_file: File the launcher writes its exit code to on exit.
        container_name: Container name, used to stop it on cancel.
        runtime: Runtime the job was launched with.
        pid: Process id of the launcher.
        started: ISO timestamp.
    """

    job_id: str
    kind: str
    run_id: str | None
    launch_script: str
    log: str
    status_file: str
    container_name: str | None
    runtime: str | None
    pid: int | None = None
    started: str = ""


def jobs_dir(config: Config) -> Path:
    """Directory holding job records."""
    path = config.work_path / "jobs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def new_job_id(kind: str) -> str:
    """Return a unique, sortable job id."""
    return f"{kind}-{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"


def save_job(config: Config, job: Job):
    """Persist a job record."""
    (jobs_dir(config) / f"{job.job_id}.json").write_text(json.dumps(asdict(job), indent=2))


def load_job(config: Config, job_id: str) -> Job:
    """Load a job record.

    Raises:
        FileNotFoundError: If no record exists for ``job_id``.
    """
    path = jobs_dir(config) / f"{job_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"no job '{job_id}' under {jobs_dir(config)}")
    return Job(**json.loads(path.read_text()))


def list_jobs(config: Config) -> list[Job]:
    """All job records, newest first."""
    records = sorted(jobs_dir(config).glob("*.json"), reverse=True)
    return [Job(**json.loads(p.read_text())) for p in records]


def start_job(config: Config, job: Job) -> Job:
    """Launch a job's script as a detached process that outlives this server.

    Args:
        config: User configuration.
        job: Job whose ``launch_script`` exists.

    Returns:
        The job with ``pid`` and ``started`` filled in and persisted.
    """
    Path(job.log).parent.mkdir(parents=True, exist_ok=True)
    log = open(job.log, "ab")
    proc = subprocess.Popen(
        ["bash", job.launch_script],
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    log.close()
    _CHILDREN[job.job_id] = proc
    job.pid = proc.pid
    job.started = datetime.now().isoformat(timespec="seconds")
    save_job(config, job)
    return job


def _process_alive(job: Job) -> bool:
    child = _CHILDREN.get(job.job_id)
    if child is not None:
        return child.poll() is None
    if not job.pid:
        return False
    try:
        os.kill(job.pid, 0)
    except OSError:
        return False
    out = subprocess.run(
        ["ps", "-p", str(job.pid), "-o", "stat=,command="], capture_output=True, text=True
    ).stdout.strip()
    return bool(out) and not out.startswith("Z") and job.launch_script in out


def job_state(job: Job) -> dict:
    """Report whether a job is running, succeeded or failed.

    Returns:
        Dict with ``state`` (running, succeeded, failed), ``exit_code`` and
        ``elapsed_s``.
    """
    status = Path(job.status_file)
    started = datetime.fromisoformat(job.started) if job.started else datetime.now()
    end = datetime.fromtimestamp(status.stat().st_mtime) if status.exists() else datetime.now()
    elapsed = int((end - started).total_seconds())
    if status.exists():
        text = status.read_text().strip()
        code = int(text) if text.lstrip("-").isdigit() else None
        state = "succeeded" if code == 0 else "failed"
        return {"state": state, "exit_code": code, "elapsed_s": elapsed}
    if _process_alive(job):
        return {"state": "running", "exit_code": None, "elapsed_s": elapsed}
    return {
        "state": "failed",
        "exit_code": None,
        "elapsed_s": elapsed,
        "note": "launcher exited without recording a status (killed or crashed)",
    }


def tail(path: str, lines: int = 40) -> str:
    """Last ``lines`` lines of a text file, or an empty string."""
    p = Path(path)
    if not p.exists():
        return ""
    with open(p, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        fh.seek(max(0, size - 64 * 1024))
        data = fh.read().decode("utf-8", errors="replace")
    return "\n".join(data.splitlines()[-lines:])


def cancel_job(job: Job) -> dict:
    """Stop a running job and its container.

    Returns:
        Dict describing what was stopped.
    """
    actions = []
    if job.container_name and job.runtime in ("docker", "podman"):
        subprocess.run([job.runtime, "stop", "-t", "10", job.container_name], capture_output=True)
        actions.append(f"stopped container {job.container_name}")
    if job.pid:
        try:
            os.killpg(job.pid, signal.SIGTERM)
            actions.append(f"sent SIGTERM to process group {job.pid}")
        except OSError:
            actions.append("launcher already exited")
    return {"job_id": job.job_id, "actions": actions}


def _git(args: list[str], timeout: int = 30) -> str | None:
    try:
        proc = subprocess.run(["git", *args], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout if proc.returncode == 0 else None


def source_commit(config: Config) -> str | None:
    """Identify the HATCHet source the image should be built from.

    For a local checkout this is ``HEAD``, suffixed with ``-dirty-<hash>`` when there
    are uncommitted changes; the hash covers the diff and the untracked-file list, so
    different uncommitted states get different identities. For a git source it is the
    commit ``hatchet_ref`` points at on the remote.

    Returns:
        Commit identity, or None when it cannot be resolved (offline, bad ref).
    """
    if config.hatchet_local_source:
        src = str(Path(config.hatchet_local_source).expanduser().resolve())
        head = _git(["-C", src, "rev-parse", "HEAD"])
        if head is None:
            return None
        porcelain = _git(["-C", src, "status", "--porcelain"]) or ""
        if not porcelain.strip():
            return head.strip()
        diff = _git(["-C", src, "diff", "HEAD"]) or ""
        digest = hashlib.sha1((diff + porcelain).encode()).hexdigest()[:10]
        return f"{head.strip()}-dirty-{digest}"
    out = _git(["ls-remote", config.hatchet_repo, config.hatchet_ref])
    if not out:
        return None
    return out.split()[0]


def read_chrom_sizes(config: Config) -> dict[str, dict[str, int]] | None:
    """Chromosome sizes of every genome in HATCHet's own reference directory.

    Reads the files HATCHet ships in ``data/reference`` rather than a copy: from the
    image by streaming them out as a tar archive, or from the checkout for the ``local``
    runtime.

    Returns:
        ``{genome: {chromosome: size}}``, or None when the image or checkout is not
        available.
    """
    names = {genome: sizes for genome, (sizes, _) in GENOMES.items()}
    texts = {}
    if config.runtime == "local":
        try:
            ref = Path(reference_dir(config))
        except ValueError:
            return None
        texts = {n: (ref / n).read_text() for n in names.values() if (ref / n).exists()}
    else:
        ref = reference_dir(config)
        tar_argv = ["tar", "-C", ref, "-cf", "-", *names.values()]
        if config.runtime in ("docker", "podman"):
            argv = [config.runtime, "run", "--rm", "--network", "none", config.image, *tar_argv]
        elif config.runtime in ("apptainer", "singularity") and config.apptainer_image:
            argv = [config.runtime, "exec", config.apptainer_image, *tar_argv]
        else:
            return None
        try:
            proc = subprocess.run(argv, capture_output=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired):
            return None
        if proc.returncode != 0:
            return None
        with tarfile.open(fileobj=io.BytesIO(proc.stdout)) as archive:
            for member in archive.getmembers():
                fh = archive.extractfile(member)
                if fh is not None:
                    texts[Path(member.name).name] = fh.read().decode()
    sizes = {}
    for genome, name in names.items():
        if name not in texts:
            continue
        table = {}
        for line in texts[name].splitlines():
            parts = line.split("\t")
            if len(parts) >= 2 and parts[1].strip().isdigit():
                table[parts[0]] = int(parts[1])
        sizes[genome] = table
    return sizes or None


def image_labels(config: Config) -> dict | None:
    """Labels of the configured image, or None when it does not exist.

    Only Docker and Podman images are inspected; for Apptainer the presence of the
    ``.sif`` file is reported instead.
    """
    if config.runtime in ("docker", "podman"):
        proc = subprocess.run(
            [
                config.runtime,
                "image",
                "inspect",
                "--format",
                "{{json .Config.Labels}}",
                config.image,
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            return None
        return json.loads(proc.stdout.strip() or "{}") or {}
    if config.runtime in ("apptainer", "singularity"):
        sif = config.apptainer_image
        return {"sif": sif} if sif and Path(sif).exists() else None
    return None
