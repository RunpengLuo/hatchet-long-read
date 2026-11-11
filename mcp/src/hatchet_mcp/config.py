"""Persistent configuration, container-runtime detection and Gurobi licence checks.

The configuration lives in a single JSON file, by default
``~/.config/hatchet-mcp/config.json`` (override with ``HATCHET_MCP_CONFIG``). Nothing in
this module imports HATCHet, so the server runs in a minimal environment and delegates all
computation to the container.
"""

import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, fields
from importlib import resources
from pathlib import Path

CONFIG_ENV = "HATCHET_MCP_CONFIG"
DEFAULT_CONFIG_PATH = Path.home() / ".config" / "hatchet-mcp" / "config.json"

CONTAINER_RUNTIMES = ("docker", "podman", "apptainer", "singularity")
RUNTIMES = CONTAINER_RUNTIMES + ("local",)

DEFAULT_HATCHET_REPO = "https://github.com/raphael-group/hatchet.git"
DEFAULT_HATCHET_REF = "hatchet3-dev"
DEFAULT_IMAGE = "hatchet-mcp:latest"

IMAGE_HATCHET_DIR = "/opt/hatchet"
REFERENCE_SUBDIR = "data/reference"

# NB: genome key -> (chromosome sizes, region bed) file names under HATCHet's data/reference
GENOMES = {
    "hg38": ("hg38.chrom.sizes", "hg38.regions.bed"),
    "hg19": ("hg19.chrom.sizes", "hg19.regions.bed"),
    "chm13": ("T2T-CHM13v2.0.sizes", "T2T-CHM13v2.0.regions.bed"),
}


@dataclass
class Config:
    """User configuration for hatchet-mcp.

    Attributes:
        runtime: Container runtime (docker, podman, apptainer, singularity) or ``local``
            to run an existing HATCHet install directly on the host.
        work_dir: Root for analysis runs, generated scripts, logs and job records.
        tmp_dir: Scratch space mounted into the container as HOME and TMPDIR.
        hatchet_repo: Git repository the image clones HATCHet from.
        hatchet_ref: Branch or tag of ``hatchet_repo`` to build.
        hatchet_local_source: Local HATCHet checkout to build from instead of cloning;
            use it to test unreleased code.
        image: Tag of the built Docker/Podman image.
        apptainer_image: Path to a ``.sif`` image for apptainer/singularity.
        registry_image: Registry reference an apptainer ``.sif`` is built from.
        local_env_prefix: Conda prefix holding HATCHet, used when ``runtime`` is local.
        gurobi_license: Path to a Gurobi licence file, mounted read-only at run time.
        solver: ILP solver passed to compute-cn and ``hatchet check``.
        configured: Whether the user has confirmed the configuration.
    """

    runtime: str | None = None
    work_dir: str = str(Path.home() / "hatchet-mcp" / "work")
    tmp_dir: str = str(Path.home() / "hatchet-mcp" / "scratch")
    hatchet_repo: str = DEFAULT_HATCHET_REPO
    hatchet_ref: str = DEFAULT_HATCHET_REF
    hatchet_local_source: str | None = None
    image: str = DEFAULT_IMAGE
    apptainer_image: str | None = None
    registry_image: str | None = None
    local_env_prefix: str | None = None
    gurobi_license: str | None = None
    solver: str = "cbc"
    configured: bool = False

    @property
    def work_path(self) -> Path:
        """Absolute work directory."""
        return Path(self.work_dir).expanduser().resolve()

    @property
    def tmp_path(self) -> Path:
        """Absolute scratch directory."""
        return Path(self.tmp_dir).expanduser().resolve()


def config_path() -> Path:
    """Return the configuration file path, honouring ``HATCHET_MCP_CONFIG``."""
    return Path(os.environ.get(CONFIG_ENV, DEFAULT_CONFIG_PATH)).expanduser()


def load_config() -> Config:
    """Load the configuration, falling back to defaults for a first run.

    Returns:
        The stored configuration, or a default one with ``configured`` False and the
        first ready container runtime pre-selected.
    """
    path = config_path()
    if path.exists():
        data = json.loads(path.read_text())
        known = {f.name for f in fields(Config)}
        return Config(**{k: v for k, v in data.items() if k in known})
    config = Config()
    ready = [r["name"] for r in detect_runtimes() if r["ready"]]
    config.runtime = ready[0] if ready else None
    return config


def save_config(config: Config) -> Path:
    """Write the configuration to disk.

    Args:
        config: Configuration to persist.

    Returns:
        Path the configuration was written to.
    """
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(config), indent=2) + "\n")
    return path


def _run(cmd, timeout=20):
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, (proc.stdout or proc.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)


def detect_runtimes() -> list[dict]:
    """Report which container runtimes are installed and usable.

    Docker and Podman are only ready when their daemon/service answers; Apptainer and
    Singularity are daemonless, so an installed binary counts as ready.

    Returns:
        One dict per installed runtime with ``name``, ``path``, ``version``, ``ready``
        and ``detail``.
    """
    found = []
    for name in CONTAINER_RUNTIMES:
        path = shutil.which(name)
        if not path:
            continue
        _, version = _run([name, "--version"])
        if name in ("docker", "podman"):
            code, out = _run([name, "info", "--format", "{{.ServerVersion}}"])
            ready = code == 0 and bool(out)
            detail = "daemon reachable" if ready else f"daemon not reachable: {out[:200]}"
        else:
            ready, detail = True, "daemonless runtime"
        found.append(
            {
                "name": name,
                "path": path,
                "version": version.splitlines()[0] if version else "",
                "ready": ready,
                "detail": detail,
            }
        )
    return found


def inspect_gurobi_license(path: str | None) -> dict:
    """Classify a Gurobi licence file by the key names it contains.

    Only key names are read; values are never returned or logged. A node-locked licence
    carries a ``HOSTID`` bound to one physical machine and does not validate inside a
    container, whose host ID differs. Web License Service (WLS) and token-server
    licences are machine-independent.

    Args:
        path: Path to ``gurobi.lic``, or None.

    Returns:
        Dict with ``path``, ``type`` (none, missing, wls, token-server, node-locked,
        unknown), ``usable_in_container`` and a human-readable ``note``.
    """
    if not path:
        return {
            "path": None,
            "type": "none",
            "usable_in_container": False,
            "note": "No Gurobi licence configured; compute-cn will use the open-source CBC solver.",
        }
    lic = Path(path).expanduser()
    if not lic.is_file():
        return {
            "path": str(lic),
            "type": "missing",
            "usable_in_container": False,
            "note": f"Licence file not found: {lic}",
        }
    keys = set()
    for line in lic.read_text(errors="ignore").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            keys.add(line.split("=", 1)[0].strip().upper())
    if {"WLSACCESSID", "WLSSECRET", "LICENSEID"} <= keys:
        kind, usable = "wls", True
        note = "Web License Service licence: works inside containers (needs internet access)."
    elif "TOKENSERVER" in keys:
        kind, usable = "token-server", True
        note = "Token-server licence: works inside containers if the server is reachable."
    elif "HOSTID" in keys:
        kind, usable = "node-locked", False
        note = (
            "Node-locked licence bound to this machine's host ID. It will NOT validate "
            "inside a container. Request a Gurobi WLS licence (free for academics) to use "
            "Gurobi in the container; until then compute-cn uses CBC."
        )
    else:
        kind, usable = "unknown", False
        note = "Unrecognised licence format; compute-cn will use CBC."
    return {
        "path": str(lic.resolve()),
        "type": kind,
        "usable_in_container": usable,
        "note": note,
    }


def choose_solver(license_info: dict, runtime: str | None) -> str:
    """Pick the compute-cn solver for a licence and runtime.

    Args:
        license_info: Output of :func:`inspect_gurobi_license`.
        runtime: Configured runtime; on ``local`` a node-locked licence is valid.

    Returns:
        ``gurobi`` when the licence will validate where HATCHet runs, else ``cbc``.
    """
    if license_info["usable_in_container"]:
        return "gurobi"
    if runtime == "local" and license_info["type"] == "node-locked":
        return "gurobi"
    return "cbc"


def container_context_dir() -> Path:
    """Directory holding the Dockerfile used as the image build context."""
    return Path(str(resources.files("hatchet_mcp") / "container"))


def reference_dir(config: Config) -> str:
    """HATCHet's reference directory as seen by the stage scripts.

    Inside a container this is the copy shipped with the HATCHet source in the image;
    for the ``local`` runtime it is the configured HATCHet checkout.

    Raises:
        ValueError: If the local runtime has no HATCHet checkout configured.
    """
    if config.runtime == "local":
        if not config.hatchet_local_source:
            raise ValueError("runtime 'local' needs hatchet_local_source for data/reference")
        return str(Path(config.hatchet_local_source).expanduser().resolve() / REFERENCE_SUBDIR)
    return f"{IMAGE_HATCHET_DIR}/{REFERENCE_SUBDIR}"
