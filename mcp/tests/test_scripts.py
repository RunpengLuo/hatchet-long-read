"""Generated bash scripts: flag translation, mounts and shell syntax."""

import subprocess

import pytest

from hatchet_mcp.config import Config
from hatchet_mcp.scripts import (
    CONTAINER_PATHS,
    AnalysisParams,
    container_command,
    render_build,
    render_check,
    render_cluster_bins,
    render_compute_cn,
    render_launch,
    render_steps,
)

HOST = {"bb": "/in/bb", "job": "/work/runs/r1", "tmp": "/scratch-host"}


def _bash_ok(text, tmp_path):
    script = tmp_path / "s.sh"
    script.write_text(text)
    return subprocess.run(["bash", "-n", str(script)], capture_output=True).returncode == 0


def test_tumor_clones_translate_to_hatchet_clone_counts():
    params = AnalysisParams(sample_id="S", genome="hg38", min_tumor_clones=2, max_tumor_clones=4)
    assert (params.min_clone_flag, params.max_clone_flag) == (3, 5)
    text = render_compute_cn(CONTAINER_PATHS, params, "r1")
    assert "--minClone 3" in text.replace(" \\\n  ", " ")
    assert "--maxClone 5" in text.replace(" \\\n  ", " ")


def test_unset_parameters_keep_hatchet_defaults():
    text = render_cluster_bins(CONTAINER_PATHS, AnalysisParams(sample_id="S", genome="hg38"), "r1")
    assert "--minK" not in text and "--maxK" not in text and "--restarts" not in text
    text = render_cluster_bins(
        CONTAINER_PATHS, AnalysisParams(sample_id="S", genome="hg38", max_k=8), "r1"
    )
    assert "--maxK" in text


@pytest.mark.parametrize(
    "ploidy, dip, tet",
    [("both", True, True), ("diploid", True, False), ("tetraploid", False, True)],
)
def test_ploidy_flags(ploidy, dip, tet):
    text = render_compute_cn(
        CONTAINER_PATHS, AnalysisParams(sample_id="S", genome="hg38", ploidy=ploidy), "r1"
    )
    assert ("--diploid" in text) is dip and ("--tetraploid" in text) is tet


def test_parallel_workers_force_single_solver_thread():
    one = render_compute_cn(
        CONTAINER_PATHS, AnalysisParams(sample_id="S", genome="hg38", cores=1), "r1"
    )
    many = render_compute_cn(
        CONTAINER_PATHS, AnalysisParams(sample_id="S", genome="hg38", cores=8), "r1"
    )
    assert "--solver_threads" not in one and "--solver_threads" in many


def test_previews_go_to_scratch_not_the_run_directory():
    text = render_compute_cn(CONTAINER_PATHS, AnalysisParams(sample_id="S", genome="hg38"), "r1")
    assert "PREVIEW=/scratch/previews/r1" in text and "JOB=/job" in text
    assert 'out="$PREVIEW/${rel%.pdf}"' in text and '"${pdf%.pdf}"' not in text


def test_plot_cn_writes_pdf_originals_and_previews_every_stage_dir():
    text = render_compute_cn(CONTAINER_PATHS, AnalysisParams(sample_id="S", genome="hg38"), "r1")
    assert "--plot_img_type" not in text
    assert "render_pngs /job/compute-cn" in text and "render_pngs /job/plot-cn" in text


def test_references_come_from_hatchet_source_in_image():
    text = render_cluster_bins(CONTAINER_PATHS, AnalysisParams(sample_id="S", genome="hg38"), "r1")
    assert "/opt/hatchet/data/reference/hg38.chrom.sizes" in text


def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        AnalysisParams(
            sample_id="S", genome="hg38", min_tumor_clones=3, max_tumor_clones=2
        ).validate()
    with pytest.raises(ValueError):
        AnalysisParams(sample_id="S", genome="mm10").validate()


def test_container_mounts_and_licence():
    cbc = Config(runtime="docker", solver="cbc", gurobi_license="/h/gurobi.lic")
    argv = container_command(cbc, HOST, "/job/scripts/x.sh", "c1", cores=4, memory_gb=8)
    joined = " ".join(argv)
    assert "/in/bb:/data/bb:ro" in joined and "/work/runs/r1:/job:rw" in joined
    assert "--cpus 4" in joined and "--memory 8g" in joined
    assert "gurobi.lic" not in joined
    gurobi = Config(runtime="docker", solver="gurobi", gurobi_license="/h/gurobi.lic")
    joined = " ".join(container_command(gurobi, HOST, "/job/scripts/x.sh", "c1"))
    assert "/h/gurobi.lic:/opt/gurobi/gurobi.lic:ro" in joined
    assert "GRB_LICENSE_FILE=/opt/gurobi/gurobi.lic" in joined


def test_all_scripts_are_valid_bash(tmp_path):
    params = AnalysisParams(sample_id="S 1", genome="hg38", max_k=8, time_limit_hours=2)
    docker = Config(runtime="docker")
    local = Config(runtime="local", local_env_prefix="/env", hatchet_local_source="/src")
    texts = [
        render_cluster_bins(CONTAINER_PATHS, params, "r1"),
        render_compute_cn(CONTAINER_PATHS, params, "r1"),
        render_check(CONTAINER_PATHS, "cbc"),
        render_launch(docker, HOST, "cluster_bins.sh", "c1", "/work/.status/cb.exit", cores=4),
        render_launch(local, HOST, "cluster_bins.sh", "c1", "/work/.status/cb.exit"),
        render_build(
            docker, "/ctx", "/work/build", "abc123", "/work/.status/b.exit", ["docker", "run", "x"]
        ),
        render_steps("steps", "/work/.status/s.exit", [["echo", "a"], ["echo", "b"]]),
    ]
    for text in texts:
        assert _bash_ok(text, tmp_path), text


def test_build_from_local_source_stages_with_rsync():
    config = Config(runtime="docker", hatchet_local_source="/src/hatchet")
    text = render_build(config, "/ctx", "/work/build", "abc-dirty-1234", "/s")
    assert "rsync -a --delete" in text and "--exclude tmp/" in text
    assert "COMMIT=abc-dirty-1234" in text and "SOURCE=local" in text
    assert '"HATCHET_COMMIT=$COMMIT"' in text


def test_host_timezone_from_tzif_footer(tmp_path, monkeypatch):
    from hatchet_mcp import scripts

    monkeypatch.delenv("TZ", raising=False)
    fake = tmp_path / "localtime"
    fake.write_bytes(b"TZif2" + b"\x00" * 40 + b"\nEST5EDT,M3.2.0,M11.1.0\n")
    real_read = scripts.Path.read_bytes
    monkeypatch.setattr(
        scripts.Path,
        "read_bytes",
        lambda self: fake.read_bytes() if str(self) == "/etc/localtime" else real_read(self),
    )
    assert scripts.host_timezone() == "EST5EDT,M3.2.0,M11.1.0"
    monkeypatch.setenv("TZ", "UTC")
    assert scripts.host_timezone() == "UTC"
