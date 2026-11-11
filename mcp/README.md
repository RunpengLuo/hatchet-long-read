# hatchet-mcp: run HATCHet copy-number inference from an AI agent

[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](src/hatchet_mcp/__init__.py)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](#)

An MCP (Model Context Protocol) server that lets an agent such as Claude Code set up a HATCHet container, validate a `bb_dir`, run `cluster-bins` and `compute-cn` inside it, and show the model-selection, clustering, QC and copy-number figures after each stage, without the user writing a config file or installing HATCHet.

## How it works

The server itself depends only on `fastmcp` and never imports HATCHet. Every stage is a generated bash script that calls a `hatchet` subcommand inside a container:

```
agent -> hatchet-mcp (host) -> <stage>.launch.sh -> docker run ... -> <stage>.sh -> hatchet <subcommand>
```

Both scripts are written to the run directory, so any stage can be inspected or repeated by hand. Jobs run detached and are recorded on disk, so a long `compute-cn` keeps running after the agent session ends and can be picked up from a later one.

## Install

1. Install a container runtime and start it. Docker and Podman are supported; Apptainer and Singularity are wired but untested.
2. Install the server into its own environment:

```sh
mamba create -n hatchet-mcp python=3.12 pip
mamba run -n hatchet-mcp pip install /path/to/hatchet-long-read/mcp
```

3. Register it with Claude Code, available from every project:

```sh
claude mcp add hatchet-mcp -s user -- "$(mamba run -n hatchet-mcp which hatchet-mcp)"
```

4. Check it connected:

```sh
claude mcp list
```

## Use

Start `claude` in the directory holding your data and ask it to analyse the sample. The server carries instructions that walk the agent through the steps below, pausing to ask you where needed. [mcp-server.md](mcp-server.md) walks through a full session step by step.

| Step | Tool | What happens |
|---|---|---|
| 1 | `setup_status` | Detects container runtimes, shows the proposed work and scratch directories, and asks for your Gurobi licence path |
| 2 | `configure` | Saves your answers to `~/.config/hatchet-mcp/config.json` |
| 3 | `build_container` | Builds the image, then runs `hatchet check` inside it |
| 4 | - | Asks for the `bb_dir`, genome build, sample id, expected tumor clones, ploidy, cores, memory and time limit |
| 5 | `inspect_input` | Validates the `bb_dir` and reports bins, chromosomes, tumor and normal samples |
| 6 | `start_analysis` | Runs `cluster-bins`; shows model selection across K, the selected clustering and QC plots |
| 7 | `continue_analysis` | Runs `compute-cn` and `plot-cn`; shows the model-selection result, purity and clone proportions, and the 1D and 2D copy-number profiles |

`check_job` follows any running job and returns its results and figures once it finishes. `open_figure` opens figures in your PDF viewer. `show_results`, `list_runs` and `cancel_job` cover the rest.

> [!IMPORTANT]
> HATCHet's `--minClone` and `--maxClone` count the normal clone. The server asks for the number of **tumor** clones and adds one, so a request for 2 to 3 tumor clones becomes `--minClone 3 --maxClone 4`.

## The image

`build_container` creates the conda environment from HATCHet's own `environment.yaml`, pip-installs HATCHet from source so its C++ HMM backend is compiled, and runs `hatchet check`, which simulates a cohort and verifies every stage against ground truth. HATCHet's reference files are used from its `data/reference` directory inside the image.

The source is either a git reference (default `raphael-group/hatchet` at `hatchet3-dev`) or a local checkout, set with `configure(hatchet_local_source=...)`, which is the way to test unreleased code. The source commit is recorded as an image label, and `setup_status` reports when the image falls behind it. The conda environment is rebuilt only when `environment.yaml` changes.

## Gurobi

`compute-cn` needs an ILP solver in every mode. Without a usable Gurobi licence the server uses the open-source CBC solver bundled in the image.

> [!WARNING]
> A node-locked Gurobi licence (one containing `HOSTID`) is bound to one physical machine and will not validate inside a container. `configure` detects this from the licence's key names, without reading their values, and falls back to CBC. To use Gurobi in the container, obtain a Web License Service (WLS) licence, which academic users can request at no cost.

The licence is mounted read-only at run time and never copied into the image.

## Outputs

```
<work_dir>/
  runs/<run_id>/
    run.json                   inputs, parameters, stage states
    scripts/                   cluster_bins.sh, compute_cn.sh and their .launch.sh wrappers
    cluster-bins/              bulk.bbc, bulk.seg, model_scores.tsv, plots/
    compute-cn/                best.bbc.ucn, best.seg.ucn, summary.tsv, plots/
    plot-cn/                   1D and 2D profiles of the selected solution
    logs/
  jobs/                        one record per background job
  setup/                       build scripts, hatchet check outputs
<tmp_dir>/
  previews/<run_id>/           PNG previews for the agent, mirroring runs/<run_id>/
```

You see HATCHet's own PDFs. When a stage finishes, the agent opens its headline figures in your viewer with `open_figure` (model selection, the selected clustering or copy-number profiles, and one QC overview) and lists every other PDF by path with its page count.

MCP tool results carry text and images but no document type, so the agent reads PNG previews instead. The stage scripts render them into the scratch directory, mirroring each PDF's path under `<tmp_dir>/previews/<run_id>/`, which keeps the run directory to HATCHet's outputs. The previews are regenerated on each run and are safe to delete.

> [!NOTE]
> HATCHet's figure titles count the normal clone, so `n=2` in a title means one tumor clone. Results report both `n_tumor_clones` and `hatchet_n_clones`.

## Runtimes

| Runtime | Notes |
|---|---|
| `docker`, `podman` | Build and run supported; containers run as your user so outputs are owned by you |
| `apptainer`, `singularity` | Run a `.sif` built from a published image (`registry_image`); untested |
| `local` | Run an existing HATCHet install on the host (`local_env_prefix`, `hatchet_local_source`); a node-locked Gurobi licence works here |

## Development

```sh
mamba run -n hatchet-mcp pip install -e ".[test]"
mamba run -n hatchet-mcp python -m pytest
HATCHET_MCP_E2E=1 mamba run -n hatchet-mcp python -m pytest tests/test_e2e_docker.py
```

The end-to-end test launches the server over stdio as a client would, builds the image, simulates an input inside it, and calls every workflow tool.
