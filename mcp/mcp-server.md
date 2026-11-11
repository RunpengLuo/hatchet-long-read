# Using hatchet-mcp

hatchet-mcp lets an AI coding agent such as Claude Code run HATCHet for you. It builds a container with HATCHet inside, validates your `bb_dir`, runs `cluster-bins` and `compute-cn` in the container, and opens the resulting figures. This guide walks through a session from installation to results; [README.md](README.md) describes the design.

> [!NOTE]
> hatchet-mcp is for interactive analysis of one patient at a time. Cohort production runs still belong in the Snakemake workflow, which adds a DAG, resumability, per-rule logs and benchmarks, and cluster submission.

## Before you start

You need:

- A running container runtime. Docker and Podman are supported; Apptainer and Singularity can run a published image but are untested.
- A `bb_dir` from upstream preprocessing, with bin coordinates on hg38, hg19 or CHM13.
- Optionally, a Gurobi licence. Without one, `compute-cn` uses the open-source CBC solver bundled in the image.

HATCHet itself does not need to be installed: it lives in the container.

## 1. Install the server

The server needs only Python and `fastmcp`. Give it its own environment:

```sh
mamba create -n hatchet-mcp python=3.12 pip
mamba run -n hatchet-mcp pip install /path/to/hatchet-long-read/mcp
```

## 2. Register it with Claude Code

```sh
claude mcp add hatchet-mcp -s user -- /path/to/envs/hatchet-mcp/bin/hatchet-mcp
```

`mamba run -n hatchet-mcp which hatchet-mcp` prints the path to use.

| Argument | Meaning |
|---|---|
| `hatchet-mcp` | Server name shown by `claude mcp list` and `/mcp` |
| `-s user` | Scope: available from every directory. The default, `local`, is not |
| `--` | End of options; everything after is the command that starts the server |

Registration only records how to start the server. Claude Code launches it as a child process at the start of each session and talks to it over stdin and stdout, so there is no server to start or stop yourself.

### Scopes

| Scope | Stored in | Visible to |
|---|---|---|
| `local` (default) | `~/.claude.json`, keyed to a project root | You, within that project only |
| `user` | `~/.claude.json`, global | You, everywhere |
| `project` | `.mcp.json` at the repo root, committed | Anyone who clones the repo |

> [!IMPORTANT]
> `local` scope keys on the enclosing **git root**, not the current directory. Registering from a subdirectory covers the whole repository, and registering from a directory that is not itself a repository attaches the server to whichever repository contains it.

### Sharing with collaborators

A committed `.mcp.json` offers the server to anyone who clones the repository, after a one-time approval prompt:

```json
{
  "mcpServers": {
    "hatchet-mcp": { "command": "/path/to/envs/hatchet-mcp/bin/hatchet-mcp", "args": [] }
  }
}
```

> [!WARNING]
> The command path is machine-specific, so this only helps collaborators whose layout matches. Each collaborator can instead run the `claude mcp add` command above with their own path.

## 3. Verify the connection

```sh
claude mcp get hatchet-mcp
```

Look for `Connected` on the `Status` line. This starts the server and performs the MCP handshake, so it is a real health check. A failure is almost always a wrong path to `hatchet-mcp`.

## 4. First session: one-time setup

Start `claude` in any directory and ask it to set up HATCHet, for example:

```
Use hatchet-mcp to set up HATCHet. My Gurobi licence is at ~/gurobi.lic.
```

The agent calls `setup_status`, which detects the available container runtimes and proposes a configuration, then asks you for:

1. the path to your Gurobi licence file, or none to use CBC;
2. whether to change the work directory (default `~/hatchet-mcp/work`) or the scratch directory (default `~/hatchet-mcp/scratch`).

`configure` saves the answers to `~/.config/hatchet-mcp/config.json` (set `HATCHET_MCP_CONFIG` to use another file). `build_container` then builds the image and runs `hatchet check` inside it, which simulates a cohort and verifies every stage against known ground truth. The first build creates HATCHet's conda environment and compiles its C++ backend, so it takes several minutes; later builds reuse that environment unless `environment.yaml` changes.

### Where HATCHet comes from

By default the image fetches `raphael-group/hatchet` at `hatchet3-dev`. To build from a local checkout instead, for example to test unreleased changes, give the agent its path. The image records the source commit, including a fingerprint of any uncommitted changes, and `setup_status` reports when the image falls behind its source.

> [!IMPORTANT]
> The build verifies the image with `hatchet check`, so the source must include that subcommand. If the configured branch predates it, build from a local checkout that has it.

### Gurobi licences

`configure` identifies the licence type from its key names, never reading their values:

| Licence | Contains | Works in the container |
|---|---|---|
| Web License Service (WLS) | `WLSACCESSID`, `WLSSECRET`, `LICENSEID` | Yes, with internet access |
| Token server | `TOKENSERVER` | Yes, if the server is reachable |
| Node-locked | `HOSTID` | **No**: bound to one physical machine; CBC is used instead |

Academic users can request a WLS licence at no cost. The licence is mounted read-only when a job runs and is never copied into the image.

## 5. Run an analysis

Tell the agent where the data is and what you expect. It asks for anything missing:

| Question | Notes |
|---|---|
| `bb_dir` | Directory of bin-by-sample matrices from preprocessing |
| Genome build | `hg38`, `hg19` or `chm13`; HATCHet's own reference files for that build are used |
| Sample id | Labels the outputs |
| Expected number of tumor clones | A range, for example 1 to 3 |
| Ploidy | `diploid`, `tetraploid` (whole-genome duplication) or `both` |
| Cores, memory, time limit | Container resources and a wall-clock cap per stage |

Anything you leave unset keeps HATCHet's default. For example:

```
Analyse the bb_dir at /path/to/bb_dir (hg38, sample P1) with 1 to 3 tumor clones,
both ploidies and 8 cores. Do a quick first pass.
```

Before launching anything, `inspect_input` checks the `bb_dir` against HATCHet's input format and reports the number of bins, chromosomes and regions, SNP coverage, the tumor and normal samples, and which genome builds the coordinates fit. The agent stops if the input is invalid.

> [!IMPORTANT]
> HATCHet's `--minClone` and `--maxClone` count the normal clone. You give the number of **tumor** clones and the server adds one, so 1 to 3 tumor clones becomes `--minClone 2 --maxClone 4`.

### Keeping runs short

`compute-cn` fits every combination of ploidy and clone count along a regularisation path, restarting coordinate descent from many random seeds. Its runtime therefore scales with the number of ploidies tested, the width of the clone range, `reg_steps` (default 15) and `cd_nseeds` (default 400). `cluster-bins` scales with the number of bins and tumor samples, the width of the K range and the number of restarts. For a first pass, test one ploidy, narrow the clone and K ranges, or ask for `reg_steps` 3.

## 6. What you get after each stage

`start_analysis` launches `cluster-bins`. Once you have reviewed its results, `continue_analysis` launches `compute-cn` and `plot-cn`; it accepts changed clone, ploidy and resource settings if the clustering suggests them. Each stage runs in the background, and the agent follows it with `check_job` and reports when it finishes.

After **cluster-bins**:

- the selected number of clusters K and the best score for each K tried;
- a table of the clusters with their bins, SNPs, read-depth ratio and B-allele frequency;
- figures: model selection across K, the selected clustering, and EM convergence as QC.

After **compute-cn**:

- the selected ploidy, and therefore whether a whole-genome duplication was called;
- the number of tumor clones, the purity and clone proportions of each sample, and each clone's ploidy;
- figures: model selection with the decision across ploidy and clone count first, the 1D copy-number profile, the 1D allele-specific profile, the 2D mhBAF versus copy-number plot, and RDR scaling as QC.

You see HATCHet's own PDFs: the agent opens the headline figures in your viewer with `open_figure` (`open` on macOS, `xdg-open` on Linux; only PDFs inside the work directory) and lists every other PDF by path. The agent reads PNG previews instead, rendered into `<scratch>/previews/<run_id>/`; they are regenerated on each run and safe to delete.

> [!NOTE]
> HATCHet's figure titles also count the normal clone, so `n=2` in a title means one tumor clone.

## 7. Ask follow-up questions

Once a solution exists, the agent reads the result tables directly:

- Which chromosome arms are LOH in the tumor clones?
- List every segment where the clones differ, longest first.
- Do the two tumor samples share clones at different proportions?
- Re-run compute-cn forcing diploid only; how much worse is the fit?

## 8. Manage runs

Jobs run detached and are recorded on disk, so they keep running if you close the session. In a later session:

| Ask for | Tool |
|---|---|
| Your runs and the state of each stage | `list_runs` |
| Progress or results of a job | `check_job` |
| The results and figures of a finished stage again | `show_results` |
| Stopping a job and its container | `cancel_job` |

Every run keeps the scripts it executed under `<work>/runs/<run_id>/scripts/`: `<stage>.sh` holds the `hatchet` command, and `<stage>.launch.sh` wraps it in `docker run` and records the exit code. Repeat a stage by hand with:

```sh
bash <work>/runs/<run_id>/scripts/compute_cn.launch.sh
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `claude mcp get` shows a failure | Wrong path to `hatchet-mcp` | Re-register with the path from `which hatchet-mcp` |
| No runtime ready in `setup_status` | Docker daemon not running | Start Docker, then ask again |
| Build succeeds but `hatchet check` fails to run | Source predates `hatchet check` | Build from a local checkout that has it |
| Gurobi requested but CBC used | Node-locked licence | Use a WLS licence, or accept CBC |
| `setup_status` says the image is out of date | Source changed since the build | Ask for a rebuild, or keep using the current image |
| A figure does not open | Not macOS or Linux, or the PDF lies outside the work directory | Open the listed PDF path yourself |

## Uninstall

```sh
claude mcp remove hatchet-mcp -s user
docker rmi hatchet-mcp:latest
mamba env remove -n hatchet-mcp
rm -rf ~/.config/hatchet-mcp
```

Remove the work and scratch directories too if you no longer need their runs.
