# `check`
`check` runs the whole pipeline end to end on a synthetic dataset bundled inside the package and verifies the result. It takes no input files: use it to confirm that an installation (including the compiled `_hmm_cpp` extension, the plotting stack, and a MILP solver) actually works.

## Input

None. `check` simulates its own input from a fixed seed, so it works from any working directory and ships no data files. The simulation ([`simulate.py`](../../src/hatchet/check/simulate.py)) builds 1 matched normal + 1 tumor sample at 80% purity over `chr1` and `chr22`, split into four arm-level regions carrying distinct copy-number states. Per bin, RDR is the mixed-clone expectation plus Gaussian noise and B-allele counts are Beta-Binomial around the mixed-clone BAF; the simulated CN profile, purity, and RDR scaling factor are returned in memory and used as the ground truth for the assertions below.

| Parameter | Default | Description |
|---|---|---|
| `-O` / `--out_dir` | *(temporary directory)* | Directory for intermediate and output files |
| `--tmpdir` | *(system temp)* | Parent directory to create the temporary working directory under |
| `--solver` | *(auto)* | MILP solver for compute-cn: `gurobi` or `cbc` |
| `--verbosity` | 0 | Logging verbosity (0, 1, 2) |

## Usage

```console
$ hatchet check
$ hatchet check --solver cbc -O /tmp/hatchet-check
$ hatchet check --tmpdir /scratch/$USER
```

## Main parameters

- **Output location (`-O`/`--out_dir`).** Without it, the simulated input and the three stages' outputs go to a temporary directory that is deleted when every assertion passes and kept when one fails, so a failing run can be inspected. With it, everything is kept under `<out_dir>/{input,bbc,results,plots}`.

- **Scratch location (`--tmpdir`).** Sets the parent directory the temporary working directory is created under, for machines where the system temp is small or not writable (a cluster node with a tiny `/tmp` and a large scratch filesystem). Cleanup is unchanged: the run still deletes after itself on success. The directory must already exist, and `--tmpdir` is ignored with a warning when `--out_dir` is given. Setting the `TMPDIR` environment variable has the same effect.

- **Solver (`--solver`).** Left unset, `check` picks `gurobi` when a license is found and falls back to `cbc`, and exits with an error if neither is available.

## Output

`check` runs [`cluster-bins`](cluster-bins.md) -> [`compute-cn`](compute-cn.md) -> [`plot-cn`](plot-cn.md), then prints one `PASS`/`FAIL` line per assertion and exits non-zero if any fails. The table is printed rather than logged, so it appears at every verbosity level; the stages themselves stream their usual logs.

Assertions cover three things:

- **Files.** Each stage wrote the outputs the next stage consumes (`bulk.bbc`, `bulk.seg`, `model_scores.tsv`; `best.bbc.ucn`, `best.seg.ucn`, `gammas.tsv`; the three plot-cn figures).
- **Invariants.** cluster-bins separated the CN states into distinct clusters; the normal clone is `1|1` everywhere; clone proportions sum to 1.
- **Recovery.** Tumor purity, the RDR scaling factor `gamma`, and the per-region tumor copy-number states match what was simulated. Allele order is ignored when comparing a state, since the A/B labelling is not identifiable.

The seed and the stage parameters are fixed, the latter at the smallest settings that still exercise every code path, so a run is reproducible. `check` is a smoke test of the installation, not a benchmark of inference quality.
