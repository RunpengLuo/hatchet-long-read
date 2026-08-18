# `compute-cn`
`compute-cn` performs allele-specific integer copy number and clone proportion deconvolution with regularization using integer linear programming (ILP) or a coordinate descent algorithm.

## Input

The BBC and SEG tables produced by `cluster-bins` (`--bbc`, `--seg`; e.g. `bbc/bulk.bbc` and `bbc/bulk.seg`), plus the reference `--genome_size` and `--region_bed`. See [cluster-bins Output](cluster-bins.md#output) for the `bbc/` file layout.

| Parameter | Default | Description |
|---|---|---|
| `--bbc` | *(required)* | Input BBC table (e.g., `bbc/bulk.bbc`; with `--wide_format`, `bbc/bulk.bbc.tsv.gz`) |
| `--seg` | *(required)* | Input SEG table (e.g., `bbc/bulk.seg`) |
| `--wide_format` | False | EXPERIMENTAL: read wide-format BBC/SEG written by cluster-bins `--wide_format`, and write the `.ucn` outputs in the matching wide layout |
| `--result_dir` | *(required)* | Output directory for computed CN results |
| `--genome_size` | *(required)* | Reference chromosome sizes file |
| `--region_bed` | *(required)* | Reference chromosome BED file |
| `--patient_id` | `panel` | Output filename prefix for per-(ploidy, n) plots |
| `--force` | False | Re-solve even if results already exist (default: skip existing) |
| `--verbosity` | 0 | Verbose level: 0, 1, or 2 |

## Usage

```console
$ hatchet compute-cn --help
usage: hatchet compute-cn [-h] --result_dir RESULT_DIR --bbc BBC --seg SEG
                          [--mode {both,cd,ilp}] [--model_select {elbow,bic}]
                          [--force] [--solver {gurobi,cbc}]
                          [--timelimit TIMELIMIT]
                          [--fcn_ci_alpha FCN_CI_ALPHA]
                          [--min_ci_margin MIN_CI_MARGIN]
                          [--obj_type {imf,ci}] [--minClone MINCLONE]
                          [--maxClone MAXCLONE] [--diploid] [--tetraploid]
                          [--reg_term {RAW,MAXCN,DBOX_L1,DBOX_L0,DROOT_SUM,DADJ_SUM}]
                          [--reg_steps REG_STEPS] [--reg_bound REG_BOUND]
                          [--fix_cn_dip FIX_CN_DIP] [--fix_cn_tet FIX_CN_TET]
                          [--zero_cn_thres ZERO_CN_THRES] [--cd_tol CD_TOL]
                          [--no_ampdel] [--num_cnstates NUM_CNSTATES]
                          [-eD DIPLOIDCMAX] [-eT TETRAPLOIDCMAX]
                          [--min_prop MIN_PROP] [--purities PURITIES]
                          [--cd_niters CD_NITERS]
                          [--cd_convergence_iters CD_CONVERGENCE_ITERS]
                          [--cd_nseeds CD_NSEEDS] [--cd_njobs CD_NJOBS]
                          [--cd_seed CD_SEED] [--u_init {dirichlet,bubble}]
                          [--u_dir_alpha U_DIR_ALPHA]
                          [--solver_threads SOLVER_THREADS]
                          [--verbosity VERBOSITY] --genome_size GENOME_SIZE
                          --region_bed REGION_BED [--patient_id PATIENT_ID]
```

## Main parameters

Here we describe the main parameters; the full deconvolution and optimization parameter table follows.

| Parameter | Default | Description |
|---|---|---|
| `--mode` | `cd` | Solver mode: `cd`, `ilp`, or `both` |
| `--solver` | `gurobi` | ILP solver backend: `gurobi` or `cbc` |
| `--model_select` | `bic` | Clone-number/ploidy selection: `elbow` or `bic` |
| `--timelimit` | None | ILP solver time limit in seconds |
| `--obj_type` | `imf` | Fitting objective: `imf` (weighted L1) or `ci` (CI-violation hinge) |
| `--fcn_ci_alpha` | 0.05 | Significance level for the FCN confidence interval (0.05 -> 95% CI) |
| `--min_ci_margin` | 0.1 | Hard minimum CI half-width in FCN space |
| `--minClone` | 2 | Minimum number of tumor clones |
| `--maxClone` | 4 | Maximum number of tumor clones |
| `--diploid` | False | Solve under diploid assumption |
| `--tetraploid` | False | Solve under tetraploid/WGD assumption |
| `--reg_term` | `DBOX_L1` | Regularizer: `RAW`, `MAXCN`, `DBOX_L1`, `DBOX_L0`, `DROOT_SUM`, or `DADJ_SUM` |
| `--reg_steps` | 15 | Number of steps in the regularization path |
| `--reg_bound` | 0.15 | Maximum penalty weight for the regularization path |
| `--fix_cn_dip` | None | Fix diploid cluster CN states, e.g. `6:2|0;8:3|1` |
| `--fix_cn_tet` | None | Fix tetraploid cluster CN states, e.g. `6:4|2` |
| `--zero_cn_thres` | 0.005 | Clusters with weight ≥ this fraction of total cannot take a (0,0) CN state |
| `--no_ampdel` | False | Disable the amp/del symmetry constraint |
| `--num_cnstates` | -1 | Constrain the number of distinct CN states per clone (-1 = unconstrained) |
| `-eD` / `--diploidcmax` | 8 | Max copy number for diploid mode (0 = inferred from scaled FCN) |
| `-eT` / `--tetraploidcmax` | 12 | Max copy number for tetraploid mode (0 = inferred from scaled FCN) |
| `--min_prop` | 0.01 | Minimum clone proportion |
| `--purities` | None | Semicolon-separated `sample:purity` pairs; fixes normal-clone proportion to 1 − purity |
| `--cd_niters` | 10 | CD: max outer iterations per seed |
| `--cd_convergence_iters` | 2 | CD: consecutive convergence iterations required to stop |
| `--cd_tol` | 0.001 | CD: stop when U-step objective changes less than this |
| `--cd_nseeds` | 400 | CD: number of random restarts |
| `--cd_njobs` | 8 | CD: number of parallel worker processes |
| `--cd_seed` | 42 | CD: random seed for reproducibility |
| `--u_init` | `dirichlet` | U initialization: `dirichlet` or `bubble` |
| `--u_dir_alpha` | *(solver default)* | Dirichlet alpha for U initialization; lower = sparser |
| `--solver_threads` | *(solver default)* | Max threads per solver call (Gurobi); set to 1 for parallel CD workers |

### Fractional copy-number scaling factor estimation

compute-cn first estimates a per-sample RDR scaling factor (gamma) that maps read-depth ratios to fractional copy numbers (FCN), together with tumor purity and clonal CN anchors (`get_scaling_factor`), run separately for the no-WGD (diploid) and WGD (tetraploid) hypotheses. Balanced clusters are taken from the cluster-bins `is_balanced` calls; the user can override the anchors with `--fix_cn_dip` / `--fix_cn_tet`. The decision tree:

- **Baseline (s0).** A user-pinned balanced cluster (`--fix_cn_dip` `1|1` / `--fix_cn_tet` `2|2`), otherwise the largest balanced cluster; sets `gamma = 2 / RD(s0)`.
- **User-pinned anchor.** If a `--fix_cn_dip`/`--fix_cn_tet` cluster is imbalanced, it sets the imbalanced anchor directly.
- **Grid-fit check.** A candidate (purity, gamma) is accepted only if every cluster's observed RDR and BAF are representable by integer copy numbers up to the ploidy's maximum CN.
- **Anchor search.** Otherwise, imbalanced clusters are searched over enumerated CN candidates (filtered by purity validity, an RDR/BAF concordance test, and grid-fit), keeping the one with minimum RDR MSE.
- **Fallback (no-WGD).** If no imbalanced anchor is found, keep only the balanced baseline (s0 as (1,1)) and leave purity to the downstream deconvolution.
- **Fallback (WGD).** Otherwise read a second balanced cluster s1 as (1,1) against s0 as (2,2), giving `gamma = 2 / RD(s1)`, `purity = RD(s0) / RD(s1) - 1`; skipped if no valid s1.

The estimated per-(sample, ploidy) gamma is written to `results/gammas.tsv`.

### Distance-based Constrained Allele-specific Copy-number Factorization (D-CACF)
The (D-CACF) problem is solved by either ILP (`ilp`) or coordinate-descent (`cd` default) algorithm set by `--mode` using an external ILP solver (`--solver`), see [Setup-ILP-Solver](../../README.md#setup-ilp-solver).

#### Main model parameters
- **Maximum copy number (`-eD`/`--diploidcmax`, `-eT`/`--tetraploidcmax`).** Caps the per-segment integer CN at 8 (diploid) / 12 (tetraploid) by default; set to 0 to infer the cap from the scaled fractional copy numbers.

- **Purity and proportions (`--purities`, `--min_prop`).** `--purities` fixes each sample's normal fraction via `sample:purity` pairs; `--min_prop` (default 0.01) is the smallest clone proportion retained.

#### Regularization terms

A regularization term avoids overfitting the CN solution to observation noise. `--reg_term` selects the penalty; compute-cn sweeps `--reg_steps` (default 15) penalty weights up to `--reg_bound` (default 0.15) and picks the elbow of the fit-vs-penalty (Pareto) curve. `--num_cnstates` can additionally cap the number of distinct CN states per clone (`-1` = unconstrained).

For cluster $m$ (weight $w_m$), tumor clones $n = 1,\dots,N$ (clone $0$ = normal), with A/B-allele copy numbers $a_{m,n}$ / $b_{m,n}$:

| `--reg_term` | Penalty | Description |
|---|---|---|
| `RAW` | $0$ | No regularization; fit term only. |
| `DBOX_L1` (default) | $\sum_m w_m\left[(\max_n a_{m,n} - \min_n a_{m,n}) + (\max_n b_{m,n} - \min_n b_{m,n})\right]$ | Per-cluster allelic CN range (max - min) across tumor clones. |
| `DBOX_L0` | $\sum_m w_m \cdot \mathbf{1}[\text{span}_m > 0]$ | Per-cluster indicator of subclonality (nonzero allelic spread); $\text{span}_m$ is the `DBOX_L1` term for cluster $m$. |
| `MAXCN` | $\sum_m w_m (\max_n a_{m,n} + \max_n b_{m,n})$ | Per-cluster maximum tumor-clone total copy number. |
| `DROOT_SUM` | $\sum_m w_m \sum_n \left(\lvert a_{m,n} - a_{m,0}\rvert + \lvert b_{m,n} - b_{m,0}\rvert\right)$ | L1 distance of each tumor clone from the normal clone. |
| `DADJ_SUM` | $\sum_m w_m \sum_{n_1<n_2} \left(\lvert a_{m,n_1} - a_{m,n_2}\rvert + \lvert b_{m,n_1} - b_{m,n_2}\rvert\right)$ | Total pairwise L1 distance between tumor-clone pairs. |

### Ploidy and clone number

- **Ploidy (`--diploid`, `--tetraploid`).** **diploid** assumes no WGD, and **tetraploid** assumes one WGD. If neither flag is set, compute-cn solves both instances.

- **Number of clones (`--minClone`, `--maxClone`).** Integer CN and clone proportions are inferred for every clone count n in [`--minClone` (default 2), `--maxClone` (default 4)]. We recommend to set higher `--maxClone` if more than 1 tumor samples are provided.

- **Model selection (`--model_select`).** For each ploidy case, the clone number is picked by either BIC (default) or an elbow along the log-likelihood curve using the `kneed` library. The log-likelihood scores each candidate by the observed RDR (Gaussian) and phased allele counts (Beta-Binomial) given the deconvolved integer states and clone proportion. with per-cluster RDR variance and BAF dispersion provided from the seg file. Then, the ploidy with lower number of clones is chosen as final model-selected solution.

> [!TIP]
> Model selection is a heuristic. We recommend users to inspect the selection curves (`results/plots/model_selection.pdf`) alongside `results/summary.tsv`, and if another (ploidy, n) fits better, use its solution instead - every candidate is written to `results/chosen.<ploidy>.*.ucn` and `results/results.<ploidy>.n*.ucn.tsv`.

## Output

Copy-number solutions written to `--result_dir`: the model-selected `best.bbc.ucn` / `best.seg.ucn`, per-(ploidy, n) solutions, `gammas.tsv`, `summary.tsv`, and plots under `plots/`.

```
<result_dir>/                             # compute-cn output (--result_dir)
  best.bbc.ucn                            # model-selected CN solution (per-bin)
  best.seg.ucn                            # model-selected CN solution (per-segment)
  chosen.<ploidy>.bbc.ucn                 # best-n solution per ploidy (per-bin)
  chosen.<ploidy>.seg.ucn                 # best-n solution per ploidy (per-segment)
  results.<ploidy>.n*.bbc.ucn.tsv         # every (ploidy, n) solution (per-bin)
  results.<ploidy>.n*.seg.ucn.tsv         # every (ploidy, n) solution (per-segment)
  gammas.tsv                              # RDR scaling factors per sample and ploidy
  summary.tsv                             # fit/regularization metrics per solution
  sols/                                   # solver inputs + full pool of candidate solutions
    solver_input.<ploidy>.tsv             # per-(cluster, sample) FCN + CI + weights fed to the solver
    objectives.tsv                        # per-restart objectives: ploidy, n, sol_id, restart_id, imf_obj, reg_obj
    <ploidy>_n*/                          # one dir per (ploidy, n)
      <mode>_<sol_id>.tsv                 # one per candidate solution on the regularization path
      u0_seeds.tsv                        # U-initialization seeds for coordinate descent
  plots/
    scaling_2d.diploid.pdf                # RDR-vs-BAF scaling diagnostic (noWGD)
    scaling_2d.tetraploid.pdf             # RDR-vs-BAF scaling diagnostic (WGD, if inferred)
    model_selection.pdf                   # Pareto front + elbow/BIC selection page
    <ploidy>_n*/                          # per-(ploidy, n) plots
      <patient_id>.<ploidy>_n*.1D.pdf        # 1D genome-wide CN profile (selected solution)
      <patient_id>.<ploidy>_n*.1D.FCN_AB.pdf # allele-specific 1D profile
      <patient_id>.<ploidy>_n*.2D.pdf        # RDR-vs-BAF 2D scatter
      <patient_id>.pool_<ploidy>_n*.pdf      # pool panel of all Pareto (alternative) solutions
  compute-cn.log                          # run log (ends with the runtime/peak-RSS table)
```

The `*.bbc.ucn` files (`best.bbc.ucn`, `chosen.<ploidy>.bbc.ucn`,
`results.<ploidy>.n*.bbc.ucn.tsv`) contain every `bulk.bbc` column with additional clone proportions `u_<clone>` and A/B-allele copy numbers `cn_<clone>`. The `*.seg.ucn` files are segment-level tables that interpolates from `*.bbc.ucn` files with respect to `region.bed` boundaries.

| Added column | Description |
|---|---|
| `cn_normal` | Normal-clone allele-specific copy-number state `a\|b` (always `1\|1`) |
| `u_normal` | Normal-clone proportion in `SAMPLE` (1 - purity) |
| `cn_clone<i>` | Tumor clone i allele-specific integer CN state `a\|b` (major\|minor) for the bin's cluster |
| `u_clone<i>` | Proportion of tumor clone i in `SAMPLE` |

With `--wide_format`, the `.ucn` files use the same VCF-like layout with additional sample-independent `cn_<clone>` columns and sample-specific fields `u_<clone>` inside `FORMAT` field.

### Alternative solutions

For each `(ploidy, n)`, `compute-cn` explores a regularization path and keeps the **full pool** of
candidate solutions, not only the model-selected one. Each candidate is written under
`sols/<ploidy>_n<n>/` as `<mode>_<sol_id>.tsv` (`<mode>` is `cd`/`ilp`; `<sol_id>` encodes the
regularization weight, e.g. `cd_p0.0500_s0.tsv`). The pool is visualized as a single panel
`plots/<ploidy>_n<n>/<patient_id>.pool_<ploidy>_n<n>.pdf` (one row per Pareto solution, the selected
one marked `*`); `sols/objectives.tsv` records each solution's fit/regularization objectives and
`plots/model_selection.pdf` shows the Pareto front with the elbow/BIC pick.

Each candidate solution TSV has one row per `(cluster, sample)`:

| Column | Description |
|---|---|
| `CLUSTER`, `SAMPLE`, `#BINS` | Cluster ID, tumor sample, and number of bins in the cluster |
| `f_a`, `f_b` | Observed fractional copy number of the A / B allele (`RD * gamma`, split by BAF) |
| `exp_f_a`, `exp_f_b` | Expected fractional CN of the A / B allele under this solution (`sum_clone u * cn`) |
| `fa_lo`, `fa_hi`, `fb_lo`, `fb_hi` | Confidence-interval bounds on `f_a` / `f_b` |
| `cn_normal`, `u_normal`, `cn_clone<i>`, `u_clone<i>` | Per-clone CN state `a\|b` and proportion (interleaved) |
| `ci_accepted` | Whether `exp_f_a` / `exp_f_b` fall within the CI bounds |

To re-render or customize any single solution, pass its `results.<ploidy>.n*.bbc.ucn.tsv` (or a
`--solfile`) to the standalone `hatchet plot-cn`.
