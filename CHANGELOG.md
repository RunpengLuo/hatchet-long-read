# Changelog

## [3.0.0b2] - 2026-09-13

**Added**

- Memory-efficient VCF-like wide BBC/SEG layout gated by `--wide_format` ([io_utils.py:331](src/hatchet/io_utils.py#L331), [io_utils.py:422](src/hatchet/io_utils.py#L422)).
- `read_seg_file` returns per-(cluster, sample) DataFrames, replacing in-solver `build_data` ([io_utils.py:222](src/hatchet/io_utils.py#L222)).
- 2D-diagnostic plot styling parameters (`plot_diag_*`) for compute-cn ([hatchet.yaml:132](src/hatchet/hatchet.yaml#L132)).
- `--bal_lrt_margin` sets the cluster-level neutral zone of the balanced-cluster interval LRT, previously borrowed from `--bal_margin` ([hatchet_parser.py:266](src/hatchet/hatchet_parser.py#L266)).
- `--cna_plus_plus_d` exposes the exponent of the `D^l` adaptive-sampling weight in cna++ seeding ([hatchet_parser.py:90](src/hatchet/hatchet_parser.py#L90)).
- Negative-binomial RDR emission selected by `rdr_emission`, as an alternative to the Gaussian on per-bin counts: NB2 log-likelihood ([hmm_likelihoods.py:64](src/hatchet/cluster_bins/hmm/hmm_likelihoods.py#L64)), ECM dispersion/mean updates ([hmm_m_steps.py:171](src/hatchet/cluster_bins/hmm/hmm_m_steps.py#L171)), and emission-aware free-parameter counts ([hmm_utils.py:54](src/hatchet/cluster_bins/hmm/hmm_utils.py#L54)). Python backend only, and not yet reachable from the CLI.

**Changed**

- **Breaking:** the CLI, Snakemake config, and output filename prefix use `sample_id` in place of `patient_id` ([hatchet_parser.py](src/hatchet/hatchet_parser.py), [config/snakemake-hatchet.yaml:15](config/snakemake-hatchet.yaml#L15)).
- Matched-normal BB dispersion `tau` pools all normals, not just the first ([cluster_utils.py:47](src/hatchet/cluster_bins/cluster_utils.py#L47)).
- SEG cluster column renamed `#ID` -> `CLUSTER` to match the BBC key ([cluster_utils.py:297](src/hatchet/cluster_bins/cluster_utils.py#L297)).
- cluster-bins keeps per-K `(N, M)` arrays instead of an upfront long frame ([cluster_bins.py](src/hatchet/cluster_bins/cluster_bins.py)).
- `filenames.py` becomes `const.py`, which also holds the BBC/SEG column schema and the shared dtype/format helpers ([const.py:25](src/hatchet/const.py#L25)).
- The C++ EM loop batches `lgamma` evaluations and updates BAF means by Newton iteration instead of a scalar solve ([lgamma_batch.h](src/hatchet/cluster_bins/hmm/_hmm_cpp/lgamma_batch.h), [m_steps.cpp](src/hatchet/cluster_bins/hmm/_hmm_cpp/m_steps.cpp)).
- Plot parameters namespaced under `plot_*` (e.g. `--img_type` -> `--plot_img_type`) ([hatchet_parser.py:749](src/hatchet/hatchet_parser.py#L749)).
- compute-cn per-(ploidy, n) plots default to `pdf` ([config/snakemake-hatchet.yaml](config/snakemake-hatchet.yaml)).
- `scaling_2d.pdf` is split into one file per ploidy branch ([plot_compute_cn.py:160](src/hatchet/plot/plot_compute_cn.py#L160)).
- The Pareto plot annotates its x-axis with the active regularization formula ([plot_compute_cn.py](src/hatchet/plot/plot_compute_cn.py)).
- `--bal_margin` now only sets the per-bin `|BAF - 0.5|` band selecting the diploid RDR anchor during seeding; a bin's BAF standard deviation and a cluster's BAF standard error differ by orders of magnitude, so one value could not serve both ([hmm_init.py:148](src/hatchet/cluster_bins/hmm/hmm_init.py#L148)).
- cluster-bins 1D mhBAF panel colored by cluster, matching the RDR panel ([plot_cluster_bins.py:302](src/hatchet/plot/plot_cluster_bins.py#L302)).
- `scripts/overwrite_result.py` becomes `scripts/overwrite_bbc_seg.py` and segments through the region-aware `segmentation()` ([scripts/overwrite_bbc_seg.py](scripts/overwrite_bbc_seg.py)).
- `docs/reference.md` folded into each `docs/modules/<command>.md`.

**Fixed**

- cna++ seeding took the log of an already-logged RDR under `--log_rdr` ([hmm_init.py](src/hatchet/cluster_bins/hmm/hmm_init.py)).
- The diploid branch pinned purity from a noWGD pair that may not exist; it is now left free in that case ([scaling.py](src/hatchet/compute_cn/scaling.py)).
- The Snakefile mirrors `wide_format` from cluster-bins into compute-cn, so BBC/SEG output names agree across rules ([Snakefile:60](Snakefile#L60)).
- CBC 2.10.13 aborts on its `-mipstart` warm-start file and on some degenerate instances; warm start is skipped for CBC and a solver crash is reported as a failed solve rather than killing the run ([inference.py](src/hatchet/compute_cn/solve/inference.py)).
- Confine `plot_scaling_2d` whitegrid style to `rc_context`, restoring 1D axes borders ([plot_compute_cn.py:187](src/hatchet/plot/plot_compute_cn.py#L187)).
- Fix mhBAF folding for k-means pre-initialization ([hmm_utils.py:92](src/hatchet/cluster_bins/hmm/hmm_utils.py#L92)).

**Removed**

- Experimental `cnt_cd` solver mode; recoverable from commit `3f97f73e` ([inference.py](src/hatchet/compute_cn/solve/inference.py)).
- CLI flags `--u_bin_p` and `--show_prop` ([hatchet_parser.py](src/hatchet/hatchet_parser.py)).
- Separate `runtime.log`; profiling now appended to the per-command log ([utils.py](src/hatchet/utils.py)).

## [3.0.0b1] - 2026-07-28

**Changed**

- **Gaussian RDR + Beta-Binomial BAF factorial HMM with phase switch correction.** This replaces HATCHet2's [`combine_counts.py`](https://github.com/raphael-group/hatchet/blob/84ebfbac765a8329899a3d03c3791324bbb1fe3e/src/hatchet/utils/combine_counts.py) and [`cluster_bins.py`](https://github.com/raphael-group/hatchet/blob/84ebfbac765a8329899a3d03c3791324bbb1fe3e/src/hatchet/utils/cluster_bins.py). The new segmentation algorithm models phasing switch errors as prior information through genetic-map-derived switch probabilities, and is implemented with a C++/OpenMP backend (`run_baum_welch`, [hmm_model.py:142](src/hatchet/cluster_bins/hmm/hmm_model.py#L142)).
- **HMM model selection.** Added ICL scoring to account for cluster-assignment uncertainty when selecting the cluster number K (`model_select_K`, [hmm_utils.py:6](src/hatchet/cluster_bins/hmm/hmm_utils.py#L6)).
- **Balanced-cluster identification.** Each HMM-decoded cluster is tested for allelic balance (BAF≈0.5) by an interval likelihood-ratio test with parametric bootstrap on a symmetric Beta-Binomial mixture (`label_balanced_clusters`, [cluster_utils.py:316](src/hatchet/cluster_bins/cluster_utils.py#L316)). The `is_balanced` column in the `SEG.UCN` file marks the chosen balanced clusters. This replaces HATCHet2's user-defined hard cutoff `diploidbaf` ([cluster_bins.py:446](https://github.com/raphael-group/hatchet/blob/84ebfbac765a8329899a3d03c3791324bbb1fe3e/src/hatchet/utils/cluster_bins.py#L446)).
- **Scaling factor estimation.** A **decision-tree strategy** identifies a clonal imbalanced cluster to infer the RDR scaling factor (gamma), tumor purity, and clonal CN anchors (`get_scaling_factor`, [scaling.py:8](src/hatchet/compute_cn/scaling.py#L8)). This replaces HATCHet2's `scale_rdr`, which derived a single purity/scale in closed form from user-supplied `--clonal` cluster copy numbers ([solve/utils.py:63](https://github.com/raphael-group/hatchet/blob/84ebfbac765a8329899a3d03c3791324bbb1fe3e/src/hatchet/utils/solve/utils.py#L63)).
  - **Baseline.** A user-pinned balanced cluster (`--fix_cn_dip` `1|1` / `--fix_cn_tet` `2|2`), or otherwise the largest balanced cluster.
  - **User-pinned anchor.** If provided, a `--fix_cn_dip`/`--fix_cn_tet` cluster sets the imbalanced anchor directly.
  - **Grid-fit check.** Shared filter accepting a candidate (purity, gamma) only if every cluster's observed RDR and BAF fall within the ranges representable by integer copy numbers up to the ploidy's maximum CN under that scaling.
  - **Anchor search.** Otherwise, imbalanced clusters are searched over enumerated CN candidates, filtered by purity validity, a statistical RDR/BAF concordance test, and grid-fit, then pick the candidate imbalanced cluster with minimum MSE between observed and CN-expected RDR.
  - **Fallback (no-WGD).** If no imbalanced anchor is found, keep only the balanced baseline (s0 read as (1,1)) and leave purity to be inferred by the downstream deconvolution.
  - **Fallback (WGD).** If no imbalanced anchor is found, read a second balanced cluster s1 as (1,1) against the baseline s0 as (2,2), giving gamma = 2/RD(s1) and purity = RD(s0)/RD(s1) - 1. If no such s1 gives a valid bounded purity and grid-fit, WGD path is skipped.
- **Clone-number model selection.** For each ploidy case, the clone number is picked by an elbow on the log-likelihood (or BIC) vs. clone-number curve using the `kneed` library (`model_selection_ploidy`, [model_select.py:137](src/hatchet/compute_cn/model_select.py#L137)). This replaces HATCHet2's model selection against ILP objectives.
- **[Universal-Genotyping-Pipeline](https://github.com/raphael-group/Universal-Genotyping-Pipeline).** This replaces the error-prone preprocessing sub-modules in HATCHet2: [count_alleles.py](https://github.com/raphael-group/hatchet/blob/84ebfbac765a8329899a3d03c3791324bbb1fe3e/src/hatchet/utils/count_alleles.py), [count_reads.py](https://github.com/raphael-group/hatchet/blob/84ebfbac765a8329899a3d03c3791324bbb1fe3e/src/hatchet/utils/count_reads.py), and [phase_snps.py](https://github.com/raphael-group/hatchet/blob/84ebfbac765a8329899a3d03c3791324bbb1fe3e/src/hatchet/utils/phase_snps.py).
- **[Modular Snakemake workflow](./Snakefile).** One rule per stage replaces HATCHet2's monolithic [run.py:30-31, 211-268](https://github.com/raphael-group/hatchet/blob/84ebfbac765a8329899a3d03c3791324bbb1fe3e/src/hatchet/utils/run.py#L211-L268) driver.
- **1D/2D plotting library.** Genome-wide 1D RDR/BAF tracks and a 2D RDR-vs-BAF joint scatter
  with cluster coloring, marginal distribution, gap masking, and density-based transparency (`plot_1d`, `plot_2d`), since extracted into the
  [cnplot](https://github.com/raphael-group/cnplot) library that [plot/](src/hatchet/plot/) now builds on.

**Added**

- **Integer CN deconvolution with regularization.** CN deconvolution solver is extended with multiple selectable regularizers (`MAXCN`, `DBOX_L1`/`DBOX_L0`, `DROOT_SUM`, `DADJ_SUM`, or unregularized `RAW`, default: `DBOX_L1`) to avoid overfitting the observation noise. The pareto-front solution is picked using the `kneed` library.
  - `MAXCN`: penalizes the per-cluster maximum total copy number.
  - `DBOX_L{0,1}`: penalizes the `L{0,1}` distance for maximum and minimum allele CN for tumor clones per cluster.
  - `DROOT_SUM`: penalizes the L1 distance of each tumor clone's copy numbers from the normal clone.
  - `DADJ_SUM`: penalizes the total pairwise L1 distance between all tumor-clone pairs.