# `cluster-bins`
`cluster-bins` performs local-global genome segmentation using a Gaussian RDR + Beta-Binomial BAF multi-sample factorial HMM with phase switch correction.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `HATCHET_DISABLE_CPP` | `0` | When set to `1`, `true`, or `yes` (case-insensitive), `cluster-bins` uses the pure-Python (Numba) HMM backend instead of the compiled C++ extension (`_hmm_cpp`). |

## Input

The preprocessed bin-by-sample matrices in `--bb_dir` (RDR, phased allele counts, bin metadata, and the sample table), plus the reference `--genome_size` and `--region_bed`.

| Parameter | Default | Description |
|---|---|---|
| `--bb_dir` | *(required)* | Input directory: NPZ count matrices and `bb.tsv.gz` |
| `--bbc_dir` | *(required)* | Output directory for BBC/SEG TSV files |
| `--genome_size` | *(required)* | Reference chromosome sizes file |
| `--wide_format` | False | EXPERIMENTAL: write BBC as a space-efficient wide gzip TSV (`bulk.bbc.tsv.gz`) and `bulk.seg` in the matching VCF-like wide layout |
| `--force` | False | Re-run even if results already exist (default: skip existing) |
| `--verbosity` | 0 | Verbose level: 0, 1, or 2 |

```
<bb_dir>/
  bb.tsv.gz         # BED format bin meta-informations, one row per bin, aligned with matrix rows.
  sample_ids.tsv    # dataset meta-informations, one row per sample, aligned with matrix columns.
                    #   columns: SAMPLE, sample_type (normal|tumor), [assay_type]
  bb.rdr.npz        # (bins, tumor samples) float  - NumPy format, read-depth ratio, tumor samples only
  bb.depth.npz      # (bins, samples) float        - NumPy format, read depth, all samples
  bb.Aallele.npz    # (bins, samples) int          - NumPy format, A-haplotype allele counts
  bb.Ballele.npz    # (bins, samples) int          - NumPy format, B-haplotype allele counts
  bb.Tallele.npz    # (bins, samples) int          - NumPy format, total allele counts
```

`bb.tsv.gz` columns (one row per bin; row order defines the matrix row order):

| Column | Description |
|---|---|
| `#CHR` | Chromosome of the bin |
| `START` / `END` | Bin genomic interval (bp) in BED format (0-index, left-close right-open) |
| `region_id` | Region/segment identifier; bins are grouped into segments by this column (HMM decoding runs per segment) |
| `switchprobs` | Per-bin phase switch transition probability |
| `#SNPS` | Number of heterozygous SNPs in the bin |

`sample_ids.tsv` columns (one row per sample; row order defines the matrix column order):

| Column | Required | Description |
|---|---|---|
| `SAMPLE` | yes | Sample name; also the matrix column label |
| `sample_type` | yes | `normal` or `tumor`; the tumor-only `bb.rdr.npz` holds just the `tumor` columns |
| `assay_type` | no | Optional sequencing-assay label; samples are grouped by assay |

## Usage

```console
$ hatchet cluster-bins --help
usage: hatchet cluster-bins [-h] --bb_dir BB_DIR --bbc_dir BBC_DIR [--force]
                            [--minK MINK] [--maxK MAXK] [-t T]
                            [--restarts RESTARTS]
                            [--top_restarts TOP_RESTARTS]
                            [--n_local_trials N_LOCAL_TRIALS]
                            [--niters NITERS] [--decode_method {viterbi,map}]
                            [--score_method {bic,icl}]
                            [--score_criteria SCORE_CRITERIA]
                            [--init_method {cna_plus_plus,kmeans_plus_plus}]
                            [--free_baf_c0]
                            [--training_method {baum_welch,viterbi}]
                            [--min_tau MIN_TAU] [--max_tau MAX_TAU]
                            [--share_tau | --no-share_tau] [--baf_eps BAF_EPS]
                            [--min_covar MIN_COVAR] [--ig_alpha IG_ALPHA]
                            [--tau_iters TAU_ITERS] [--seed SEED] [--log_rdr]
                            --genome_size GENOME_SIZE --region_bed REGION_BED
                            [--verbosity VERBOSITY]
                            [--bal_lrt_alpha BAL_LRT_ALPHA]
                            [--bal_margin BAL_MARGIN]
                            [--filter_std FILTER_STD] [--min_nbins MIN_NBINS]
                            [--ub_nbins UB_NBINS] [--skip_mhbafs]
```

## Main parameters

Here we describe the main parameters; the full HMM and model-selection parameter table follows.

| Parameter | Default | Description |
|---|---|---|
| `--minK` | 3 | Minimum number of HMM cluster states |
| `--maxK` | 30 | Maximum number of HMM cluster states |
| `-t` | 1e-6 | Initial off-diagonal transition mass |
| `--restarts` | 10 | Number of random restarts per K |
| `--top_restarts` | *(= restarts)* | Number of top-scoring inits to run full EM on |
| `--n_local_trials` | 3 | Candidate bins evaluated per k-means++ seeding step |
| `--niters` | 50 | Number of EM iterations per restart |
| `--decode_method` | `map` | HMM decoding: `viterbi` (most-likely path) or `map` (marginal per-bin posterior) |
| `--score_method` | `icl` | Model selection score: `bic` or `icl` |
| `--score_criteria` | `min` | How to pick K from the score curve: `min`, `elbow`, or `margin-<int>` |
| `--init_method` | `cna_plus_plus` | Init method: `cna_plus_plus` (HMM-aware) or `kmeans_plus_plus` (sklearn KMeans++) |
| `--training_method` | `baum_welch` | HMM training: `baum_welch` (soft EM) or `viterbi` (hard EM) |
| `--free_baf_c0` | False | Allow cluster-0 BAF to update during EM (default: fixed at 0.5) |
| `--min_tau` | 1 | Minimum Beta-Binomial dispersion tau |
| `--max_tau` | 1e6 | Maximum Beta-Binomial dispersion tau |
| `--share_tau` | True | Share BB dispersion tau across clusters within a sample (`--no-share_tau` for per-cluster) |
| `--tau_iters` | 3 | Number of EM iterations during which BAF dispersion tau is updated |
| `--baf_eps` | 1e-3 | BAF mean Brent search bounds `[baf_eps, 1-baf_eps]`; sequencing error floor |
| `--min_covar` | 1e-3 | Minimum RDR variance floor applied after each M-step |
| `--ig_alpha` | 10.0 | Inverse-gamma prior shape parameter for RDR variance updates |
| `--log_rdr` | False | Use log(RDR) instead of raw RDR in the Gaussian emission |
| `--seed` | 42 | Random seed for HMM init step |
| `--bal_lrt_alpha` | 0.05 | Significance level for the balanced-cluster interval LRT |
| `--bal_margin` | 0.03 | Half-width of neutral zone `[0.5-δ, 0.5+δ]` for the balanced-cluster test |
| `--filter_std` | 2.0 | Filter clusters whose variance deviates from mean by `filter_std × std` |
| `--min_nbins` | 10 | Remove clusters with fewer than `min_nbins` bins |
| `--ub_nbins` | 50 | Variance-outlier filtering only applies to clusters with #bins ≤ `ub_nbins` |
| `--skip_mhbafs` | False | Skip minor-haplotype BAF folding after decoding |

### HMM inference

- **Number of clusters (`--minK`, `--maxK`).** cluster-bins fits the factorial HMM for every K in the closed interval [`--minK` (default 3), `--maxK` (default 30)] and selects the best K by a model score (below). Widen the range if the chosen K lands at either end; narrow it to save runtime when the expected number of distinct states is known.

- **Off-diagonal transition mass (`-t`, default 1e-6).** The off-diagonal transition cost of the HMM transition matrix balances *global* information (RDR/BAF shared across samples) against *local* information (keeping adjacent bins in the same segment). Smaller `-t` favors local continuity (smoother, more contiguous segments); larger `-t` favors global grouping. Reduce it by orders of magnitude for noisier or low-coverage data.

- **Model selection (`--score_method`).** The number of clusters (`K`) is chosen based on model-selection over `--score_method` (`icl`, default, or `bic`). `icl` penalizes overlapping clusters compared `bic` using additonal posterior entropy cost, so it tends to return fewer, better-separated clusters.

> [!TIP]
> Model selection is a heuristic, not a guarantee. We recommend the user to inspect the model-selection score curve plot (`bbc/plots/model_scores.pdf`) alongside the per-K RDR-BAF 1D/2D clustering plots (`bbc/plots/K<K>.pdf`) to see if a better fitted solution may exist and use that solution instead for `compute-cn` - every solution is written to `bbc/labels/bulk<K>.bbc` / `bbc/labels/bulk<K>.seg`.

### Balanced-cluster detection

After HMM state decoding, each cluster is tested for being allelic balanced (BAF = 0.5) or not with an interval likelihood-ratio test on the raw allele counts. `--bal_margin` (default 0.03) is the half-width of the neutral zone `[0.5-δ, 0.5+δ]` treated as a balanced cluster, and `--bal_lrt_alpha` (default 0.05) is the significance level; a cluster is called *balanced* only if it passes in every sample. These clusters anchor the BAF baseline used downstream by `compute-cn`.

> [!TIP]
> Under strong reference-mapping bias the BAF of truly balanced clusters can be pulled off 0.5, so the automatic test may failed to detect them. In this case, we recommend users to manually inspect the BAF centers using 1D/2D clustering plots and decide the balanced cluster by toggling the values of `is_balanced` column (True/False) under `seg` file.

### Post-cluster filtering

Outlier clusters are removed before output: any cluster with fewer than `--min_nbins` bins (default 10) is dropped, and among small clusters (<= `--ub_nbins` bins, default 50) those whose variance exceeds the mean by more than `--filter_std` standard deviations (default 2.0) are treated as variance outliers and removed. Raise `--min_nbins` or lower `--filter_std` to filter more aggressively.

## Output

Clustering results written to `--bbc_dir`: the model-selected $K$ cluster results (`bulk.bbc`, `bulk.seg`), per-$K$ sweeps under `labels/`, `model_scores.tsv`, and diagnostic plots under `plots/`.

```
<bbc_dir>/                         # cluster-bins output (--bbc_dir)
  bulk.bbc                         # per-bin cluster assignments (optimal K)
  bulk.seg                         # per-cluster summary statistics (optimal K)
  labels/
    bulk<K>.bbc                    # per-bin assignments for each swept K
    bulk<K>.seg                    # per-cluster summary for each swept K
  cluster_infos/                   # per-K cluster-label TSVs
  plots/                           # ELBO traces, RDR-BAF scatter, model score
  model_scores.tsv                 # BIC and ICL scores across K
  cluster-bins.log                 # run log (ends with the runtime/peak-RSS table)
```

`bulk.bbc` (per-bin, one row per bin x tumor sample) columns:

| Column | Description |
|---|---|
| `#CHR` | Chromosome of the bin |
| `START` / `END` | Bin genomic interval (bp) in BED format (0-index, left-close right-open) |
| `SAMPLE` | Tumor sample name |
| `#SNPS` | Number of heterozygous SNPs in the bin |
| `CLUSTER` | Cluster ID assigned to the bin |
| `RD` | Read-depth ratio (RDR) of the bin in `SAMPLE` |
| `COV` | Mean per-base coverage of the bin in `SAMPLE` |
| `BAF` | minor-haplotype B-allele frequency (mhBAF) of the bin in `SAMPLE` |
| `ALPHA` | Phased major-allele read count (`Tallele - BETA`) |
| `BETA` | Phased minor-haplotype read count |

With `--wide_format`, `bulk.bbc` is instead a single gzip TSV `bulk.bbc.tsv.gz` (one row per bin) in
a VCF-like layout (`region_id`/`switchprobs` stay in `bb.tsv.gz`):

| Column | Description |
|---|---|
| `#CHR`, `START`, `END`, `#SNPS`, `CLUSTER` | Bin-level columns, as in `bulk.bbc` |
| `PHASE` | Haplotype phase orientation of the bin (0/1), from `bb.phased.tsv.gz` |
| `PHASE_POSTS` | Posterior probability of the bin's phase |
| `FORMAT` | Colon-joined per-sample field keys, `RD:COV:BAF:ALPHA:BETA` |
| `<sample>` | One column per tumor sample; the `FORMAT` fields colon-joined in that order |

`bulk.seg` (per-cluster, one row per cluster x tumor sample) columns:

| Column | Description |
|---|---|
| `CLUSTER` | Cluster ID |
| `SAMPLE` | Sample ID |
| `#BINS` | Number of bins |
| `#SNPS` | Number of gHETs |
| `LENGTH` | Total lengths in bp |
| `ALPHA` / `BETA` | Summed phased major / minor-haplotype counts |
| `COV` | Normalized sequencing coverage |
| `BAF` | Cluster minor-haplotype BAF |
| `BAF-se` | Standard error of `BAF` |
| `BAF-tau` | Beta-Binomial dispersion |
| `RD` | Cluster RDR |
| `RD-se` | Standard error of `RD` |
| `RD-var` | RDR variance |
| `is_balanced` | Whether the cluster is labeled allelic balanced (BAF near 0.5) |
| `is_filtered` | Whether the cluster is filtered out (excluded from `compute-cn`) |

With `--wide_format`, `bulk.seg` (and `labels/bulk<K>.seg`) uses the same VCF-like layout as the
wide BBC - one row per cluster, still named `bulk.seg` (plain TSV, not gzipped; the `FORMAT` column
makes it self-describing):

| Column | Description |
|---|---|
| `CLUSTER`, `#BINS`, `#SNPS`, `LENGTH` | Cluster-level columns, as in the long `bulk.seg` |
| `is_balanced`, `is_filtered` | Cluster-level flags, as in the long `bulk.seg` |
| `FORMAT` | Colon-joined per-sample field keys, `ALPHA:BETA:COV:BAF:BAF-se:BAF-tau:RD:RD-se:RD-var` |
| `<sample>` | One column per tumor sample; the `FORMAT` fields colon-joined in that order |
