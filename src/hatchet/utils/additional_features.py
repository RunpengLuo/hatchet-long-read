import pandas as pd
import numpy as np

import os
import gzip
import subprocess

from hatchet.utils.Supporting import (
    ensure,
    log,
    error,
    which,
    url_exists,
    bcolors,
    numericOrder,
)

def sort_chroms(chromosomes: list):
    assert len(chromosomes) != 0
    use_chr  = True if str(chromosomes[0]).startswith("chr") else False
    if not use_chr:
        return sorted(chromosomes)
    chr2ord = {}
    for i in range(1,23):
        chr2ord[f"chr{i}"] = i
    chr2ord["chrX"] = 23
    chr2ord["chrY"] = 24
    return sorted(chromosomes, key=lambda x: chr2ord[x])

"""
    check if all chromosomes with prefix chr or not
    return True if all prefix with chr
    return False if all prefix without chr, or the input is empty
"""
def use_chr_prefix(chromosomes: list):
    # Check that chr notation is consistent across chromosomes
    using_chr = [a.startswith("chr") for a in chromosomes]
    if any(using_chr):
        ensure(
            all(using_chr),
            "Some chromosomes use chr notation while others do not.",
        )
        return True
    return False

def has_header(file: str, header_prefix=["CHR", "Chr", "CHROMOSOME", "#ID"]):
    fd = gzip.open(file, "rt") if file.endswith(".gz") else open(file, "r")
    first_line = fd.readline()
    fd.close()
    for hp in header_prefix:
        if first_line.startswith(hp):
            return True
    return False

# load segment file in BED format with 1-indexed and left-close right-open format
# CHR\tSTART\tEND\t...
def load_seg_file(seg_file: str, use_chr_bam: bool, additional_columns=[]):
    num_columns = 3 + len(additional_columns)
    if has_header(seg_file):
        seg_df = pd.read_csv(seg_file, sep="\t")
    else:
        seg_df = pd.read_csv(seg_file, sep="\t", header=None,
                             usecols=range(num_columns),
                             names=["CHR", "START", "END"] + additional_columns)

    seg_df["CHR"] = seg_df["CHR"].astype("str")
    use_chr_seg = use_chr_prefix(seg_df["CHR"].tolist())
    if use_chr_bam and not use_chr_seg:
        seg_df["CHR"] = seg_df["CHR"].apply(lambda s: "chr" + s)
    if not use_chr_bam and use_chr_seg:
        seg_df["CHR"] = seg_df["CHR"].apply(lambda s: s[3:])

    chs = seg_df["CHR"].unique().tolist()
    return seg_df, chs

"""
Reads haplotype blocks from gtf file
"""
def load_gtf_file(gtf_file: str):
    return pd.read_csv(
        gtf_file, sep="\t", header=None, comment="#",
        names=[
            "CHR",
            "source",
            "feature",
            "START",
            "END",
            "score",
            "strand",
            "frame",
            "attribute",
        ],
    )

# only load bed format dataframe
# original gtf format use 1-based inclusive indexing
# convert to 0-indexing left-open right-close format
# https://genome.ucsc.edu/FAQ/FAQformat.html#format4
def load_gtf_file_bed(gtf_file: str):
    df = load_gtf_file(gtf_file)[["CHR", "START", "END"]]
    df.loc[:, "START"] = df.START - 1
    return df[["CHR", "START", "END"]]

def load_mosdepth_files(sample_names: list, mosdepth_files: list):
    bed_mosdepths = []
    for sname, sbed_file in zip(sample_names, mosdepth_files):
        with gzip.open(sbed_file, "rt") as f:
            bed_data = pd.read_csv(f, sep="\t", 
                                   names=["CHR", "START", "END", "AVG_DEPTH"])
            bed_data["sample"] = sname
            bed_mosdepths.append((sname, bed_data))
    return bed_mosdepths

# bb/bulk.bb
def init_bb_dataframe():
    bb_column_names = {
        "CHR": "str",
        "UNIT": "str",
        "START": "int",
        "END": "int",
        "SAMPLE": "str",
        "RD": "float",
        "TOTAL_READS": "int",
        "NORMAL_READS": "int",
        "SNPS": "int",
        "BCOUNT": "int",
        "TOTAL_SNP_READS": "int",
        "HAPLO": "str",
        "SNP_POS": "int",
        "SNP_REF_COUNTS": "int",
        "SNP_ALT_COUNTS": "int",
        "BAF": "float",
        "BLOCK_START": "int",
        "BLOCK_END": "int"
    }

    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in bb_column_names.items()})

# bb/bulk.bb, same format as HATCHetv2
# #CHR	START	END	SAMPLE	RD	#SNPS	COV	ALPHA	BETA	BAF	TOTAL_READS	NORMAL_READS	
# ORIGINAL_BAF	CORRECTED_READS	GC	UNCORR_RD	GCCORR
def init_bb_dataframe_v2():
    bb_column_names = {
        "CHR": "str",
        "START": "int",
        "END": "int",
        "SAMPLE": "str",
        "RD": "float",
        "#SNPS": "int",
        "COV": "float",
        "ALPHA": "int",
        "BETA": "int",
        "BAF": "float",
        "TOTAL_READS": "int",
        "NORMAL_READS": "int",
        "ORIGINAL_BAF": "float",
        "CORRECTED_READS": "int",
        "UNCORR_RD": "float",
        "GC": "float",
        "GCCORR": "float",
    }

    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in bb_column_names.items()})

def init_summary_dataframe():
    bb_column_names = {
        "CHR": "str",
        "BLOCK_START": "int",
        "BLOCK_END": "int",
        "BLOCK_LEN": "int",
        "THRES_START": "int",
        "THRES_END": "int",
        "NUM_THRESHOLDS": "int",
        "NUM_SNPS": "int",
        "NUM_BIN": "int"
    }

    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in bb_column_names.items()})



def get_array_file_path(dirname: str, ch: str, uncompressed=False):
    ret1 = os.path.join(dirname, f"{ch}.total.gz")
    ret2 = os.path.join(dirname, f"{ch}.thresholds.gz")
    if uncompressed:
        return [ret1[:-3], ret2[:-3]]
    return [ret1, ret2]

# return all non-exists files
def check_array_files(dirname: str, chromosomes: list):
    expected = [os.path.join(dirname, "samples.txt")]
    for ch in chromosomes:
        [ptotal, pthres] = get_array_file_path(dirname, ch)
        expected.extend([ptotal, pthres])
    return [a for a in expected if not os.path.isfile(a)]

def expected_mosdepth_files(dirname: str, sample_names: list, use_region: bool):
    mosdepth_suffixes = [".mosdepth.global.dist.txt", 
                       ".mosdepth.summary.txt"]
    if use_region:
        mosdepth_suffixes += [".regions.bed.gz",
                            ".regions.bed.gz.csi"]
    else:
        mosdepth_suffixes += [".per-base.bed.gz", 
                            ".per-base.bed.gz.csi"]
    expected = []
    for name in sample_names:
        expected.extend([os.path.join(dirname, f"{name}{sfx}") for sfx in mosdepth_suffixes])
    return expected

def expected_starts_files(dirname: str, chromosomes: list, sample_names: list):
    expected = []
    for name in sample_names:
        expected.extend([os.path.join(dirname, f"{name}.{ch}.starts.gz") for ch in chromosomes])
    return expected

def expected_count_files(dirname: str, chromosomes: list, sample_names: list, use_region=False):
    expected_starts = expected_starts_files(dirname, chromosomes, sample_names)
    expected_mosdp = expected_mosdepth_files(dirname, sample_names, use_region)
    return expected_starts + expected_mosdp

def check_count_files(dirname: str, chromosomes: list, sample_names: list, use_region=False):
    return [a for a in expected_count_files(dirname, chromosomes, sample_names, use_region) if not os.path.isfile(a)]

# at most <max_threads> per task is assigned if sum <= nproc
# use <= 1 process per task
def workload_assignment(nproc: int, ntask: int, max_threads=4):
    nworker = min(nproc, ntask)
    threads_per_task = None
    if nproc > ntask:
        nthread_per_task = min(nproc // ntask, max_threads)
        threads_per_task = [nthread_per_task] * ntask
        if nthread_per_task < max_threads:
            for i in range(nproc % ntask):
                threads_per_task[i] += 1
    else:
        threads_per_task = [1] * ntask
    return nworker, threads_per_task

def count_comment_lines(filename: str, comment_symbol="#"):
    num_header = 0
    if filename.endswith(".gz"):
        fd = gzip.open(filename, "rt")
    else:
        fd = open(filename, "r")
    for line in fd:
        if line.startswith(comment_symbol):
            num_header += 1
        else:
            break
    fd.close()
    log(f"Number of skipped header line={num_header} in {filename}\n", level="DEBUG")
    return num_header

def load_vcf_file(vcf_file: str, read_type: str, phased=True):
    num_comment = count_comment_lines(vcf_file, "#")
    vcf_header = ["CHR", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", "PHASE"]
    # skiprows is necessary versus comment
    # since vcf main content field may still contains comment symbol
    df = pd.read_table(vcf_file, compression="gzip", sep="\t", 
                       names=vcf_header,
                       usecols=["CHR", "POS", "PHASE"],
                       quoting=3,
                       low_memory=False,
                       dtype={"CHR": object, "POS": np.uint32},
                       skiprows=num_comment)

    if phased:
        df["FLIP"] = df.PHASE.str.contains("1|0", regex=False).astype(np.int8)
        df["NOFLIP"] = df.PHASE.str.contains("0|1", regex=False).astype(np.int8)
        # Drop entries without phasing output
        df = df[df.FLIP + df.NOFLIP > 0]
        # For exact duplicate entries, drop one
        df = df.drop_duplicates()
        # For duplicate entries with the same (CHR, POS) but different phase, drop all
        df = df.drop_duplicates(subset=["CHR", "POS"], keep=False)
    return df

# FIXME to be simplified
"""
# TODO move it to separate file since it is commonly used by two module
return:
1. SNP absolute positions
2. SNP total counts in 2D matrix with dim #SNPs * #samples
3. SNP dataframe
"""
def read_snps(baf_file: str, ch: str, all_names: list, phasefile=None, read_type="EMPTY"):
    """
    Read and validate SNP data for this patient (TSV table output from HATCHet deBAF.py).
    """
    # remove normal sample -- not looking for SNP counts from normal
    all_names = [name for name in all_names if name != "normal"]
    # Read in HATCHet BAF table
    all_snps = pd.read_table(
        baf_file,
        header=None,
        names=["CHR", "POS", "SAMPLE", "REF", "ALT", "REFC", "ALTC"],
        dtype={
            "CHR": object,
            "POS": np.uint32,
            "SAMPLE": object,
            "ALT": np.uint32,
            "REF": np.uint32,
            "REFC": object,
            "ALTC": object,
        },
    )

    # Keep only SNPs on this chromosome
    snps = all_snps[all_snps.CHR == ch].sort_values(by=["POS", "SAMPLE"], ignore_index=True)
    if len(snps) == 0:
        raise ValueError(
            error(f"Chromosome {ch} not found in SNPs file (chromosomes in file: {all_snps.CHR.unique()})")
        )

    snps_samples = snps.SAMPLE.unique().tolist()
    if len(all_names) != len(snps_samples) or set(all_names) != set(snps_samples):
        raise ValueError(
            error(f"Expected sample names did not match sample names in SNPs file.\n\
                Expected: {sorted(all_names)}\n  Found:{sorted(snps.SAMPLE.unique())}")
        )

    # Add total counts column
    snps["TOTAL"] = snps.ALT + snps.REF

    if phasefile != None:
        phased_df = load_vcf_file(phasefile, read_type, phased=True)
        # Merge tables: keep only those SNPs for which we have phasing output
        snps = pd.merge(snps, phased_df, on=["CHR", "POS"], how="left")

    # Create counts array and find SNPs that are not present in all samples
    snp_counts = snps.pivot(index="POS", columns="SAMPLE", values="TOTAL")
    missing_pos = snp_counts.isna().any(axis=1)

    # Remove SNPs that are absent in any sample
    snp_counts = snp_counts.dropna(axis=0)
    snps = snps[~snps.POS.isin(missing_pos[missing_pos].index)]

    # Pivot table for dataframe should match counts array and have no missing entries
    check_pivot = snps.pivot(index="POS", columns="SAMPLE", values="TOTAL")
    assert np.array_equal(check_pivot, snp_counts), "SNP file reading failed"
    assert not np.any(check_pivot.isna()), "SNP file reading failed"
    assert np.array_equal(all_names, list(snp_counts.columns))   # make sure that sample order is the same
    return np.array(snp_counts.index), np.array(snp_counts), snps

"""
load all unique snp positions from baf_file for all chromosomes
return a 2d SNP arrays, arr[i] represents snp positions for chromosomes[i]
chromosomes are pre-sorted.
baf and chromosomes should use same format for chr notation.
"""
def load_snps_positions(baf_file: str, chromosomes: list):
    baf_df = pd.read_csv(baf_file, sep="\t", header=None,
                             usecols=range(2),
                             names=["CHR", "POS"])
    baf_groups = baf_df.groupby("CHR")
    ret = {}
    for ch in chromosomes:
        if ch.endswith("X") or ch.endswith("Y"):
            log(msg=f"Warning, found SNP for sex chromosomes {ch}, ignored")
            ret.append(None) # append an empty array here
            continue
        baf_ch = baf_groups.get_group(ch)
        baf_ch = baf_ch.drop_duplicates(ignore_index=True)
        baf_ch = baf_ch.sort_values(by=["POS"], ignore_index=True)
        ret[ch] = baf_ch["POS"].to_numpy(dtype=np.uint32)
    return ret

"""
convert segments to thresholds in BED format (segment per row)
snp_positions: 1D array, 1-based and pre-sorted
seg_df_df: 0-based and left-close&right-open, possibly unsorted.
return nx2 thresholds
"""
def segments2thresholds(ch: str, snp_positions: np.ndarray, seg_df_ch: pd.DataFrame, consider_snp=True):
    seg_df_ch = seg_df_ch.sort_values(by="START", ignore_index=True)
    segments = seg_df_ch[["START", "END"]].to_numpy(dtype=np.uint32)
    snp_positions_1 = snp_positions - 1 # translate to 0-based index
    thresholds = None
    init_thres = False
    for [sstart, sstop] in segments:
        if not consider_snp:
            sub_segments = np.array([[sstart, sstop]])
        else:
            # find all snp positions within boundery
            left_idx = np.argmax(snp_positions_1 >= sstart)
            if snp_positions_1[left_idx] < sstart or snp_positions_1[left_idx] >= sstop:
                log(msg=f"warning! no SNP found in pre-defined segment: {ch}:{sstart}-{sstop}\n", level="STEP")
                # argmax -> 0, no SNP found with position >= sstart.
                continue
            right_idx = np.argmax(snp_positions_1 >= sstop)
            if snp_positions_1[right_idx] < sstop:
                # argmax returns 0, no SNP found after sstop, bound left only
                bounded_snp_positions = snp_positions_1[left_idx:]
            else:
                bounded_snp_positions = snp_positions_1[left_idx:right_idx]
            
            if len(bounded_snp_positions) < 2: # only one SNP found, use segment boundary
                sub_segments = np.array([[sstart, sstop]])
            else:
                # for every adjacent SNP position, record the midpoint as the interval splitting position.
                snp_thresholds = np.ceil(np.vstack([bounded_snp_positions[:-1], 
                                                    bounded_snp_positions[1:]]).mean(axis=0)).astype(np.uint32)
                if snp_thresholds[0] != sstart:
                    snp_thresholds = np.concatenate([[sstart], snp_thresholds])

                if snp_thresholds[-1] != sstop:
                    snp_thresholds = np.concatenate([snp_thresholds, [sstop]])
                sub_segments = np.column_stack((snp_thresholds[:-1], snp_thresholds[1:]))

        if not init_thres:
            thresholds = sub_segments
            init_thres = True
        else:
            thresholds = np.concatenate([thresholds, sub_segments], axis=0)
    return thresholds, init_thres

def store_adp_binning(starts: list, ends: list, snpsv_ch: pd.DataFrame, 
                           ch: str, outdir: str, prefix: str):
    if len(starts) == 0 or snpsv_ch.empty:
        log(f"starts empty for {ch} {prefix}", level="INFO")
        print(snpsv_ch)
        return

    log(f"Save temp results for adaptive binning in {outdir}\n", level="INFO")
    os.makedirs(outdir, exist_ok=True)
    if "REFC" not in snpsv_ch.columns:
        assert "REF" in snpsv_ch.columns and "ALT" in snpsv_ch.columns
        snpsv_ch = snpsv_ch.rename(columns={"REF": "REFC", "ALT": "ALTC"})
    df = snpsv_ch[["SAMPLE", "CHR", "POS", "TOTAL", "REFC", "ALTC"]]
    # nbins = len(starts)
    big_column_names = {
        "SAMPLE": "str",
        "CHR": "str",
        "POS": "int",
        "TOTAL": "int",
        "REFC": "int",
        "ALTC": "int",
        "BIN_ID": "int",
        "BIN_START": "int",
        "BIN_END": "int"
    }
    big_df = pd.DataFrame({c: pd.Series(dtype=t) for c, t in big_column_names.items()})
    for i in range(len(starts)):
        snp_df: pd.DataFrame = df[(df.POS >= starts[i]) & (df.POS <= ends[i])].copy()
        snp_df["BIN_ID"] = i
        snp_df["BIN_START"] = starts[i]
        snp_df["BIN_END"] = ends[i]
        snp_df.to_csv(f"{outdir}/TEMP_{prefix}_{ch}_BIN_{i}.tsv", sep="\t", header=True, index=False)
        big_df = pd.concat([big_df, snp_df], ignore_index=True)
    big_df.to_csv(f"{outdir}/TEMP_{prefix}_{ch}_ALL.tsv", sep="\t", header=True, index=False)
    log(f"Number of bins: {len(starts)}\n", level="INFO")
    return