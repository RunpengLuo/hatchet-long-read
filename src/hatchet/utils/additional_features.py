import pandas as pd
import numpy as np

import os
import gzip

from hatchet.utils.Supporting import (
    ensure,
    log,
    error,
    which,
    url_exists,
    bcolors,
    numericOrder,
)
import hatchet.utils.Supporting as sp

def sort_chroms(chromosomes: list):
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
    using_chr = [a.startswith('chr') for a in chromosomes]
    if any(using_chr):
        ensure(
            all(using_chr),
            'Some chromosomes use chr notation while others do not.',
        )
        return True
    return False

def has_header(file: str, header_prefix=["CHR", "Chr", "CHROMOSOME", "#ID"]):
    fd = gzip.open(file, 'r') if file.endswith(".gz") else open(file, 'r')
    first_line = fd.readline()
    fd.close()
    for hp in header_prefix:
        if first_line.startswith(hp):
            return True
    return False

# load segment file in BED format with 1-indexed and left-close right-open format
# CHR\tSTART\tEND\t...
def load_seg_file(seg_file: str, use_chr_bam: bool):
    if has_header(seg_file):
        seg_df = pd.read_csv(seg_file, sep='\t')
    else:
        seg_df = pd.read_csv(seg_file, sep='\t', header=None,
                             usecols=range(3),
                             names=["CHR", "START", "END"])

    seg_df["CHR"] = seg_df["CHR"].astype("str")
    use_chr_seg = use_chr_prefix(seg_df["CHR"].tolist())
    if use_chr_bam and not use_chr_seg:
        seg_df["CHR"] = seg_df["CHR"].apply(lambda s: "chr" + s)
    if not use_chr_bam and use_chr_seg:
        seg_df["CHR"] = seg_df["CHR"].apply(lambda s: s[3:])

    chs = seg_df["CHR"].unique().tolist()

    # threshold_grps = seg_df.groupby(by="CHR", sort=False)
    return seg_df, chs

"""
Reads haplotype blocks from gtf file
"""
def load_gtf_file(gtf_file: str):
    return pd.read_csv(
        gtf_file, sep='\t', header=None, comment='#',
        names=[
            'CHR',
            'source',
            'feature',
            'START',
            'END',
            'score',
            'strand',
            'frame',
            'attribute',
        ],
    )

def load_mosdepth_files(sample_names: list, mosdepth_files: list):
    bed_mosdepths = []
    for sname, sbed_file in zip(sample_names, mosdepth_files):
        with gzip.open(sbed_file, 'rt') as f:
            bed_data = pd.read_csv(f, sep='\t', 
                                   names=['CHR', 'START', 'END', 'AVG_DEPTH'])
            bed_data['sample'] = sname
            bed_mosdepths.append((sname, bed_data))
    return bed_mosdepths

# bb/bulk.bb
def init_bb_dataframe():
    bb_column_names = [
        'CHR',
        'UNIT',
        'START',
        'END',
        'SAMPLE',
        'RD',
        'TOTAL_READS',
        'NORMAL_READS',
        'SNPS',
        'BCOUNT',
        'TOTAL_SNP_READS',
        'HAPLO',
        'SNP_POS',
        'SNP_REF_COUNTS',
        'SNP_ALT_COUNTS',
        'BAF',
        'BLOCK_START',
        'BLOCK_END',
    ]
    return pd.DataFrame(columns=bb_column_names)



def get_array_file_path(dirname: str, ch: str, use_prebuilt_segfile: bool):
    midfix = "segfile_" if not use_prebuilt_segfile else ""
    return [os.path.join(dirname, f"{ch}.{midfix}total.gz"), 
            os.path.join(dirname, f"{ch}.{midfix}threshold.gz")]

# return all non-exists files
def check_array_files(dirname: str, chromosomes: list, use_prebuilt_segfile: bool):
    expected = [os.path.join(dirname, 'samples.txt')]
    for ch in chromosomes:
        [ptotal, pthres] = get_array_file_path(dirname, ch, use_prebuilt_segfile)
        expected.extend([ptotal, pthres])
    return [a for a in expected if not os.path.isfile(a)]

def expected_count_files(dirname: str, chromosomes: list, sample_names: list, use_region: bool):
    expected = []
    sample_suffixes = [".mosdepth.global.dist.txt", 
                       ".mosdepth.summary.txt"]
    if use_region:
        sample_suffixes += [".regions.bed.gz",
                            ".regions.bed.gz.csi"]
    else:
        sample_suffixes += [".per-base.bed.gz", 
                            ".per-base.bed.gz.csi"]

    for name in sample_names:
        expected.extend([os.path.join(dirname, f"{name}.{ch}.starts.gz") for ch in chromosomes])
        expected.extend([os.path.join(dirname, f"{name}{sfx}") for sfx in sample_suffixes])
    return expected

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


# FIXME to be simplified
"""
# TODO move it to separate file since it is commonly used by two module
return:
1. SNP absolute positions
2. SNP counts in 2D matrix with dim #SNPs * #samples
3. SNP dataframe
"""
def read_snps(baf_file, ch, all_names, phasefile=None):
    """
    Read and validate SNP data for this patient (TSV table output from HATCHet deBAF.py).
    """
    all_names = [
        name for name in all_names if name != 'normal'
    ]   # remove normal sample -- not looking for SNP counts from normal

    # Read in HATCHet BAF table
    all_snps = pd.read_table(
        baf_file,
        names=['CHR', 'POS', 'SAMPLE', 'REF', 'ALT', 'REFC', 'ALTC'],
        dtype={
            'CHR': object,
            'POS': np.uint32,
            'SAMPLE': object,
            'ALT': np.uint32,
            'REF': np.uint32,
            'REFC': object,
            'ALTC': object,
        },
    )

    # Keep only SNPs on this chromosome
    snps = all_snps[all_snps.CHR == ch].sort_values(by=['POS', 'SAMPLE'])
    snps = snps.reset_index(drop=True)

    if len(snps) == 0:
        raise ValueError(
            sp.error(f'Chromosome {ch} not found in SNPs file (chromosomes in file: {all_snps.CHR.unique()})')
        )

    n_samples = len(all_names)
    ensure(n_samples == len(snps.SAMPLE.unique(), 
                            f'Expected {n_samples} samples, found {len(snps.SAMPLE.unique())} samples in SNPs file.'))
    ensure(set(all_names) != set(snps.SAMPLE.unique()), 
                f'Expected sample names did not match sample names in SNPs file.\n\
                Expected: {sorted(all_names)}\n  Found:{sorted(snps.SAMPLE.unique())}')


    # Add total counts column
    snpsv = snps.copy()
    snpsv['TOTAL'] = snpsv.ALT + snpsv.REF

    if phasefile is not None:
        # Read in phasing output
        phases = pd.read_table(
            phasefile,
            compression='gzip',
            comment='#',
            names='CHR\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tPHASE'.split(),
            usecols=['CHR', 'POS', 'PHASE'],
            quoting=3,
            low_memory=False,
            dtype={'CHR': object, 'POS': np.uint32},
        )
        # if the last column in phase file has other information,
        # take the GT part, which is the first elements split by colon
        # (e.g., GT:AD:DP:GQ:PL). This is needed for ONT data and
        # does not have any effect on Illumina shapeit phased data
        phases.PHASE = phases.PHASE.astype(str).apply(lambda x: x.split(":")[0])
        phases['FLIP'] = phases.PHASE.str.contains('1|0', regex=False).astype(np.int8)  # noqa: W605
        phases['NOFLIP'] = phases.PHASE.str.contains('0|1', regex=False).astype(np.int8)  # noqa: W605

        # Drop entries without phasing output
        phases = phases[phases.FLIP + phases.NOFLIP > 0]

        # For exact duplicate entries, drop one
        phases = phases.drop_duplicates()

        # For duplicate entries with the same (CHR, POS) but different phase, drop all
        phases = phases.drop_duplicates(subset=['CHR', 'POS'], keep=False)

        # Merge tables: keep only those SNPs for which we have phasing output
        snpsv = pd.merge(snpsv, phases, on=['CHR', 'POS'], how='left')

    # Create counts array and find SNPs that are not present in all samples
    snp_counts = snpsv.pivot(index='POS', columns='SAMPLE', values='TOTAL')
    missing_pos = snp_counts.isna().any(axis=1)

    # Remove SNPs that are absent in any sample
    snp_counts = snp_counts.dropna(axis=0)
    snpsv = snpsv[~snpsv.POS.isin(missing_pos[missing_pos].index)]

    # Pivot table for dataframe should match counts array and have no missing entries
    check_pivot = snpsv.pivot(index='POS', columns='SAMPLE', values='TOTAL')
    assert np.array_equal(check_pivot, snp_counts), 'SNP file reading failed'
    assert not np.any(check_pivot.isna()), 'SNP file reading failed'
    assert np.array_equal(all_names, list(snp_counts.columns))   # make sure that sample order is the same
    return np.array(snp_counts.index), np.array(snp_counts), snpsv

def adaptive_binning():

    pass