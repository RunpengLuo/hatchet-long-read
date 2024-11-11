import os
import sys
import subprocess
import gzip
import time
import numpy as np
import pandas as pd
from importlib.resources import path
import hatchet.data
from hatchet.utils.ArgParsing import parse_combine_counts_args
from hatchet.utils.Supporting import log, logArgs
from hatchet.utils.rd_gccorrect import rd_gccorrect

from hatchet.utils.additional_features import (
    load_gtf_file_bed,
    load_seg_file,
    load_mosdepth_files,
    init_bb_dataframe,
    init_summary_dataframe,
    get_array_file_path,
    read_snps
)
from hatchet.utils.handle_segments import (
    intersect_segments
)


def main(args):
    log(msg="combine_counts_lr test version\n", level="STEP")
    log(msg="# Parsing and checking input arguments\n", level="STEP")
    args = parse_combine_counts_args(args)
    logArgs(args, 80)

    ts = time.process_time()

    baffile = args["baffile"] # tumor.1bed
    chromosomes = args["chromosomes"]
    outfile = args["outfile"]
    all_names = args["sample_names"]
    msr = args["min_snp_reads"]
    mtr = args["min_total_reads"]
    use_chr = args["use_chr"]
    phase = args["phase"]
    threads = args["processes"]
    blocksize = args["blocksize"]
    max_snps_per_block = args["max_snps_per_block"]
    test_alpha = args["test_alpha"]
    multisample = args["multisample"]
    no_normal = args["no_normal"] # set based on samples.txt first name
    ponfile = args["ponfile"]
    referencefasta = args["referencefasta"]
    XX = args["XX"]
    rdr_dir = args["array"]

    segfile = args["seg_file"]
    refversion = args["ref_version"]

    outdir = outfile[:str.rindex(outfile, "/")]

    if refversion != None:
        with path(hatchet.data, f"{refversion}.segments.bed") as p:
            segfile = str(p)
        log(msg=f"use prebuilt reference file from {refversion}: {str(segfile)}\n", level="INFO")
    
    seg_df, _ = load_seg_file(segfile, use_chr)
    
    haplotype_file = args["gtf_file"]
    if haplotype_file == None:
        log(msg="haplotype file is not provided, not tested, exit\n", level="ERROR")
        raise ValueError()
    
    mosdepth_files = args["mos_rg_files"]
    if mosdepth_files == None or len(mosdepth_files) == 0 or len(mosdepth_files) != len(all_names):
        log(msg="mosdepth region files is either not provided or not contain enough files, not tested, exit\n", level="ERROR")
        raise ValueError()
    
    # all mosdepth from same chromosome across all samples; n is sample name
    bed_mosdepths = load_mosdepth_files(all_names, mosdepth_files)
    haplotype_blocks = load_gtf_file_bed(haplotype_file)

    log(msg=f"Correct Haplotype blocks by segment file, #total hap blocks(raw)={len(haplotype_blocks)}\n", level="STEP")
    corr_hap_blocks = intersect_segments(seg_df, haplotype_blocks, raise_err=True)
    log(msg=f"#total hap blocks(filtered)={len(corr_hap_blocks)}\n", level="STEP")
    # This step make sure that any haplotype blocks outside segments region are shrinked, such as centromere regions

    summary_df = init_summary_dataframe()
    big_bb = init_bb_dataframe()
    for ch in chromosomes:
        log(msg=f"adaptive binning {ch}\n", level="STEP")
        # TODO can be optimized
        snp_positions, snp_totals, snpsv = read_snps(baffile, ch, all_names, phasefile=phase)
        snp_positions = snp_positions - 1 # translate to 0-based index
        mosdepth_ch = [(n, mos[mos.CHR == ch]) for n, mos in bed_mosdepths]

        # load total and threshold file from count-reads, one SNP per segment
        tot_file, thres_file = get_array_file_path(rdr_dir, ch)
        tot_arr_ch = np.loadtxt(tot_file, dtype=np.uint32)

        thres_df_ch, _ = load_seg_file(thres_file, use_chr)
        log(msg=f"{ch}\t#thresholds (raw)={len(thres_df_ch)}\n", level="STEP")
        thres_arr_ch = thres_df_ch[["START", "END"]].to_numpy(dtype=np.uint32)

        hap_blocks_ch = corr_hap_blocks[corr_hap_blocks.CHR == ch]
        log(msg=f"{ch}\t#hap blocks (raw)={len(hap_blocks_ch)}\n", level="STEP")

        # TODO parallel this step
        for _, row in hap_blocks_ch.iterrows():
            hb_start, hb_stop = row.START, row.END

            # compute left-most and right-most threshold segment 
            # overlaps with current haplotype block
            tidx1 = np.argmax(thres_arr_ch[:, 1] >= hb_start)
            thres1 = thres_arr_ch[tidx1]
            if thres1[1] < hb_start:
                # no thres segment left-overlap with current block
                continue
            tidx2 = np.argmax(thres_arr_ch[:, 1] >= hb_stop)
            thres2 = thres_arr_ch[tidx2]
            if thres2[0] >= hb_stop:
                # no thres segment right-overlap with current block
                continue
            if tidx1 > tidx2:
                continue

            # effect_start = max(thres1[0], hb_start)
            # effect_stop = min(thres2[1], hb_stop)

            block_snps_idx = np.where((snp_positions >= thres1[0]) & (snp_positions < thres2[1]))[0]
            if len(block_snps_idx) == 0:
                # no SNPs in current block
                continue
            # extract snp snp_positions within the haplotype block region
            block_snp_pos = snp_positions[block_snps_idx]
            block_snp_total = snp_totals[block_snps_idx]


            block_thres  = thres_arr_ch[tidx1:tidx2 + 1, ] # n-by-2
            block_totals = tot_arr_ch[tidx1:tidx2 + 1, ]  # n-by-4

            block_mos = [(n, mos[(mos.START > hb_start - 1000) & (mos.END < hb_stop + 1000)]) 
                for n, mos in mosdepth_ch]
            
            (starts, ends, totals, bss, rdrs) = adaptive_bins_segment_lr(
                snp_thresholds=block_thres,
                total_counts=block_totals,
                snp_positions=block_snp_pos,
                snp_counts=block_snp_total,
                ch=ch,
                min_snp_reads=msr,
                min_total_reads=mtr,
                no_normal=no_normal,
                mos_block=block_mos
            )

            summary_df.loc[len(summary_df)] = [
                ch,
                hb_start,
                hb_stop,
                hb_stop - hb_start,
                block_thres[0, 0],
                block_thres[-1,1],
                len(block_thres),
                len(block_snp_pos),
                len(starts)
            ]

            if len(starts) == 0:
                continue
            
            block_bb = handle_hap_block_bins(ch, all_names, snpsv,
                                             hb_start, hb_stop,
                                             starts, ends, totals, 
                                             rdrs, no_normal)
            big_bb = pd.concat([big_bb, block_bb], ignore_index=True)
        # end
    # end
    log(msg=f"finish adaptive binning\n", level="STEP")
    
    # check unscaled version
    big_bb.to_csv(f"{outdir}/bulk.raw.bb", index=False, sep="\t")

    # correct RD by total normal / sample counts
    if no_normal:
        for sample, df in big_bb.groupby("SAMPLE", sort=False):
            mead_rdr = df.RD.mean()
            mean_rdr = mean_rdr if mean_rdr > 0 else 1 # avoid div0
            log(msg=f"no-normal, sample:{sample} mean RDR factor={mean_rdr}\n", level="STEP")
            big_bb.loc[big_bb.SAMPLE == sample, "RD"] = df["RD"] / mead_rdr
    else:
        rc = pd.read_table(args["totalcounts"], header=None, names=["SAMPLE", "#READS"])
        nreads_normal = rc[rc.SAMPLE == all_names[0]].iloc[0]["#READS"]
        for sample in all_names[1:]:
            nreads_sample = rc[rc.SAMPLE == sample].iloc[0]["#READS"]
            correction = nreads_normal / nreads_sample
            big_bb.loc[big_bb.SAMPLE == sample, "RD"] *= correction
    
    if args["gc_correct"]:
        log(
            msg="# In long read sequencing pipeline, GC correction for RD is not recommended\n",
            level="WARN",
        )
        # log(
        #     msg="# Performing GC bias correction on read depth signal\n",
        #     level="STEP",
        # )
        # big_bb = rd_gccorrect(big_bb, referencefasta)
    
    # convert back to 1-indexed inclusive
    # TODO can make this consistent?
    big_bb.loc[:, "START"] = big_bb.START + 1
    big_bb.to_csv(outfile, index=False, sep="\t")
    summary_df.to_csv(f"{outdir}/summary.tsv", index=False, header=True, sep="\t")
    log(msg=f"combine-counts-ont completed, processed time (exclude sp): {time.process_time()-ts}sec\n", level="STEP")
    return



"""
get b-allelic frequency count
"""
def get_b_count(df: pd.DataFrame):
    return df.apply(lambda row: row.REF if row.FLIP == 1 else row.ALT, axis=1).sum()

def handle_hap_block_bins(ch: str, all_names: list, snpsv: pd.DataFrame,
                          block_start: int, block_stop: int, 
                          starts: list, ends: list, totals: list, rdrs: list,
                          no_normal=False):
    block_bb = init_bb_dataframe()
    for i in range(len(starts)): # per bin
        start, end = starts[i], ends[i]
        df = snpsv[(snpsv.POS >= start) & (snpsv.POS <= end)]
        df: pd.DataFrame = df.dropna(subset=["FLIP"])
        df_groups = df.groupby("SAMPLE")
        if len(df) == 0:
            continue
        normal_reads = 0 if no_normal else totals[i][0]
        for s in range(0 if no_normal else 1, len(all_names)):
            sample = all_names[s]
            df_sample = df_groups.get_group(sample)
            num_snps = len(df_sample)
            total_reads = totals[i][s]
            rdr = rdrs[i][s]
            if num_snps <= 0 or total_reads <= 0 or rdr <= 0:
                continue

            total_snp_reads = df_sample.TOTAL.sum()
            assert total_snp_reads > 0, f"ERROR! total_snp_reads is 0 for {sample}:{start}-{end}"
            b_count = get_b_count(df_sample)
            block_bb.loc[len(block_bb)] = [
                ch, "unit", start, end, sample,
                rdr, total_reads, normal_reads,
                num_snps, b_count, total_snp_reads,
                "", "", "", "", b_count / total_snp_reads,
                block_start, block_stop
            ]

    for s in block_bb.SAMPLE.unique():
        med_baf = block_bb[block_bb.SAMPLE==s].BAF.median()
        if med_baf > 0.5:
            baf = block_bb.loc[block_bb.SAMPLE==s, "BAF"]
            block_bb.loc[block_bb.SAMPLE==s, "BAF"] = 1 - baf
    return block_bb

"""
Given a group of consecutive bins, merge bins from left to right

"""
def adaptive_bins_segment_lr(
    snp_thresholds: np.ndarray,
    total_counts: np.ndarray,
    snp_positions: np.ndarray,
    snp_counts: np.ndarray,
    ch: str,
    min_snp_reads=2000,
    min_total_reads=5000,
    no_normal=False,
    mos_block=None
):
    """
    Compute adaptive bins for a single haplotype block.
    Parameters: TBD
    """
    n_samples = total_counts.shape[1] // 2
    n_thresholds  = len(snp_thresholds)

    # mean read depth for ith sample, ignore starts count!
    odd_index = np.array([i * 2 + 1 for i in range(n_samples)], dtype=np.int8)

    starts = []
    ends = []
    rdrs = []
    totals = []
    bss = []

    # per-threshold total aligned bases
    bin_total = np.zeros(n_samples, dtype=np.uint32)
    bin_snp_size = n_samples if no_normal else n_samples - 1
    bin_snp = np.zeros(bin_snp_size, dtype=np.uint32) 

    start = None
    end = None
    j = 0 # snp counter
    n_snps = len(snp_positions)
    for i in range(n_thresholds):
        if start == None:
            start = snp_thresholds[i, 0]
        cstart = snp_thresholds[i, 0]
        end = snp_thresholds[i, 1]

        while j != n_snps and snp_positions[j] < end:
            if snp_positions[j] >= cstart: # ensure inclusive SNP
                bin_snp += snp_counts[j]
            j += 1

        merge_last_bin = False
        if not no_normal and total_counts[i, 1] == 0:
            log(msg=f"WARN! normal depth = 0 for current bin {ch}:{cstart}-{end} in block\n", level="STEP")
            if all(bin_total == 0) or i <= 0: # no previous bins
                log(msg=f"no previous bin, skip\n", level="STEP")
                start = None
                continue
            end = snp_thresholds[i - 1, 1] # previous bin endpoint
            log(msg=f"merge previous bin:{start}-{end} (ignore condition)\n", level="STEP")
            last_total = bin_total // (end - start)
        else:
            bin_total += total_counts[i, odd_index] * (end - snp_thresholds[i, 0]) # sum up total aligned bases
            merged_depth = bin_total / (end - start)

            if np.any(bin_snp < min_snp_reads) or np.any(merged_depth < min_total_reads):
                if i + 1 < n_thresholds: # hope next one
                    continue 
                merge_last_bin = len(starts) != 0

            if merge_last_bin:
                last_total = (totals[-1] * (ends[-1] - starts[-1]) + bin_total) // (end - starts[-1])
                start = starts[-1]
            else:
                last_total = bin_total // (end - start)
        
        # start and end has updated bin information
        rdrs_bin = []
        if mos_block != None:
            mos_intersect = [(n, mos[(mos.START > start - 1000) & (mos.END < end + 1000)]) 
                for n, mos in mos_block]
            if no_normal:
                rdrs_bin = [np.mean(_mos_int["AVG_DEPTH"]) for (_, _mos_int) in mos_intersect]
            else:
                norm = np.mean(mos_intersect[0][1]["AVG_DEPTH"])
                rdrs_bin = [np.mean(_mos_int["AVG_DEPTH"]) / norm for (_, _mos_int) in mos_intersect]
        else:
            if no_normal:
                rdrs_bin = np.array(last_total / last_total, dtype=np.float32)
            else:
                rdrs_bin = np.array(last_total / last_total[0], dtype=np.float32)

        if merge_last_bin: # replace the previous bin
            totals[-1] = last_total
            ends[-1] = end
            bss[-1] += bin_snp
            rdrs[-1] = rdrs_bin
        else:
            totals.append(last_total)
            starts.append(start)
            ends.append(end)
            bss.append(bin_snp)
            rdrs.append(rdrs_bin)
        
        # init next round
        start = None
        end = None
        bin_total = np.zeros(n_samples, dtype=np.uint32)
        bin_snp = np.zeros(bin_snp_size, dtype=np.uint32) 
    
    # log(msg=f"---hblock {snp_thresholds[0]}-{snp_thresholds[-1]} has {len(starts)} bins\n", level="STEP")
    # TODO bss may also be useful?
    return starts, ends, totals, bss, rdrs

if __name__ == "__main__":
    main()