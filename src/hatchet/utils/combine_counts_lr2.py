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
    load_seg_file,
    load_mosdepth_files,
    init_bb_dataframe_v2,
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
    referencefasta = args["referencefasta"]
    XX = args["XX"]
    rdr_dir = args["array"]

    ponfile = args["pon_file"]
    segfile = args["seg_file"]
    refversion = args["ref_version"]

    DEBUG = True

    outdir = outfile[:str.rindex(outfile, "/")]
    if DEBUG:
        os.makedirs(f"{outdir}/snp_sv", exist_ok=True) # DEBUG

    if os.path.isfile(outfile):
        log(msg=f"output={outfile} exists, skip combine-counts-lr\n", level="STEP")
        return

    if refversion != None:
        with path(hatchet.data, f"{refversion}.segments.bed") as p:
            segfile = str(p)
        log(msg=f"use prebuilt reference file from {refversion}: {segfile}\n", level="INFO")
    else:
        log(msg=f"use external segment file: {segfile}\n", level="INFO")
    
    seg_df, _ = load_seg_file(segfile, use_chr)
    log(msg=f"Total number of segments={len(seg_df)}\n", level="STEP")

    n_samples = len(all_names)
    n_tumors = n_samples if no_normal else n_samples - 1

    odd_index = np.array([i * 2 + 1 for i in range(n_samples)], dtype=np.int8)
    even_index = np.array([i * 2 for i in range(n_samples)], dtype=np.int8)

    big_bb = init_bb_dataframe_v2()
    for ch in chromosomes:
        log(msg=f"adaptive binning {ch}\n", level="STEP")
        snp_positions, _, snp_sv = read_snps(baffile, ch, all_names, phasefile=phase)
        
        snp_positions = snp_positions - 1 # translate to 0-based index
        snp_unphased = snp_sv[snp_sv.SAMPLE == all_names[0 if no_normal else 1]].FLIP.isna().tolist()

        # load total and threshold file from count-reads, one SNP per segment
        tot_file, thres_file = get_array_file_path(rdr_dir, ch)
        tot_arr_ch = np.loadtxt(tot_file, dtype=np.uint32)
        thres_df_ch, _ = load_seg_file(thres_file, use_chr)
        thres_arr_ch = thres_df_ch[["START", "END"]].to_numpy(dtype=np.uint32)

        log(msg=f"{ch}\t#thresholds={len(thres_df_ch)}\n", level="STEP")
        assert len(tot_arr_ch) == len(thres_arr_ch)
        
        ch_bb = init_bb_dataframe_v2()
        for _, seg_row in seg_df[seg_df["CHR"] == ch].iterrows():
            sstart, sstop = seg_row.START, seg_row.END    
            hblock_states = []
            # partition thres_seg by phased/unphased blocks
            is_unphased = None
            sidx = 0
            tidx = 0
            snp_idx = None
            snp_sidx = 0

            for tidx, [tstart, tstop] in enumerate(thres_arr_ch):
                if not (tstart >= sstart and tstop <= sstop):
                    continue

                block_snps_idx = np.where((snp_positions >= tstart) & (snp_positions < tstop))[0]
                assert len(block_snps_idx) == 1, f"more than one snp found in a threshold block=({tstart},{tstop})"
                snp_idx = block_snps_idx[0]
                if is_unphased == None: # first time visit
                    is_unphased = snp_unphased[snp_idx]
                    sidx = tidx
                    snp_sidx = snp_idx
                    continue

                if is_unphased != snp_unphased[snp_idx]:
                    hblock_states.append((ch, sidx, tidx, snp_sidx, snp_idx, is_unphased))
                    is_unphased = snp_unphased[snp_idx] # switch state
                    sidx = tidx
                    snp_sidx = snp_idx

            if sidx < tidx: # add last block
                hblock_states.append((ch, sidx, tidx + 1, 
                                      snp_sidx, min(snp_idx + 1, len(snp_positions)), is_unphased))
            
            if len(hblock_states) == 0:
                log(msg=f"warning, no haplo-block found in {ch}:{sstart}-{sstop}\n", level="STEP")
                continue

            hblock_df = pd.DataFrame(hblock_states, columns=["#CHR", "START_TIDX", "STOP_TIDX", 
                                                             "START_SIDX", "STOP_SIDX", "UNPHASED"])
            if DEBUG:
                hblock_df.to_csv(f"{outdir}/debug_hblocks_{ch}_{sstart}_{sstop}.tsv", sep="\t", header=True)

            for _, row in hblock_df.iterrows():
                if row.UNPHASED:
                    continue

                # block_snp_total = snp_totals[row.START_SIDX:row.STOP_SIDX]
                block_snp_pos = snp_positions[row.START_SIDX:row.STOP_SIDX]

                block_thres = thres_arr_ch[row.START_TIDX:row.END_TIDX]
                block_totals = tot_arr_ch[row.START_TIDX:row.END_TIDX]

                # split phased block to multiple smaller bins
                num_thres = len(block_thres)
                bps = num_thres // msr
                for i in range(bps):
                    si = i * msr
                    st = si + msr
                    if i == bps - 1 and st < num_thres:
                        st = num_thres
                    
                    num_snps = st - si

                    bin_snp_pos = block_snp_pos[si:st]
                    bin_totals = block_totals[si:st]
                    bin_thres = block_thres[si:st]
                    bin_start = bin_thres[0][0]
                    bin_end = bin_thres[-1][1]

                    # compute mhBAF
                    bin_bafs_h1 = np.zeros(n_tumors, dtype=np.float64)
                    bin_bafs_h2 = np.zeros(n_tumors, dtype=np.float64)
                    bin_cov = np.zeros(n_tumors, dtype=np.uint32)
                    bin_alpha = np.zeros(n_tumors, dtype=np.uint32)
                    bin_beta = np.zeros(n_tumors, dtype=np.uint32)
                    for s in range(n_tumors):
                        sample_name = all_names[s if no_normal else s + 1]
                        bin_snps = snp_sv[(snp_sv.SAMPLE == sample_name) & (snp_sv.POS >= bin_snp_pos[0]) & (snp_sv.POS < bin_snp_pos[-1])]
                        phases = bin_snps.FLIP.astype(np.uint8).to_numpy()
                        assert len(bin_snps) == num_snps
                        alpha = np.sum(np.choose(phases, [bin_snps.REF, bin_snps.ALT]))
                        beta = np.sum(np.choose(phases, [bin_snps.ALT, bin_snps.REF]))
                        bin_alpha[s] = alpha
                        bin_beta[s] = beta
                        bin_bafs_h1[s] = beta / (alpha + beta)
                        bin_bafs_h2[s] = alpha / (alpha + beta)
                        bin_cov[s] = np.ceil((alpha + beta) / num_snps)

                    if np.mean(bin_bafs_h1) < np.mean(bin_bafs_h2):
                        bin_bafs = bin_bafs_h1
                    else:
                        bin_bafs = bin_bafs_h2
                        bin_tmp = bin_alpha
                        bin_alpha = bin_beta
                        bin_beta = bin_tmp

                    # compute normal, tumor reads
                    total_reads = np.sum(bin_totals[:, even_index], axis=0)
                    if no_normal:
                        normal_reads = 0
                    else:
                        normal_reads = total_reads[0]
                    
                    for s in range(n_tumors):
                        s2 = s if no_normal else s + 1
                        ch_bb.loc[len(ch_bb)] = [
                            ch, bin_start, bin_end, all_names[s2],
                            0.0, num_snps, bin_cov[s], 
                            bin_alpha[s], bin_beta[s], bin_bafs[s],
                            total_reads[s2], normal_reads, bin_bafs[s], 
                            0, 0.0, 0.0, 0.0
                        ]
            # 1) compute mhBAF & pick phasing per phased block, then split blocks (not necessary<0.5 single sample)
            # 2) or, split phased blocks, compute mhBAF for each of them (<0.5 single sample)
        # end
        big_bb = pd.concat([big_bb, ch_bb], ignore_index=True)
    # end
    log(msg=f"finish adaptive binning\n", level="STEP")

    # correct RD by total normal / sample counts
    if no_normal:
        for sample, df in big_bb.groupby("SAMPLE", sort=False):
            mean_rc = max(df.TOTAL_READS.mean(), 1)
            log(msg=f"no-normal, sample:{sample} mean read count={mean_rc}\n", level="STEP")
            big_bb.loc[big_bb.SAMPLE == sample, "RD"] = df["TOTAL_READS"] / mean_rc
    else:
        rc = pd.read_table(args["totalcounts"], header=None, names=["SAMPLE", "#READS"])
        nreads_normal = rc[rc.SAMPLE == all_names[0]].iloc[0]["#READS"]
        for sample, df in big_bb.groupby("SAMPLE", sort=False):
            nreads_sample = rc[rc.SAMPLE == sample].iloc[0]["#READS"]
            correction = nreads_normal / nreads_sample
            big_bb.loc[big_bb.SAMPLE == sample, "CORRECTED_READS"] = (
                df["TOTAL_READS"] * correction
            ).astype(np.int64)

            big_bb.loc[big_bb.SAMPLE == sample, "RD"] = (
                df["CORRECTED_READS"] / df["NORMAL_READS"]
            )
            big_bb.loc[big_bb.SAMPLE == sample, "UNCORR_RD"] = (
                df["TOTAL_READS"] / df["NORMAL_READS"]
            )
    
    if args["gc_correct"]:
        log(
            msg="# In long read sequencing pipeline, GC correction for RD is not recommended, skip\n",
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
    log(msg=f"combine-counts-lr completed, processed time (exclude sp): {time.process_time()-ts}sec\n", level="STEP")
    return

if __name__ == "__main__":
    main()