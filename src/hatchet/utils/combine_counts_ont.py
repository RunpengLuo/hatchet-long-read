import os
import sys
import subprocess
import gzip
import time
import numpy as np
import pandas as pd
from importlib.resources import path
import hatchet.data
from hatchet.utils.ArgParsing import (
    parse_combine_counts_args,
    parse_count_reads_args,
    parse_genotype_snps_arguments,
)
from hatchet.utils.Supporting import log, logArgs
from hatchet.utils.combine_counts import (
    adaptive_bins_arm,
    read_snps,
    read_total_and_thresholds,
)
from hatchet.utils.rd_gccorrect import rd_gccorrect

from hatchet.utils.additional_features import (
    load_gtf_file,
    load_gtf_file_bed,
    load_seg_file,
    load_mosdepth_files,
    init_bb_dataframe,
    get_array_file_path
)
from hatchet.utils.handle_segments import (
    intersect_segments
)


def main(args):
    log(msg='combine_counts_ont test version\n', level='STEP')
    log(msg='# Parsing and checking input arguments\n', level='STEP')
    args = parse_combine_counts_args(args)
    logArgs(args, 80)

    ts = time.process_time()

    baffile = args['baffile']
    threads = args['processes']
    chromosomes = args['chromosomes']
    outfile = args['outfile']
    all_names = args['sample_names']
    msr = args['min_snp_reads']
    mtr = args['min_total_reads']
    use_chr = args['use_chr']
    phase = args['phase']
    blocksize = args['blocksize']
    max_snps_per_block = args['max_snps_per_block']
    test_alpha = args['test_alpha']
    multisample = args['multisample']
    nonormalFlag = args['nonormalFlag']
    ponfile = args['ponfile']
    referencefasta = args['referencefasta']
    XX = args['XX']
    rdr_dir = args['array']

    segfile = args['segfile']
    refversion = args["refversion"]

    if refversion != None:
        with path(hatchet.data, f'{args["refversion"]}.segments.bed') as p:
            segfile = str(p)
        log(msg=f"use prebuilt reference file from {refversion}: {str(segfile)}\n", level='INFO')
    
    seg_df, _ = load_seg_file(segfile, use_chr)
    
    haplotype_file = args['gtf']
    if haplotype_file == None:
        log(msg="haplotype file is not provided, not tested, exit\n", level='ERROR')
        raise ValueError()
    
    mosdepth_files = args["mos_rg_files"]
    if mosdepth_files == None or len(mosdepth_files) == 0 or len(mosdepth_files) != len(all_names):
        log(msg="mosdepth region files is either not provided or not contain enough files, not tested, exit\n", level='ERROR')
        raise ValueError()
    
    # all mosdepth from same chromosome across all samples; n is sample name
    bed_mosdepths = load_mosdepth_files(all_names, mosdepth_files)
    haplotype_blocks = load_gtf_file_bed(haplotype_file)

    log(msg=f"Correct Haplotype blocks by segment file, #total hap blocks(raw)={len(haplotype_blocks)}\n", level='STEP')
    corr_hap_blocks = intersect_segments(seg_df, haplotype_blocks, raise_err=True)
    log(msg=f"#total hap blocks(filtered)={len(corr_hap_blocks)}\n", level='STEP')
    # This step make sure that any haplotype blocks outside segments region are shrinked, such as centromere regions

    big_bb = init_bb_dataframe()
    min_normal_reads = 1
    for ch in chromosomes:
        log(msg=f"adaptive binning {ch}\n", level='STEP')
        # TODO can be optimized
        snp_positions, snp_totals, snpsv = read_snps(baffile, ch, all_names, phasefile=phase)
        mosdepth_ch = [(n, mos[mos.CHR == ch]) for n, mos in bed_mosdepths]

        # load total and threshold file from count-reads, one SNP per segment
        tot_file, thres_file = get_array_file_path(rdr_dir, ch)
        tot_arr_ch = np.loadtxt(tot_file, dtype=np.uint32)

        # filter by normal read depth
        # tot_keep_idx = np.where(tot_arr_ch[:, 1] >= min_normal_reads)
        # tot_arr_ch = tot_arr_ch[tot_keep_idx]

        thres_df_ch, _ = load_seg_file(thres_file, use_chr)
        log(msg=f"{ch}\t#thresholds (raw)={len(thres_df_ch)}\n", level="STEP")
        # thres_df_ch = thres_df_ch.iloc[tot_keep_idx]
        # log(msg=f"{ch}\t#thresholds (filtered)={len(thres_df_ch)}\n", level="STEP")
        thres_arr_ch = thres_df_ch[["START", "END"]].to_numpy(dtype=np.uint32)

        hap_blocks_ch = corr_hap_blocks[corr_hap_blocks.CHR == ch]
        log(msg=f"{ch}\t#hap blocks (raw)={len(hap_blocks_ch)}\n", level="STEP")
        # hap_blocks_ch = intersect_segments(thres_df_ch, hap_blocks_ch, raise_err=True)
        # log(msg=f"{ch}\t#hap blocks (filtered)={len(hap_blocks_ch)}\n", level="STEP")

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
            
            (starts, ends, totals, rdrs) = adaptive_bins_segment_ont_ver2(
            # (starts, ends, totals, rdrs) = adaptive_bins_segment_ont(
                snp_thresholds=block_thres,
                total_counts=block_totals,
                snp_positions=block_snp_pos,
                snp_counts=block_snp_total,
                ch=ch,
                min_snp_reads=msr,
                min_total_reads=mtr,
                nonormalFlag=nonormalFlag,
                # mos_block=block_mos
            )
            if len(starts) == 0:
                continue
            
            block_bb = handle_hap_block_bins(ch, all_names, snpsv,
                                             hb_start, hb_stop,
                                             starts, ends, totals, 
                                             rdrs, nonormalFlag)
            big_bb = pd.concat([big_bb, block_bb], ignore_index=True)
        # end
    # end
    log(msg=f"finish adaptive binning\n", level='STEP')

    if nonormalFlag:
        for sample, df in big_bb.groupby("SAMPLE", sort=False):
            mead_rdr = max(df.RD.mean(), 1) # avoid div0
            big_bb.loc[big_bb.SAMPLE == sample, "RD"] = df["RD"] / mead_rdr
    else:
        rc = pd.read_table(args['totalcounts'], header=None, names=['SAMPLE', '#READS'])
        nreads_normal = rc[rc.SAMPLE == all_names[0]].iloc[0]['#READS']
        for sample in all_names[1:]:
            nreads_sample = rc[rc.SAMPLE == sample].iloc[0]['#READS']
            correction = nreads_normal / nreads_sample
            big_bb.loc[big_bb.SAMPLE == sample, "RD"] *= correction
    
    if args['gc_correct']:
        log(
            msg='# In long read sequencing pipeline, GC correction for RD is not recommended\n',
            level='WARN',
        )
        # log(
        #     msg='# Performing GC bias correction on read depth signal\n',
        #     level='STEP',
        # )
        # big_bb = rd_gccorrect(big_bb, referencefasta)
    
    # convert back to 1-indexed inclusive
    # TODO can make this consistent?
    big_bb.loc[:, "START"] = big_bb.START + 1
    big_bb.to_csv(outfile, index=False, sep='\t')
    log(msg=f'combine-counts-ont completed, processed time (exclude sp): {time.process_time()-ts}sec\n', level='STEP')
    return



"""
get b-allelic frequency count
"""
def get_b_count(df: pd.DataFrame):
    return df.apply(lambda row: row.REF if row.FLIP == 1 else row.ALT, axis=1).sum()

def handle_hap_block_bins(ch: str, all_names: list, snpsv: pd.DataFrame,
                          block_start: int, block_stop: int, 
                          starts: list, ends: list, totals: list, rdrs: list,
                          nonormalFlag=False):
    block_bb = init_bb_dataframe()
    for i in range(len(starts)): # per bin
        start, end = starts[i], ends[i]
        df = snpsv[(snpsv.POS >= start) & (snpsv.POS <= end)]
        df: pd.DataFrame = df.dropna(subset=["FLIP"])
        df_groups = df.groupby("SAMPLE")
        if len(df) == 0:
            continue
        normal_reads = 0 if nonormalFlag else totals[i][0]
        for s in range(0 if nonormalFlag else 1, len(all_names)):
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
2d snp_thresholds (segment version), greedy binning
assumption
1. left-most snp positions >= snp_thresholds[0, 0]
2. right-most snp positions < snp_thresholds[-1, 1]
3. snp_thresholds must be strictly adjacent, 
    i.e., snp_thresholds[i, 0] < snp_thresholds[i, 1] <= snp_thresholds[i+1, 0]
4. region [snp_thresholds[0, 0]...snp_thresholds[-1, 1] are all vaild binning region,
    centromere region is masked beforehand.
5. snp positions are distinct
6. snp_thresholds are non-overlap & sorted by starting positions
"""
def adaptive_bins_segment_ont(
    snp_thresholds: np.ndarray,
    total_counts: np.ndarray,
    snp_positions: np.ndarray,
    snp_counts: np.ndarray,
    ch: str,
    min_snp_reads=2000,
    min_total_reads=5000,
    nonormalFlag=False,
    use_averages_rd=False,
    mos_block=None,
):
    """
    Compute adaptive bins for a single haplotype block.
    Parameters: TBD
    """
    try:
        assert len(snp_thresholds) == len(total_counts), f"#tot_cts={len(total_counts)}\t#thres={len(snp_thresholds)}"
        assert len(snp_positions) == len(snp_counts), f"#snp_pos={len(snp_positions)}\t#snp_cts={len(snp_counts)}"
        assert ch[-1] not in ['X', 'Y'], "sex chromosome unsupported yet"
        # assert len(snp_positions) == len(snp_thresholds), f"#snp_pos={len(snp_positions)}\t#thres={len(snp_thresholds)}" 
    except AssertionError:
        log(msg=f"ERROR! {snp_thresholds[0]}\t{snp_thresholds[-1]}\t{snp_positions[0]}\t{snp_positions[-1]}\t{total_counts[0]}\t{total_counts[-1]}\n", level="ERROR")
        log(msg=f"{len(snp_thresholds)}\t{len(snp_positions)}\t{len(total_counts)}\n", level="ERROR")
        sys.exit(1)
    
    print(f"adaptive binning {ch}:{snp_thresholds[0]}-{snp_thresholds[-1]} #seg={len(snp_thresholds)} and #snps={len(snp_positions)}\n")

    n_samples = total_counts.shape[1] // 2
    n_thresholds  = len(snp_thresholds)

    # #starts for ith sample
    even_index = np.array([i * 2 for i in range(n_samples)], dtype=np.int8)
    # mean read depth for ith sample
    odd_index = even_index + 1

    starts = []
    ends = []
    rdrs = []
    totals = []
    bss = []
    bin_sep_idx = []

    # per-threshold bin counts
    bin_total = np.zeros(n_samples, dtype=np.uint32)
    bin_snp_size = n_samples if nonormalFlag else n_samples - 1
    bin_snp = np.zeros(bin_snp_size, dtype=np.uint32) 
    
    # one snp per threshold segment
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

        if use_averages_rd: # long read case
            bin_total += total_counts[i, odd_index]
            total_cond = np.all(bin_total >= min_total_reads)
        else: # short reads case
            bin_total += total_counts[i, even_index]
            if i + 1 < n_thresholds:
                total_cond = np.all((bin_total - total_counts[i + 1, odd_index]) >= min_total_reads)
            else:
                total_cond = np.all(bin_total >= min_total_reads)
        
        merge_last_bin = False
        if np.any(bin_snp < min_snp_reads) or total_cond == False:
            if i + 1 < n_thresholds: # hope next one
                continue
            # last bin now, merge if there is one previous bin, create new bin otherwise
            merge_last_bin = len(starts) != 0

        if not use_averages_rd and i + 1 < n_thresholds:
            print("next threshold: ", total_counts[i + 1])
            assert any(bin_total > total_counts[i + 1, odd_index])
            bin_total -= total_counts[i + 1, odd_index]

        if merge_last_bin:
            ends[-1] = end
            prev_sep_idx = bin_sep_idx[-1]
            if not use_averages_rd:
                totals[-1] += total_counts[prev_sep_idx, odd_index] # add back
            totals[-1] += bin_total
            bss[-1] += bin_snp
        else:
            starts.append(start)
            ends.append(end)
            bss.append(bin_snp)
            totals.append(bin_total)

        rdrs_bin = []
        if mos_block != None:
            # starts[-1] and ends[-1] already update, no merge last bin to check
            mos_intersect = [(n, mos[(mos.START > starts[-1] - 1000) & (mos.END < ends[-1] + 1000)]) 
                for n, mos in mos_block]
            if nonormalFlag:
                rdrs_bin = [np.mean(_mos_int['AVG_DEPTH']) for (_, _mos_int) in mos_intersect]
            else:
                norm = np.mean(mos_intersect[0][1]['AVG_DEPTH'])
                rdrs_bin = [np.mean(_mos_int['AVG_DEPTH']) / norm for (_, _mos_int) in mos_intersect[1:]]
        else:
            if nonormalFlag:
                rdrs_bin = totals[0:] / totals[0:]
            else:
                if merge_last_bin:
                    rdrs_bin = (totals[-2][1:] + totals[-1][1:]) / (totals[-2][0] + totals[-1][0])
                else:
                    rdrs_bin = totals[-1][1:] / totals[-1][0]
        rdrs_bin = np.array(rdrs_bin, dtype=np.float32)
        if merge_last_bin: # replace the previous bin
            rdrs[-1] = rdrs_bin
        else:
            rdrs.append(rdrs_bin)
        
        # init next round
        start = None
        end = None
        bin_total = np.zeros(n_samples, dtype=np.uint32)
        bin_snp = np.zeros(bin_snp_size, dtype=np.uint32) 
        bin_sep_idx.append(i) # add bin separator
    
    log(msg=f"---hblock {snp_thresholds[0]}-{snp_thresholds[-1]} has {len(starts)} bins\n", level="STEP")
    # TODO bss may also be useful?
    return starts, ends, totals, rdrs
    

"""
Given a group of consecutive bins, merge bins from left to right

"""
def adaptive_bins_segment_ont_ver2(
    snp_thresholds: np.ndarray,
    total_counts: np.ndarray,
    snp_positions: np.ndarray,
    snp_counts: np.ndarray,
    ch: str,
    min_snp_reads=2000,
    min_total_reads=5000,
    nonormalFlag=False
):
    """
    Compute adaptive bins for a single haplotype block.
    Parameters: TBD
    """
    try:
        assert len(snp_thresholds) == len(total_counts), f"#tot_cts={len(total_counts)}\t#thres={len(snp_thresholds)}"
        assert len(snp_positions) == len(snp_counts), f"#snp_pos={len(snp_positions)}\t#snp_cts={len(snp_counts)}"
        assert ch[-1] not in ['X', 'Y'], "sex chromosome unsupported yet"
        # assert len(snp_positions) == len(snp_thresholds), f"#snp_pos={len(snp_positions)}\t#thres={len(snp_thresholds)}" 
    except AssertionError:
        log(msg=f"ERROR! {snp_thresholds[0]}\t{snp_thresholds[-1]}\t{snp_positions[0]}\t{snp_positions[-1]}\t{total_counts[0]}\t{total_counts[-1]}\n", level="ERROR")
        log(msg=f"{len(snp_thresholds)}\t{len(snp_positions)}\t{len(total_counts)}\n", level="ERROR")
        sys.exit(1)

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
    bin_snp_size = n_samples if nonormalFlag else n_samples - 1
    bin_snp = np.zeros(bin_snp_size, dtype=np.uint32) 
    
    # FIXME one snp per threshold segment
    start = None
    end = None
    j = 0 # snp counter
    n_snps = len(snp_positions)
    for i in range(n_thresholds):
        if start == None:
            start = snp_thresholds[i, 0]
        cstart = snp_thresholds[i, 0]
        end = snp_thresholds[i, 1]
        bs = end - snp_thresholds[i, 0]

        while j != n_snps and snp_positions[j] < end:
            if snp_positions[j] >= cstart: # ensure inclusive SNP
                bin_snp += snp_counts[j]
            j += 1

        bin_total += total_counts[i, odd_index] * bs # sum up total aligned bases
        merged_depth = bin_total / (end - start)
        total_cond = np.all(merged_depth >= min_total_reads)
        
        merge_last_bin = False
        if np.any(bin_snp < min_snp_reads) or total_cond == False:
            if i + 1 < n_thresholds: # hope next one
                continue
            # last bin now, merge if there is one previous bin, create new bin otherwise
            merge_last_bin = len(starts) != 0
        if merge_last_bin:
            totals[-1] = (totals[-1] * (ends[-1] - starts[-1]) + bin_total) // (end - starts[-1])
            ends[-1] = end
            bss[-1] += bin_snp
        else:
            totals.append(bin_total // (end - start))
            starts.append(start)
            ends.append(end)
            bss.append(bin_snp)

        if nonormalFlag:
            rdrs_bin = np.array(totals[-1] / totals[-1], dtype=np.float32)
        else:
            if totals[-1][0] == 0:
                log(msg=f"WARN! zero normal reads found in bin {ch}:{starts[-1]}-{ends[-1]} in block {snp_thresholds[0, 0]}:{snp_thresholds[-1, 1]}\n",
                    level="STEP")
                rdrs_bin = np.zeros(n_samples, dtype=np.float32)
            else:
                rdrs_bin = np.array(totals[-1] / max(totals[-1][0], 0), dtype=np.float32)

        if merge_last_bin: # replace the previous bin
            rdrs[-1] = rdrs_bin
        else:
            rdrs.append(rdrs_bin)
        
        # init next round
        start = None
        end = None
        bin_total = np.zeros(n_samples, dtype=np.uint32)
        bin_snp = np.zeros(bin_snp_size, dtype=np.uint32) 
    
    # log(msg=f"---hblock {snp_thresholds[0]}-{snp_thresholds[-1]} has {len(starts)} bins\n", level="STEP")
    # TODO bss may also be useful?
    return starts, ends, totals, rdrs

if __name__ == '__main__':
    main()