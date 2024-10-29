import sys
import os
from multiprocessing import Pool
import subprocess as sp
import numpy as np
import pandas as pd
import gzip
import traceback
from importlib.resources import path
import hatchet.data

from hatchet.utils.ArgParsing import parse_count_reads_args
from hatchet.utils.Supporting import log, logArgs, error
import hatchet.utils.TotalCounting as tc

from hatchet.utils.additional_features import (
    load_seg_file,
    sort_chroms,
    get_array_file_path,
    check_array_files,
    expected_starts_files,
    expected_count_files,
    check_count_files,
    workload_assignment,
    read_snps,
    load_snps_positions,
    segments2thresholds
)

from hatchet.utils.count_reads import count_chromosome_wrapper, get_chr_end

def main(args=None):
    log(msg='count_reads_ont test version\n', level='STEP')
    log(msg='# Parsing and checking input arguments\n', level='STEP')
    args = parse_count_reads_args(args)
    logArgs(args, 80)

    bams = args['bams']
    names = args['names']

    if args['nonormal']:
        bams, names = zip(*sorted(zip(*(bams, names)), key=lambda x: x[1]))
        bams = list(bams)
        names = list(names)
    else:
        tbams, tnames = zip(*sorted(zip(*(bams[1:], names[1:])), key=lambda x: x[1]))
        bams = [bams[0]] + list(tbams)
        names = [names[0]] + list(tnames)

    chromosomes = args['chromosomes']
    samtools = args['samtools']
    processes = args['j']
    outdir = args['outdir']
    mosdepth = args['mosdepth']
    tabix = args['tabix']
    readquality = args['readquality']
    use_chr = args['use_chr'] # bam file use_chr

    baffile = args['baf_file']
    segfile = args['seg_file']
    refversion = args["refversion"]

    if refversion != None: # segfile == None
        with path(hatchet.data, f'{args["refversion"]}.segments.bed') as p:
            segfile = str(p)
        log(msg=f"use prebuilt reference file from {refversion}: {str(segfile)}\n", level='STEP')
    
    seg_df, seg_chroms = load_seg_file(segfile, use_chr)

    # only chromosomes that present in both BAM file and segment file are handled
    chromosomes = list(set(chromosomes).intersection(seg_chroms))
    if len(chromosomes) == 0:
        raise ValueError(error("No chromosomes present in both BAM file and segment file / refversion"))
    chromosomes = sort_chroms(chromosomes)

    #
    # compute samtools starts.gz TODO can be optimized
    if any(not os.path.isfile(f) for f in expected_starts_files(outdir, chromosomes, names)):
        n_tasks_samtools = len(bams) * len(chromosomes)
        samtools_params = zip(
            np.repeat(chromosomes, len(bams)),
            [outdir] * n_tasks_samtools,
            [samtools] * n_tasks_samtools,
            bams * len(chromosomes),
            names * len(chromosomes),
            [readquality] * n_tasks_samtools,
        )
        n_workers_samtools, _ = workload_assignment(processes, n_tasks_samtools)
        log(msg=f"count_chromosome-num_worker={n_workers_samtools}\tnum_tasks={n_tasks_samtools}\n", level='STEP')
        try:
            with Pool(n_workers_samtools) as p:
                p.map(count_chromosome_wrapper, samtools_params)
        except Exception as e:
            log(msg=f"ERROR! count_chromosome raise exception: {e}\n",level='ERROR')
            p.terminate()
            raise ValueError()
        finally:
            p.join()
            log(msg="All count_chromosome finished\n", level='STEP')
    else:
        log(msg="found all count_chromosome intermediate files, skip\n", level='STEP')

    #
    # compute mosdepth with --by BED option
    snp_positions = load_snps_positions(baffile, chromosomes)
    # global segment file
    segment_file = os.path.join(outdir, "segments.bed")
    segment_file_gz = f"{segment_file}.gz"
    if not os.path.isfile(segment_file_gz) or \
        any(not os.path.isfile(os.path.join(outdir, f"{ch}.threshold.gz")) for ch in chromosomes):
        # TODO parallel if need?
        with open(segment_file, 'w') as rg_fd:
            # compute segment files per chromosome
            for i, ch in enumerate(chromosomes):
                log(msg=f"compute segment file for {ch}\n", level='STEP')
                if str.endswith(ch, 'X') or str.endswith(ch, 'Y'):
                    # TODO: do this procedure only for XY
                    log(
                        msg='Running on sex chromosome -- ignoring SNPs and min SNP reads\n',
                        level='STEP',
                    )
                    last_start = get_chr_end(outdir, names, ch)
                    snp_positions_ch = np.arange(5000, last_start, 5000)
                else:
                    snp_positions_ch = snp_positions[i]
                seg_df_ch = seg_df[seg_df["CHR"] == ch]
                thresholds_ch, init_thres = segments2thresholds(snp_positions_ch, seg_df_ch, consider_snp=True)
                if not init_thres:
                    raise ValueError(f"ERROR, {segfile} is invalid/empty for {ch}")
                if np.any(np.diff(thresholds_ch) < 0):
                    raise ValueError(f'improper negative interval in provided segment file for chromosome {ch}')
                np.savetxt(rg_fd, thresholds_ch, fmt=str(ch)+"\t%d\t%d")
                
                segment_file_ch = os.path.join(outdir, f"{ch}.threshold.gz")
                np.savetxt(segment_file_ch, thresholds_ch, fmt=str(ch)+"\t%d\t%d")
                log(msg=f"#segments for {ch}: {len(thresholds_ch)}\n", level='STEP')
            rg_fd.close()
        
        ret = sp.run(['gzip', '-6', segment_file])
        ret.check_returncode()
        log(msg="computed all segment files", level='STEP')
    else:
        log(msg="found all segments intermediate files, skip", level='STEP')
    segment_file = segment_file_gz

    # run mosdepth against the global segment file per bam file
    n_tasks_mosdepth = len(bams)
    n_workers_mosdepth, threads_per_task = workload_assignment(processes, n_tasks_mosdepth)
    log(msg=f"mosdepth-num_worker={n_workers_mosdepth}\tnum_tasks={n_tasks_mosdepth}\tthreads_per_task=" + \
        ','.join(str(s) for s in threads_per_task) + '\n', level='STEP')
    # Note: These function calls are the only section that uses mosdepth
    mosdepth_params = [
        (
            segment_file,
            outdir,
            names[i],
            bams[i],
            threads_per_task[i],
            mosdepth,
            readquality,
        )
        for i in range(n_tasks_mosdepth)
    ]
    try:
        with Pool(n_workers_mosdepth) as p:
            p.map(run_mosdepth_rg, mosdepth_params)
    except Exception as e:
        log(msg=f"ERROR! mosdepth raise exception: {e}\n",level='ERROR')
        p.terminate()
        raise ValueError()
    finally:
        p.join()
        log(msg="All mosdepth finished\n", level='STEP')
    
    # 
    # compute count arrays
    n_tasks_count_array = len(chromosomes)
    n_workers_count_array, _ = workload_assignment(processes, n_tasks_count_array)
    log(msg=f"count_array-num_worker={n_workers_count_array}\tnum_tasks={n_tasks_count_array}\n", level='STEP')
    count_array_params = [
        (
            outdir,
            use_chr,
            names,
            ch
        )
        for ch in chromosomes
    ]
    try:
        with Pool(n_workers_count_array) as p:
            p.map(run_count_array, count_array_params)
    except Exception as e:
            log(msg=f"ERROR! count_array raise exception: {e}\n",level='ERROR')
            p.terminate()
            raise ValueError()
    finally:
        p.join()
        log(msg="All count_array finished\n", level='STEP')

    
    np.savetxt(os.path.join(outdir, 'samples.txt'), names, fmt='%s')
    if len(check_array_files(outdir, chromosomes)) > 0:
            raise ValueError(error('Missing some output arrays!'))
    log(
        msg='# Array forming completed successfully, removing intermediate count files. \n',
        level='STEP',
    )

    totals_file = os.path.join(outdir, 'total.tsv')
    if os.path.exists(totals_file):
        log(msg='# Found total reads file, exiting. \n', level='STEP')
        return

    log(
        msg='# Counting total number of reads for normal and tumor samples\n',
        level='STEP',
    )
    # TODO total counts might be redundant with above count chromosome step
    total_counts = tc.tcount(
        samtools=samtools,
        samples=[(bams[i], names[i]) for i in range(len(names))],
        chromosomes=chromosomes,
        num_workers=processes,
        q=readquality,
    )

    try:
        total = {name: sum(total_counts[name, chromosome] for chromosome in chromosomes) for name in names}
    except KeyError:
        raise KeyError(error('Either a chromosome or a sample has not been considered in the total counting!'))

    log(
        msg='# Writing the total read counts for all samples in {}\n'.format(totals_file),
        level='STEP',
    )
    with open(totals_file, 'w') as f:
        for name in names:
            f.write('{}\t{}\n'.format(name, total[name]))
    
    log(msg='count-reads-ont completed successfully\n', level='STEP')
    return

"""
run count array per chromosome

Returns: <n> x <2d> np.ndarray
    at ith segment [left_i, right_i), for jth sample
    entry [i, 2j]=#reads start in segment [left_i, right_i)
    entry [i, 2j + 1]=segment mean read depth, given by mosdepth
"""
def _run_count_array(outdir: str, use_chr: bool, all_names: list, chromosome: str):
    try:
        [tot_file, thres_file] = get_array_file_path(outdir, chromosome)
        reg_df_ch, _ = load_seg_file(thres_file, use_chr, [])
        segments = reg_df_ch[["START", "END"]].to_numpy(dtype=np.uint32)
        num_segments = len(segments)

        arr = np.zeros((num_segments, len(all_names) * 2))
        
        for idx, name in enumerate(all_names):
            starts_file = os.path.join(outdir, f"{name}.{chromosome}.starts.gz")
            fd = gzip.open(starts_file, 'r') if starts_file.endswith('.gz') else open(starts_file, 'r')
            read_starts = np.array([int(a) for a in fd])
            num_reads = len(read_starts)
            fd.close()
            for sdx, [sstart, sstop] in enumerate(segments):
                left_idx = np.argmax(read_starts >= sstart)
                if read_starts[left_idx] < sstart:
                    continue
                right_idx = np.argmax(read_starts >= sstop)
                if read_starts[right_idx] < sstop:
                    right_idx = num_reads - 1
                arr[sdx, 2*idx] = right_idx - left_idx

            mosdp_rg_file = os.path.join(outdir, f"{name}.regions.bed.gz")
            reg_df, _ = load_seg_file(mosdp_rg_file, use_chr, additional_columns=["DEPTH"])
            reg_df = reg_df[reg_df["CHR"] == chromosome]
            arr[:, 2*idx + 1] = reg_df["DEPTH"].to_numpy()

        np.savetxt(tot_file, arr, fmt='%d')
    except Exception as e:
        log(f"ERROR! count array {chromosome}: {e}\n", level="ERROR")
        raise e
    finally:
        log(f'Done count array {chromosome}\n', level='STEP')
        return arr

def run_count_array(params):
    return _run_count_array(*params)


"""
run mosdepth in region format for one sample
segment_file: all regions for all chromosomes gzipped
"""
def _run_mosdepth_rg(segment_file: str, outdir: str, sample_name: str, 
                     bam_file: str, threads: int, 
                    mosdepth: str, readquality: int):
    try:
        log(
                f'Run mosdepth (region) for {sample_name}\n',
                level='STEP',
            )
        sys.stderr.flush()

        out_prefix = os.path.join(outdir, sample_name)
        if os.path.exists(f"{out_prefix}.regions.bed.gz") and \
            os.path.exists(f"{out_prefix}.regions.bed.gz.csi"):
            log(
                f'Skipping mosdepth on sample {sample_name} ({out_prefix}* exists)\n',
                level='STEP',
            )
        else:
            msdp_cmd = [mosdepth, 
                        '-t', str(threads), 
                        '-Q', str(readquality),
                        '--by', segment_file,
                        '--no-per-base',
                        out_prefix,
                        bam_file]
            err_fd = open(f"{outdir}/run_mosdepth_{sample_name}.err.log", 'w')
            out_fd = open(f"{outdir}/run_mosdepth_{sample_name}.out.log", 'w')
            ret = sp.run(msdp_cmd, stdout=out_fd, stderr=err_fd)
            err_fd.close()
            out_fd.close()
            ret.check_returncode()
    except Exception as e:
        log(f"ERROR! mosdepth region exception {e}", level="ERROR")
        raise e
    finally:
        log(f'Done mosdepth on sample {sample_name}\n', level='STEP')
        return None

def run_mosdepth_rg(params):
    return _run_mosdepth_rg(*params)


if __name__ == '__main__':
    main()
