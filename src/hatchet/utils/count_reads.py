import sys
import os
from multiprocessing import Pool
import subprocess as sp
import numpy as np
import pandas as pd
import gzip
import subprocess
import traceback
from importlib.resources import path
import hatchet.data

from hatchet.utils.ArgParsing import parse_count_reads_args
from hatchet.utils.Supporting import log, logArgs, error
import hatchet.utils.TotalCounting as tc

from hatchet.utils.additional_features import (
    load_seg_file,
    get_array_file_path,
    check_array_files,
    expected_count_files,
    check_count_files,
    workload_assignment,
    read_snps
)

def main(args=None):
    log(msg='# Parsing and checking input arguments\n', level='STEP')
    args = parse_count_reads_args(args)
    logArgs(args, 80)

    bams = args['bams']
    names = args['names']

    if not args['nonormal']:
        tbams, tnames = zip(*sorted(zip(*(bams[1:], names[1:])), key=lambda x: x[1]))
        bams = [bams[0]] + list(tbams)
        names = [names[0]] + list(tnames)
    # TODO sort the bams when nonormal is TRUE

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

    use_prebuilt_segfile = False
    if refversion != None: # segfile == None
        use_prebuilt_segfile = True
        segfile = path(hatchet.data, f'{args["refversion"]}.segments.tsv')
    
    _, seg_chroms = load_seg_file(segfile, use_chr)

    # only chromosomes that present in both BAM file and segment file are handled
    chromosomes = list(set(chromosomes).intersection(seg_chroms))
    if len(chromosomes) == 0:
        ValueError(error("No chromosomes present in both BAM file and segment file / refversion"))

    if len(check_array_files(outdir, chromosomes, use_prebuilt_segfile)) == 0:
        log(
            msg='# Found all array files, skipping to total read counting. \n',
            level='STEP',
        )
    else:

        if len(check_count_files(outdir, chromosomes, names)) == 0:
            log(
                msg='# Found all count files, skipping to forming arrays\n',
                level='STEP',
            )

        else:
            # moved to argparse
            # if args['segfile'] and (not os.path.exists(args['segfile'])):
            #     raise ValueError(error('A path to a nonexistant segfile was given to count-reads!'))

            params = zip(
                np.repeat(chromosomes, len(bams)),
                [outdir] * len(bams) * len(chromosomes),
                [samtools] * len(bams) * len(chromosomes),
                bams * len(chromosomes),
                names * len(chromosomes),
                [readquality] * len(bams) * len(chromosomes),
            )

            n_workers_samtools, _ = workload_assignment(processes, len(bams) * len(chromosomes))
            with Pool(n_workers_samtools) as p:   # divide by 2 because each worker starts 2 processes
                p.map(count_chromosome_wrapper, params)

            n_workers_mosdepth, threads_per_task = workload_assignment(processes, len(bams))

            # Note: These function calls are the only section that uses mosdepth
            mosdepth_params = [
                (
                    outdir,
                    names[i],
                    bams[i],
                    threads_per_task[i],
                    mosdepth,
                    readquality,
                )
                for i in range(len(bams))
            ]
            with Pool(n_workers_mosdepth) as p:
                p.map(mosdepth_wrapper, mosdepth_params)

            nonexist_count_files = check_count_files(outdir, chromosomes, names)
            if len(nonexist_count_files) > 0:
                print('\t'.join(nonexist_count_files))
                raise ValueError(error('Missing some counts files!'))

        # Use Tabix to index per-position coverage bed files for each sample
        for name in names:
            perpos_file = os.path.join(outdir, name + '.per-base.bed.gz')
            # sync call
            subprocess.run([tabix, '-f', perpos_file])

        # form parameters for each worker
        params = [
            (
                outdir,
                names,
                ch,
                baffile,
                tabix,
                use_prebuilt_segfile,
                segfile,
            )
            for ch in chromosomes
        ]
        n_workers, _ = workload_assignment(processes, len(chromosomes))
        # dispatch workers
        with Pool(n_workers) as p:
            p.map(run_chromosome_wrapper, params)

        np.savetxt(os.path.join(outdir, 'samples.txt'), names, fmt='%s')

        if len(check_array_files(outdir, chromosomes, use_prebuilt_segfile)) > 0:
            raise ValueError(error('Missing some output arrays!'))
        # successful
        log(
            msg='# Array forming completed successfully, removing intermediate count files. \n',
            level='STEP',
        )
        [os.remove(f) for f in expected_count_files(outdir, chromosomes, names)]

    totals_file = os.path.join(outdir, 'total.tsv')
    if os.path.exists(totals_file):
        log(msg='# Found total reads file, exiting. \n', level='STEP')
        return

    # TODO: take -q option and pass in here
    log(
        msg='# Counting total number of reads for normal and tumor samples\n',
        level='STEP',
    )
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
    return

def mosdepth_wrapper(params):
    run_mosdepth(*params)


def run_mosdepth(outdir, sample_name, bam, threads, mosdepth, readquality):
    try:
        last_file = os.path.join(outdir, sample_name + '.per-base.bed.gz.csi')
        if os.path.exists(last_file):
            log(
                f'Skipping mosdepth on sample {sample_name} (output file {last_file} exists)\n',
                level='STEP',
            )
            return

        log(
            f'Starting mosdepth on sample {sample_name} with {threads} threads\n',
            level='STEP',
        )
        sys.stderr.flush()

        md = sp.run(
            [
                mosdepth,
                '-t',
                str(threads),
                '-Q',
                str(readquality),
                os.path.join(outdir, sample_name),
                bam,
            ]
        )
        md.check_returncode()
        log(f'Done mosdepth on sample {sample_name}\n', level='STEP')

    except Exception as e:
        log('Exception in countPos: {}\n'.format(e), level='ERROR')
        raise e

# {sample_name}.{ch}.starts.gz
def count_chromosome(ch, outdir, samtools, bam, sample_name, readquality, compression_level=6):
    try:
        outfile = os.path.join(outdir, f'{sample_name}.{ch}.starts')
        if os.path.exists(outfile):
            log(
                f'Skipping sample {sample_name} chromosome {ch} (output file exists)\n',
                level='STEP',
            )
            return

        log(f'Sample {sample_name} -- Starting chromosome {ch}\n', level='STEP')
        sys.stderr.flush()

        # Get start positions
        st = sp.Popen((samtools, 'view', '-q', str(readquality), bam, ch), stdout=sp.PIPE)
        cut = sp.Popen(('cut', '-f', '4'), stdin=st.stdout, stdout=sp.PIPE)
        gzip = sp.Popen(
            ('gzip', '-{}'.format(compression_level)),
            stdin=cut.stdout,
            stdout=open(outfile + '.gz', 'w'),
        )
        st.wait()
        cut.wait()
        gzip.wait()
        if st.returncode != 0:
            raise ValueError('samtools subprocess returned nonzero value: {}'.format(st.returncode))
        if cut.returncode != 0:
            raise ValueError('cut subprocess returned nonzero value: {}'.format(cut.returncode))

        log(f'Sample {sample_name} -- Done chromosome {ch}\n', level='STEP')

    except Exception as e:
        log('Exception in countPos: {}\n'.format(e), level='ERROR')
        raise e


def count_chromosome_wrapper(param):
    count_chromosome(*param)

def form_counts_array(starts_files, perpos_files, thresholds, chromosome, tabix, chunksize=1e5):
    """
    NOTE: Assumes that starts_files[i] corresponds to the same sample as perpos_files[i]
    Parameters:
        starts_files: list of <sample>.<chromosome>.starts.gz files each containing a list of start positions
        perpos_files: list of <sample>.per-base.bed.gz files containing per-position coverage from mosdepth
        thresholds: list of potential bin start positions (thresholds between SNPs)
        chromosome: chromosome to extract read counts for

    Returns: <n> x <2d> np.ndarray
        entry [i, 2j] contains the number of reads starting in (starts[i], starts[i + 1]) in sample j
        entry [i, 2j + 1] contains the number of reads covering position starts[i] in sample j
    """
    arr = np.zeros((thresholds.shape[0] + 1, len(starts_files) * 2))   # add one for the end of the chromosome

    for i in range(len(starts_files)):
        # populate starts in even entries
        fname = starts_files[i]
        fd = gzip.open(fname, 'r') if fname.endswith('.gz') else open(fname, 'r')
        read_starts = [int(a) for a in fd]
        fd.close()

        # if fname.endswith('.gz'):
        #     f = gzip.open(fname)
        # else:
        #     f = open(fname)

        # cx = gzip.open(fname, 'r')
        # read_starts = [int(a) for a in cx]

        # end = thresholds[0]

        # read_starts counter 
        j = 0
        for idx in range(1, len(thresholds)):
            start = thresholds[idx - 1]
            end = thresholds[idx]

            # count #read-starts in range [start, end)
            # assume read_starts is pre-sorted
            my_count = 0
            while j < len(read_starts) and read_starts[j] < end:
                # unnecessary check ALA threshold are adjacent with 1-base difference.
                assert read_starts[j] >= start
                my_count += 1
                j += 1

            arr[idx - 1, i * 2] = my_count

        arr[idx - 1, i * 2] += len(read_starts) - j
        # f.close()

    for i in range(len(perpos_files)):
        # populate threshold coverage in odd entries
        fname = perpos_files[i]

        # <sample>.per-base.bed.gz
        chr_sample_file = os.path.join(fname[:-3] + '.' + chromosome)

        if not os.path.exists(chr_sample_file):
            with open(chr_sample_file, 'w') as f:
                subprocess.run([tabix, fname, chromosome], stdout=f)

        with open(chr_sample_file, 'r') as records:
            idx = 0
            last_record = None
            for line in records:
                tokens = line.split()
                if len(tokens) == 4:
                    start = int(tokens[1])
                    end = int(tokens[2])
                    nreads = int(tokens[3])

                    while idx < len(thresholds) and thresholds[idx] - 1 < end:
                        assert thresholds[idx] - 1 >= start
                        arr[idx, (2 * i) + 1] = nreads
                        idx += 1
                    last_record = line

            if i == 0:
                # identify the (effective) chromosome end as the last well-formed record
                assert idx == len(thresholds)
                _, _, chr_end, end_reads = last_record.split()
                chr_end = int(chr_end)
                end_reads = int(end_reads)

                assert chr_end > thresholds[-1]
                assert len(thresholds) == len(arr) - 1

                # add the chromosome end to thresholds
                thresholds = np.concatenate([thresholds, [chr_end]])

                # count the number of reads covering the chromosome end
                arr[idx, 0] = end_reads
            records.close()

        if os.path.exists(chr_sample_file):
            os.remove(chr_sample_file)

    return arr, thresholds


def get_chr_end(stem, all_names, chromosome):
    starts_files = []
    for name in all_names:
        starts_files.append(os.path.join(stem, name + '.' + chromosome + '.starts.gz'))

    last_start = 0
    for sfname in starts_files:
        zcat = subprocess.Popen(('zcat', sfname), stdout=subprocess.PIPE)
        tail = subprocess.Popen(('tail', '-1'), stdin=zcat.stdout, stdout=subprocess.PIPE)
        my_last = int(tail.stdout.read().decode('utf-8').strip())

        if my_last > last_start:
            last_start = my_last

    return last_start


def run_chromosome(
    outdir,
    all_names,
    chromosome,
    baf_file,
    tabix,
    use_prebuilt_segfile,
    seg_file,
):
    """
    Construct arrays that contain all counts needed to perform adaptive binning for a single chromosome
    (across all samples).
    """

    try:
        [totals_out, thresholds_out] = get_array_file_path(outdir, chromosome, use_prebuilt_segfile)

        if os.path.exists(totals_out) and os.path.exists(thresholds_out):
            log(
                msg=f'Output files already exist, skipping chromosome {chromosome}\n',
                level='INFO',
            )
            return

        log(msg=f'Loading chromosome {chromosome}\n', level='INFO')
        # Per-position coverage bed files for each sample
        perpos_files = [os.path.join(outdir, name + '.per-base.bed.gz') for name in all_names]

        # Identify the start-positions files for this chromosome
        starts_files = [os.path.join(outdir, f"{name}.{chromosome}.starts.gz") for name in all_names]

        log(
            msg=f'Reading SNPs file for chromosome {chromosome}\n',
            level='INFO',
        )
        # Load SNP positions and counts for this chromosome

        if chromosome.endswith('X') or chromosome.endswith('Y'):
            log(
                msg='Running on sex chromosome -- ignoring SNPs and min SNP reads\n',
                level='INFO',
            )

            # TODO: do this procedure only for XY
            last_start = get_chr_end(outdir, all_names, chromosome)
            positions = np.arange(5000, last_start, 5000)

        else:
            positions, _, _ = read_snps(baf_file, chromosome, all_names)

        seg_df, _ = load_seg_file(seg_file, chromosome.startswith("chr"))
        seg_df_ch: pd.DataFrame = seg_df[seg_df["CHR"] == chromosome]
        if use_prebuilt_segfile:
            assert len(seg_df_ch) == 2 # p-arm and q-arm, can be removed after double check
            centromere_start = seg_df_ch.END.min() - 1 # inclusive
            centromere_end = seg_df_ch.START.max() - 1 # inclusive
            # create segments for computing read depth, these will be the midpoints of germline SNPs
            # during the adaptive binning process in combine_counts.py, these segments will be combined in order to meet
            # QC metrics, i.e. the minimum number of SNP-covering reads
            # REF_START(1), S1, T1, S2, T2 ... Sn, Tn, REF_END
            thresholds = np.trunc(np.vstack([positions[:-1], positions[1:]]).mean(axis=0)).astype(np.uint32)

            last_idx_p = np.argwhere(thresholds > centromere_start)[0][0]
            first_idx_q = np.argwhere(thresholds > centromere_end)[0][0]
            thresholds = np.concatenate(
                [
                    [1],
                    thresholds[:last_idx_p],
                    [centromere_start],
                    [centromere_end],
                    thresholds[first_idx_q:],
                ]
            )
        else:
            thresholds = seg_df_ch["START"].to_numpy(np.uint64)
            if len(thresholds) > 0:
                if thresholds[0] != 1:
                    thresholds = np.concatenate([[1], thresholds])

        # TODO raise error if thresholds is empty as well
        if np.any(np.diff(thresholds) < 0):
            raise ValueError(f'improper negative interval in provided segment file for chromosome {chromosome}')

        log(msg=f'Loading counts for chromosome {chromosome}\n', level='INFO')
        total_counts, complete_thresholds = form_counts_array(
            starts_files, perpos_files, thresholds, chromosome, tabix=tabix
            )
        np.savetxt(totals_out, total_counts, fmt='%d')
        np.savetxt(thresholds_out, complete_thresholds, fmt='%d')

        log(msg=f'Done chromosome {chromosome}\n', level='INFO')
    except Exception as e:
        print(f'Error in chromosome {chromosome}:')
        print(e)
        traceback.print_exc()
        raise e


def run_chromosome_wrapper(param):
    run_chromosome(*param)

if __name__ == '__main__':
    main()
