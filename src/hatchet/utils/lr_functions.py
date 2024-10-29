import os
import subprocess
import gzip
import numpy as np
import pandas as pd
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
    load_mosdepth_files,
    init_bb_dataframe
)


def genotype_snps(args):
    """
    This functions creates extracts each chromosome from the genotype file,
    and saves it to a separate file in the output directory.
    """
    log(
        msg=(
            '# Parsing the input arguments, checking the consistency of given files, and extracting required ',
            'information\n',
        ),
        level='STEP',
    )
    args = parse_genotype_snps_arguments(args)
    logArgs(args, 80)

    for chrom in args['chromosomes']:
        log(
            msg=('Extracting chromosome {} from genotype file...\n'.format(chrom),),
            level='STEP',
        )

        # call bcftools to extract chromosome
        output_dir = args['outputsnps']
        output_vcf = f'{output_dir}/{chrom}.vcf.gz'
        mindp = args['mincov']
        maxdp = args['maxcov']
        command = [
            args['bcftools'],
            'view',
            '-r',
            chrom,
            '-i',
            f'FILTER="PASS" & strlen(REF)=1 & strlen(ALT)=1 & SUM(FORMAT/AD)>={mindp} & SUM(FORMAT/AD)<={maxdp} & N_ALT=1 & (FORMAT/GT[0]=="1|0" | FORMAT/GT[0]=="0|1")',
            args['snps'],
            '-Oz',
            '-o',
            output_vcf,
        ]
        process = subprocess.Popen(command)
        process.wait()  # Wait for the process to complete


def phase_snps(args):
    """
    This functions creates extracts each phased genotype from genotpes
    file and saves it to phased.vcf.gz in the output directory.
    """
    log(
        msg=(
            '# Parsing the input arguments, checking the consistency of given files, and extracting required ',
            'information\n',
        ),
        level='STEP',
    )
    args = parse_genotype_snps_arguments(args)
    logArgs(args, 80)

    # call bcftools to extract 0|1 and 1|0 snps
    output_dir = args['outputsnps']
    output_vcf = f'{output_dir}/phased.vcf.gz'
    mindp = args['mincov']
    maxdp = args['maxcov']
    command = [
        args['bcftools'],
        'view',
        '-i',
        f'FILTER="PASS" & strlen(REF)=1 & strlen(ALT)=1 & SUM(FORMAT/AD)>={mindp} & SUM(FORMAT/AD)<={maxdp} & N_ALT=1 & (FORMAT/GT[0]=="1|0" | FORMAT/GT[0]=="0|1")',
        args['snps'],
        '-Oz',
        '-o',
        output_vcf,
    ]
    process = subprocess.Popen(command)
    process.wait()  # Wait for the process to complete


def get_b_count(df):
    # Select REF if FLIP == 1, otherwise select ALT, then sum the selected values
    sel = df.apply(lambda row: row['REF'] if row['FLIP'] == 1 else row['ALT'], axis=1)
    total_sum = sel.sum()
    return total_sum


def get_haplostring(df):
    return ''.join(list(df['FLIP'].astype(int).astype(str)))


def count_reads_lr(args):

    args = parse_count_reads_args(args)
    logArgs(args, 80)

    bams = args['bams']
    names = args['names']

    if not args['nonormal']:
        tbams, tnames = zip(*sorted(zip(*(bams[1:], names[1:])), key=lambda x: x[1]))
        bams = [bams[0]] + list(tbams)
        names = [names[0]] + list(tnames)

    chromosomes = args['chromosomes']
    processes = args['j']
    outdir = args['outdir']
    mosdepth = args['mosdepth']
    tabix = args['tabix']
    readquality = args['readquality']

    def run_mosdepth(bam, name, outdir, mosdepth, chromosomes, readquality):
        output_prefix = os.path.join(outdir, name)
        cmd = [
            mosdepth,
            '-n',
            '--fast-mode',
            '-t',
            '1',
            '-b',
            '500',
            '-Q',
            str(readquality),
            '-c',
            ','.join(chromosomes),
            output_prefix,
            bam,
        ]
        subprocess.run(cmd, check=True)
        return f'{output_prefix}.regions.bed.gz'

    def process_bed_file(bed_file, name, outdir):
        output_file = os.path.join(outdir, f'{name}_with_sample.bed')
        with open(output_file, 'w') as out_f:
            with subprocess.Popen(['zcat', bed_file], stdout=subprocess.PIPE, text=True) as proc:
                for line in proc.stdout:
                    chrom, start, end, depth = line.strip().split('\t')
                    out_f.write(f'{chrom}\t{start}\t{end}\t{depth}\t{name}\n')
        return output_file

    def split_and_compress_by_chromosome(combined_bed, outdir, chromosomes, tabix):
        chrom_files = {chrom: open(os.path.join(outdir, f'{chrom}.bed'), 'w') for chrom in chromosomes}

        with open(combined_bed, 'r') as f:
            for line in f:
                chrom = line.split('\t')[0]
                if chrom in chrom_files:
                    chrom_files[chrom].write(line)

        for chrom, file in chrom_files.items():
            file.close()
            output_gz = f'{file.name}.gz'
            subprocess.run(['bgzip', '-f', file.name], check=True)
            subprocess.run([tabix, '-p', 'bed', output_gz], check=True)

    # 1. Compute average depth using mosdepth
    mosdepth_outputs = {}
    for bam, name in zip(bams, names):
        mosdepth_outputs[name] = run_mosdepth(bam, name, outdir, mosdepth, chromosomes, readquality)

    # 2. Combine the resulting bed files and add sample name column
    processed_bed_files = []
    for name, bed_file in mosdepth_outputs.items():
        processed_bed_files.append(process_bed_file(bed_file, name, outdir))

    # Combine all processed bed files
    combined_bed = os.path.join(outdir, 'combined_samples.bed')
    with open(combined_bed, 'w') as outfile:
        for bed_file in processed_bed_files:
            with open(bed_file, 'r') as infile:
                outfile.write(infile.read())
            os.remove(bed_file)  # Remove individual processed bed files

    # 3. Split the combined bed file by chromosome, compress, and index
    split_and_compress_by_chromosome(combined_bed, outdir, chromosomes, tabix)

    # Clean up the combined bed file
    os.remove(combined_bed)


def combine_counts(args, haplotype_file, mosdepth_files):
    log(msg='# Parsing and checking input arguments\n', level='STEP')
    args = parse_combine_counts_args(args)
    logArgs(args, 80)

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
    rd_array = args['array']

    haplotype_blocks = load_gtf_file(haplotype_file)[['CHR', 'START', 'END']]

    bed_mosdepths = load_mosdepth_files(all_names, mosdepth_files)

    # bb/bulk.bb
    big_bb = init_bb_dataframe()
    for ch in chromosomes:
        positions, snp_counts, snpsv = read_snps(baffile, ch, all_names, phasefile=phase)
        # directly read per-SNP counts and per-SNP bin bounderies
        total_counts, complete_thresholds = read_total_and_thresholds(ch, rd_array)
        hap_blocks_ch = haplotype_blocks[haplotype_blocks.CHR == ch] #TODO use groupby

        # all mosdepth from same chromosome across all samples; n is sample name
        mosdepth_ch = [(n, mos[mos.CHR == ch]) for n, mos in bed_mosdepths]

        # iterate through every haplotype block
        for _, row in hap_blocks_ch.iterrows():
            block_bb = init_bb_dataframe()
            mos_block = [(n, mos[(mos.START > row.START - 1000) & (mos.END < row.END + 1000)]) 
                         for n, mos in mosdepth_ch]

            block_snps_idx = np.where((positions >= row.START) & (positions <= row.END))[0]
            if len(block_snps_idx) == 0:
                # no SNPs in current block
                continue
            # extract snp positions within the haplotype block region
            block_snp_pos = positions[block_snps_idx]
            block_snp_counts = snp_counts[block_snps_idx]
            assert len(block_snp_counts) != 0 #TODO this guard must satisfied

            # extract thresholds coverring all the extracted SNPs.
            # works if thresholds are defined in default manner (per threshold per snp)
            block_thr_idx = np.where(
                (complete_thresholds >= block_snp_pos[0]) & (complete_thresholds <= block_snp_pos[-1])
            )[0]
            if len(block_snps_idx) > len(block_thr_idx) + 1 or len(block_thr_idx) == 0:
                # centromere loci
                # FIXME use other format than threshold to account for non-consecutive segments
                log(f"SKIP centromere region with h-block: {row.START}-{row.END}, \
                    len(block_snps_idx):{len(block_snps_idx)}, len(block_thr_idx):{len(block_thr_idx)} \
                        block_snp_pos[0]:{block_snp_pos[0]}, block_snp_pos[-1]:{block_snp_pos[0]}", level="STEP")
                continue
            log(
                msg=f'snpcount {len(block_snps_idx)} thrcount {len(block_thr_idx)}\n',
                level='STEP',
            )

            # thresholds must begin and end outside the boundaries:
            # TODO block_thr_idx index assumption
            if block_thr_idx[0] > 0:
                block_thr_idx = np.concatenate([[block_thr_idx[0] - 1], block_thr_idx])
            if block_thr_idx[-1] < len(complete_thresholds) - 1:
                block_thr_idx = np.concatenate([block_thr_idx, [block_thr_idx[-1] + 1]])

            block_thr_pos = complete_thresholds[block_thr_idx]
            block_thr_counts = total_counts[block_thr_idx]

            (starts, ends, totals, rdrs) = adaptive_bins_arm(
                snp_thresholds=block_thr_pos,
                total_counts=block_thr_counts,
                snp_positions=block_snp_pos,
                snp_counts=block_snp_counts,
                chromosome=ch,
                min_snp_reads=msr,
                min_total_reads=1, #TODO
                nonormalFlag=nonormalFlag,
                mos_block=mos_block,
            )

            dfs = [snpsv[(snpsv.POS >= starts[i]) & (snpsv.POS <= ends[i])] for i in range(len(starts))]
            dfs = [df.dropna(subset=['FLIP']) for df in dfs]

            for i, df in enumerate(dfs):
                if len(df) == 0:
                    continue
                for sample, df2 in df.groupby('SAMPLE'):
                    start = starts[i]
                    end = ends[i]
                    total = int(df2.TOTAL.sum())
                    bcount = int(get_b_count(df2))
                    # bcount = min(bcount, total - bcount)
                    haplostring = get_haplostring(df2)
                    # log(
                    #     msg=f'{ch} {sample} {start} {end} {total} {bcount} {bcount/total}\n',
                    #     level='STEP',
                    # )
                    if np.isnan(rdrs[i][0]) or (totals[i][1] == 0 and totals[i][0] == 0):
                        # rd cannot be computed since there are no reads in tumor or normal
                        continue
                    tot_reads = round(min(20*totals[i][0],totals[i][1]))
                    rd = int(min(20, rdrs[i][0]))
                    nor_reads = int(totals[i][0])
                    bin_row = [
                        ch,
                        'unit',
                        start,
                        end,
                        sample,
                        rd,
                        tot_reads,
                        nor_reads,
                        len(df2),
                        bcount,
                        total,
                        '',
                        '',
                        '',
                        '',
                        bcount / total,
                        row.START,
                        row.END,
                    ]
                    block_bb.loc[len(block_bb)] = bin_row

            # find the rows for which the SAMPLE == all_names[0]
            if nonormalFlag:
                first_tum_index = 0
            else:
                first_tum_index = 1
            rowsfirst_sample_block_rows = block_bb[block_bb.SAMPLE == all_names[first_tum_index]]
            medianbaf = np.median(rowsfirst_sample_block_rows.BAF)
            if medianbaf > 0.5:
                flip_bafs = True
            else:
                flip_bafs = False
            if flip_bafs:
                # apply BAF = 1-BAF to all rows in block_bb
                block_bb['BAF'] = 1 - block_bb['BAF']

            # append block_bb to big_bb
            big_bb = pd.concat([big_bb, block_bb], ignore_index=True)

    # For each sample, correct read counts to account for differences in coverage (as in HATCHet)
    # (i.e., multiply read counts by total-reads-normal/total-reads-sample)
    rc = pd.read_table(args['totalcounts'], header=None, names=['SAMPLE', '#READS'])
    normal_name = all_names[0]
    nreads_normal = rc[rc.SAMPLE == normal_name].iloc[0]['#READS']
    if nonormalFlag:
        for sample, df in big_bb.groupby('SAMPLE'):
            mean_rd = np.mean(df['RD'])
            big_bb.loc[big_bb.SAMPLE == sample, 'RD'] = df['RD'] / mean_rd
    else:
        for sample in rc.SAMPLE.unique():
            if sample == normal_name:
                continue
            nreads_sample = rc[rc.SAMPLE == sample].iloc[0]['#READS']
            correction = nreads_normal / nreads_sample
            my_bb = big_bb[big_bb.SAMPLE == sample]

            # Correct the tumor reads propotionally to the total reads in corresponding samples
            big_bb.loc[big_bb.SAMPLE == sample, 'RD'] = my_bb.RD * correction

    if args['gc_correct']:
        log(
            msg='# In long read sequencing pipeline, GC correction for RD is not recommended\n',
            level='WARN',
        )
        log(
            msg='# Performing GC bias correction on read depth signal\n',
            level='STEP',
        )
        big_bb = rd_gccorrect(big_bb, referencefasta)
    big_bb.END = big_bb.END + 1
    big_bb.to_csv(outfile, index=False, sep='\t')
    return
