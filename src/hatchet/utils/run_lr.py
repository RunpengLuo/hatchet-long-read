from io import StringIO
import os.path
import sys
import glob
import argparse

from hatchet import config
from hatchet.utils.count_reads_lr import main as count_reads
from hatchet.utils.combine_counts_lr import main as combine_counts
from hatchet.utils.genotype_snps_lr import main as genotype_snps
from hatchet.utils.phase_snps_lr import main as phase_snps

from hatchet.utils.count_alleles_lr2 import main as count_alleles
from hatchet.utils.cluster_bins_gmm import main as cluster_bins_gmm
from hatchet.utils.cluster_bins import main as cluster_bins
from hatchet.utils.plot_bins import main as plot_bins
from hatchet.utils.plot_bins_1d2d import main as plot_bins_1d2d
from hatchet.bin.HATCHet import main as hatchet_main
from hatchet.utils.plot_cn import main as plot_cn
from hatchet.utils.plot_cn_1d2d import main as plot_cn_1d2d
from hatchet.utils.Supporting import log, error

def main(args=None):
    log(msg="hatchet longread test version\n", level="STEP")
    parser = argparse.ArgumentParser(prog="hatchet run-lr", description="Run HATCHet pipeline on long read data")
    parser.add_argument("inifile", help=".ini file for run configuration")
    args = parser.parse_args(args)

    config.read(args.inifile)

    output = config.run.output
    output = output.rstrip("/")
    os.makedirs(output, exist_ok=True)

    try:
        chromosomes = [c for c in config.run.chromosomes.split()]
    except (KeyError, AttributeError):  # if key is absent or is blank (None)
        chromosomes = []  # process all

    extra_args = []
    try:
        if config.run.processes is not None:
            extra_args = ["-j", str(config.run.processes)]
    except KeyError:
        pass
    
    # ----------------------------------------------------
    if config.run.genotype_snps:
        genotype_file = config.run.ont_genotype_file
        if not os.path.exists(genotype_file):
            raise RuntimeError(
                (
                    "Please specify a valid VCF file (genotype_file) which"
                    "contains genotyped & phased SNPs for the normal"
                )
            )
        os.makedirs(f"{output}/snps", exist_ok=True)
        params = [
            "-N",
            config.run.normal,
            "-r",
            config.run.reference,
            "-R",
            genotype_file,
            "-o",
            f"{output}/snps/",
            "--chromosomes",
            *(chromosomes or []),
            *extra_args
        ]
        genotype_snps(params)

    # ----------------------------------------------------
    if config.run.phase_snps:
        genotype_file = config.run.ont_genotype_file
        if not os.path.exists(genotype_file):
            raise RuntimeError(
                (
                    "Please specify a valid VCF file (genotype_file) which"
                    "contains genotyped & phased SNPs for the normal"
                )
            )
        os.makedirs(f"{output}/phase", exist_ok=True)
        params = [
            "-N",
            config.run.normal,
            "-r",
            config.run.reference,
            "-R",
            genotype_file,
            "-o",
            f"{output}/phase/",
            "--chromosomes",
            *(chromosomes or []),
            *extra_args
        ]
        phase_snps(params)

    # ----------------------------------------------------
    if config.run.count_alleles:
        sample_bams = config.run.bams.split()
        sample_names = ("normal " + config.run.samples).split()
        ref_file = config.run.reference
        os.makedirs(f"{output}/baf", exist_ok=True)
        if not os.path.isfile(f"{output}/baf/normal.1bed") or \
            not os.path.isfile(f"{output}/baf/tumor.1bed"):
            params = [
                "-N", config.run.normal,
                "-T", *sample_bams,
                "-S", *sample_names,
                "-r", ref_file,
                "-L", *glob.glob(f"{output}/snps/*.vcf.gz"),
                "-O", f"{output}/baf/normal.1bed",
                "-o", f"{output}/baf/tumor.1bed",
                "-l", f"{output}/baf",
                "--chromosomes", *(chromosomes or []),
                *extra_args
            ]
            count_alleles(params)
        else:
            log(
                msg=f"skip, baf/normal.1bed and baf/tumor.1bed exists\n",
                level="STEP",
            )

    # ----------------------------------------------------
    if config.run.fixed_width:
        # Old fixed-width binning

        # throw an exception with the error message saying that fixed_width is not supported with long reads
        raise RuntimeError(
            (
                "Fixed-width binning is not supported with long reads. Please use the adaptive binning method."
            )
        )

    # ----------------------------------------------------
    # Variable-width/adaptive binning
    if config.run.count_reads:
        ref_ver = ["-V", config.genotype_snps.reference_version]
        if config.count_reads.segfile != None:
            ref_ver = ["--segfile", config.count_reads.segfile] # override refversion
        
        sample_bams = config.run.bams.split()
        sample_names = config.run.samples.split()
        if config.run.normal != None:
            sample_names = ["normal"] + sample_names
        ref_file = config.run.reference
    
        os.makedirs(f"{output}/rdr", exist_ok=True)
        params = [
                "-N", config.run.normal, 
                "-T", *sample_bams,
                "-S", *sample_names,
                *ref_ver,
                "-b", f"{output}/baf/tumor.1bed",
                "-O", f"{output}/rdr",
                "--chromosomes",
                *(chromosomes or []),
                *extra_args
            ]
        count_reads(params)
    
    # ----------------------------------------------------
    if config.run.combine_counts:
        haplotype_file = config.run.ont_haplotype_file
        if not os.path.exists(haplotype_file):
            raise RuntimeError(
                (
                    "Please specify a valid GTF file (haplotype_file) which"
                    "contains haplotype blocks for the patient"
                )
            )
        mosdepth_files = config.run.ont_mosdepth_files.split()
        for mf in mosdepth_files:
            if not os.path.exists(mf):
                raise RuntimeError(
                    (f"mosdepth file does not found with given filename: {mf}")
                )

        os.makedirs(f"{output}/bb", exist_ok=True)
        phase_file = f"{output}/phase/phased.vcf.gz"
        if not os.path.exists(phase_file):
            raise RuntimeError(
                (f"NO PHASING FILE FOUND at {phase_file}.\n")
            )

        ref_ver = ["-V", config.genotype_snps.reference_version]
        if config.count_reads.segfile != None:
            ref_ver = ["--segfile", config.count_reads.segfile] # override refversion

        params = [
            "-A",
            f"{output}/rdr",
            "-p",
            phase_file,
            "-b",
            f"{output}/baf/tumor.1bed",
            "-t",
            f"{output}/rdr/total.tsv",
            "--gtf", haplotype_file,
            "--mos_rgs", *mosdepth_files,
            *ref_ver,
            "-r",
            config.run.reference,
            "-o",
            f"{output}/bb/bulk.bb",
            *extra_args
        ]
        combine_counts(params)
    
    # ----------------------------------------------------
    if config.run.cluster_bins:
        os.makedirs(f"{output}/bbc", exist_ok=True)
        params = [
            f"{output}/bb/bulk.bb",
            "-o",
            f"{output}/bbc/bulk.seg",
            "-O",
            f"{output}/bbc/bulk.bbc",
        ]

        log(msg=f"Run cluster bins\tloc_clust={config.run.loc_clust}\n", level="STEP")

        if config.run.loc_clust:
            cluster_bins(params)
        else:
            cluster_bins_gmm(params)
    
    # ----------------------------------------------------
    if config.run.plot_bins:
        os.makedirs(f"{output}/plots", exist_ok=True)
        params = [
            f"{output}/bbc/bulk.bbc",
            "--rundir",
            f"{output}/plots",
            "--ymin",
            "0",
            "--ymax",
            "3",
        ]
        plot_bins(params)

        os.makedirs(f"{output}/plots/1d2d", exist_ok=True)
        params = [
            "-b",
            f"{output}/bbc/bulk.bbc",
            "-s",
            f"{output}/bbc/bulk.seg",
            "--outdir",
            f"{output}/plots/1d2d",
            "--centers",
            "--centromeres",
        ]
        plot_bins_1d2d(params)
    

    # ----------------------------------------------------
    if config.run.compute_cn:
        os.makedirs(f"{output}/results", exist_ok=True)
        params = [
            "-x", f"{output}/results", 
            "-i", f"{output}/bbc/bulk", 
            *extra_args
        ]
        hatchet_main(params)

    # ----------------------------------------------------
    if config.run.plot_cn:
        os.makedirs(f"{output}/summary", exist_ok=True)
        params = [
            f"{output}/results/best.bbc.ucn",
            "--rundir",
            f"{output}/summary",
        ]
        plot_cn(params)

        os.makedirs(f"{output}/summary/1d2d", exist_ok=True)
        params = [f"{output}/results/best.bbc.ucn", 
                  "--outdir", f"{output}/summary/1d2d", 
                  "--bysample", "--centromeres"]
        plot_cn_1d2d(params)
    
if __name__ == "__main__":
    main()