import os.path
import subprocess as pr
from multiprocessing import Pool

import hatchet.utils.ArgParsing as ap
from hatchet.utils.Supporting import log, logArgs, error, ensure, run

def main(args=None):
    log(msg="# phase_snps_lr version\n", level="STEP")
    log(msg="# Parsing and checking input arguments\n", level="STEP")
    args = ap.parse_phase_snps_lr_arguments(args)
    logArgs(args, 80)

    bcftools = args["bcftools"]
    whatshap = args["whatshap"]
    ref = args["refgenome"]
    normal_bam = args["normal"]
    snplists = args["snps"]
    outdir = args["outdir"]
    nproc = int(args["j"])

    os.makedirs(outdir, exist_ok=True)

    chromosomes = []
    for chro in args["chromosomes"]:
        if chro.endswith("X") or chro.endswith("Y"):
            log(
                msg=f"Skipping chromosome {chro} (because it ends with X or Y)\n",
                level="WARN",
            )
        else:
            chromosomes.append(chro)

    whatshap_params = [
        (
            whatshap,
            ref,
            snplists[ch],
            normal_bam,
            ch,
            outdir
        )
        for ch in chromosomes
    ]
    rets = []
    with Pool(nproc) as p:
        rets = p.map(run_whatshap, whatshap_params)
    if any(ret != 0 for ret in rets):
        log(msg="Abort\n", level="ERROR")
        raise ValueError()
    
    ret = concat_vcfs(bcftools, outdir, chromosomes)
    if ret != 0:
        log(msg="Abort\n", level="ERROR")
        raise ValueError()
    # TODO print phasing stat?
    log(msg=f"phase-snps-lr completed\n", level="STEP")
    return

def _run_whatshap(whatshap: str, ref_file: str, snp_file: str, bam_file: str, ch: str, outdir: str):
    retcode = 0
    try:
        out_file = os.path.join(outdir, f"{ch}_phased.vcf.gz")
        if os.path.exists(out_file):
            log(
                f"Skipping whatshap on chromosome {ch}, {out_file} exists\n",
                level="STEP",
            )
            return retcode
        
        log(f"Run whatshap on chromosome {ch}\n", level="STEP")
        cmd_whatshap = [
            whatshap,
            "phase",
            "--output", out_file,
            "--reference", ref_file,
            "--chromosome", ch,
            "--ignore-read-groups", 
            snp_file,
            bam_file
        ]

        errname = os.path.join(outdir, f"{ch}_whatshap.err.log")
        outname = os.path.join(outdir, f"{ch}_whatshap.out.log")
        with open(errname, "w") as err_fd, open(outname, "w") as out_fd:
            ret = pr.run(cmd_whatshap, stdout=out_fd, stderr=err_fd)
            err_fd.close(), out_fd.close()
        ret.check_returncode()
        os.remove(errname)
        os.remove(outname)
        log(f"Done whatshap on chromosome {ch}\n", level="STEP")
    except Exception as e:
        log(f"ERROR! whatshap {ch} exception {e}\n", level="ERROR")
        retcode = 1
    return retcode

def run_whatshap(params):
    return _run_whatshap(*params)

def concat_vcfs(bcftools: str, outdir: str, chromosomes: list):
    retcode = 0
    try:
        out_file = os.path.join(outdir, "phased.vcf.gz")
        if os.path.exists(out_file):
            log(
                f"Skipping VCF concat, {out_file} exists\n",
                level="STEP",
            )
            return retcode
        infiles = [os.path.join(outdir, f"{ch}_phased.vcf.gz") for ch in chromosomes]
        if not all(os.path.exists(infile) for infile in infiles):
            log(f"ERROR! missing phased.vcf.gz for specified chromosome(s)\n", level="ERROR")
            retcode = 1
            return retcode
        cmd_bcftools = [
            bcftools,
            "concat",
            "--output-type", "z",
            "--output", out_file
        ] + infiles
        errname = os.path.join(outdir, f"concat.err.log")
        outname = os.path.join(outdir, f"concat.out.log")
        with open(errname, "w") as err_fd, open(outname, "w") as out_fd:
            ret = pr.run(cmd_bcftools, stdout=out_fd, stderr=err_fd)
            err_fd.close(), out_fd.close()
        ret.check_returncode()
        [os.remove(i) for i in infiles]
        os.remove(errname)
        os.remove(outname)
        log("Done bcftools concat, intermediate files removed\n", level="STEP")
    except Exception as e:
        log(f"ERROR! bcftools concat exception {e}\n", level="ERROR")
        retcode = 1
    return retcode
