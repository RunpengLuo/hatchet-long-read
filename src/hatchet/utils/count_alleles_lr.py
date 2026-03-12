import sys
import os
from multiprocessing import Pool
import os.path
import shlex
import subprocess as sp
from scipy.stats import beta
import tempfile
import pandas as pd

from hatchet.utils.ArgParsing import parse_count_alleles_arguments
from hatchet.utils.Supporting import log, logArgs, error, close
from hatchet.utils.multiprocessing import Worker

def main(args=None):
    log(msg="count_alleles_lr test version\n", level="STEP")
    log(msg="# Parsing and checking input arguments\n", level="STEP")
    args = parse_count_alleles_arguments(args)
    logArgs(args, 80)


    chromosomes=args["chromosomes"]
    # list of filtered phased/unphased Hete SNP VCF file per chrom, output by genotype_snps_lr.py
    snplist: dict = args["snps"]
    mincov=args["mincov"],
    maxcov=args["maxcov"],

    processes = args["j"]
    
    out_normal = args["outputNormal"]
    out_tumor = args["outputTumors"]
    outdir = args["outputSnps"]


    with tempfile.TemporaryDirectory(dir=args['outputSnps']) as tmpdirname:
        # compute normal1.bed
        selectHets_params = [
            (args["normal"][1],
            ch, 
            snplist[ch],
            mincov,
            maxcov,
            args["bcftools"],
            tmpdirname
            )
            for ch in chromosomes 
        ]
        rets = []
        try:
            with Pool(processes) as p:
                rets = p.map(run_selectHets, selectHets_params)
        except Exception as e:
            log(msg=f"ERROR! selectHets raise exception: {e}\n",level="ERROR")
            p.terminate()
            raise ValueError()
        finally:
            p.join()
            log(msg="All selectHets finished\n", level="STEP")
        

    # compute tumor1.bed

def selectHets(sample_name: str, ch: str, vcf_file: str, min_dp: int, max_dp: int, bcftools: str, 
               tmpdir: str):
    """
    vcf_files: list of single-chromosome position-sorted VCF file for one sample.
    """
    bed_file = f"{tmpdir}/{sample_name}_{ch}.bed"
    tmp_file = f"{tmpdir}/TMP_{sample_name}_{ch}.tsv"
    try:
        query_cmd = [
            bcftools,
            "query",
            "-f", "\'%CHROM\\t%POS\\t" + sample_name + "\\t[%AD\{0\}]\\t[%AD\{1\}]\\t%REF\\t%ALT\{0\}\\n\'",
            "-i", f"\'SUM(FMT/AD)>={min_dp} & SUM(FMT/AD)<={max_dp}\'",
            vcf_file
        ]
        err_fd = open(f"{tmpdir}/run_bcftools_{sample_name}_{ch}.err.log", "w")
        out_fd = open(bed_file, "w")
        ret = sp.run(query_cmd, stdout=out_fd, stderr=err_fd)
        err_fd.close()
        out_fd.close()
        ret.check_returncode()

        df = pd.read_csv(bed_file, sep='\t', header=False, usecols=range(2))
        df.to_csv(tmp_file, index=False, sep='\t', header=False)
    except Exception as e:
        log(f"ERROR! BCFtools query exception {e}", level="ERROR")
        raise e
    finally:
        log(f"Done BCFtools query on sample {sample_name} chromosome {ch}\n", level="STEP")
        return sample_name, ch, bed_file, tmp_file

def run_selectHets(params):
    return selectHets(*params)





if __name__ == "__main__":
    main()