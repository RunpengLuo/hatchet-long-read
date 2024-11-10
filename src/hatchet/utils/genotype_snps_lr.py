import os
import subprocess
import gzip
import numpy as np
import pandas as pd
from hatchet.utils.Supporting import log, logArgs
from hatchet.utils.ArgParsing import parse_genotype_snps_arguments

def main(args=None):
    """
    This functions takes a phased VCF file, 
    """
    log(msg="genotype_snps_lr test version\n", level="STEP")
    log(
        msg="# Parsing the input arguments, checking the consistency of given files, \
            and extracting required information\n",
        level="STEP",
    )
    args = parse_genotype_snps_arguments(args)
    logArgs(args, 80)

    output_dir = args["outputsnps"]
    mindp = args["mincov"]
    maxdp = args["maxcov"]

    for chrom in args["chromosomes"]:
        log(
            msg=f"Extracting chromosome {chrom} from genotype file...\n",
            level="STEP",
        )

        # call bcftools to extract chromosome
        output_vcf = f"{output_dir}/{chrom}.vcf.gz"
        if os.path.isfile(output_vcf):
            log(
                msg=f"skip, file exists\n",
                level="STEP",
            )
            continue
        command = [
            args["bcftools"],
            "view",
            "-r",
            chrom,
            "-i",
            " & ".join(["FILTER=\"PASS\"", 
                        "strlen(REF)=1", 
                        "strlen(ALT)=1",
                        f"SUM(FORMAT/AD)>={mindp}", f"SUM(FORMAT/AD)<={maxdp}",
                        "N_ALT=1", "(FORMAT/GT[0]==\"1|0\" | FORMAT/GT[0]==\"0|1\")"]),
            args["snps"],
            "-Oz",
            "-o",
            output_vcf,
        ]
        process = subprocess.Popen(command)
        process.wait()  # Wait for the process to complete
    log(msg="genotype-snps-lr completed successfully\n", level="STEP")
    return

if __name__ == "__main__":
    main()