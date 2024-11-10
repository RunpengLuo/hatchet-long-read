import os
import subprocess
import gzip
import numpy as np
import pandas as pd
from hatchet.utils.Supporting import log, logArgs
from hatchet.utils.ArgParsing import parse_genotype_snps_arguments

def main(args):
    """
    This functions extracts each phased genotype from VCF
    file and saves it to phased.vcf.gz in the output directory.
    """
    log(msg='phase_snps_lr test version\n', level='STEP')
    log(
        msg=(
            '# Parsing the input arguments, checking the consistency of given files, and extracting required ',
            'information\n',
        ),
        level='STEP',
    )
    # FIXME currently this reuse parser from genotype_snps
    args = parse_genotype_snps_arguments(args)
    logArgs(args, 80)

    # call bcftools to extract 0|1 and 1|0 snps
    output_dir = args['outputsnps']
    output_vcf = f'{output_dir}/phased.vcf.gz'
    mindp = args['mincov']
    maxdp = args['maxcov']


    if not os.path.isfile(output_vcf):
        command = [
            args['bcftools'],
            'view',
            '-i',
            " & ".join(["FILTER=\"PASS\"", 
                        "strlen(REF)=1", 
                        "strlen(ALT)=1",
                        f"SUM(FORMAT/AD)>={mindp}", f"SUM(FORMAT/AD)<={maxdp}",
                        "N_ALT=1", "(FORMAT/GT[0]==\"1|0\" | FORMAT/GT[0]==\"0|1\")"]),
            args['snps'],
            '-Oz',
            '-o',
            output_vcf,
        ]
        process = subprocess.Popen(command)
        process.wait()  # Wait for the process to complete
    else:
        log(
            msg=f"skip, file exists\n",
            level="STEP",
        )
    log(msg='phase-snps-lr completed successfully\n', level='STEP')
    return

if __name__ == '__main__':
    main()
