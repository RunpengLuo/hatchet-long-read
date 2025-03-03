import os
import os.path
import subprocess as pr
import gzip

import hatchet.utils.ArgParsing as ap
from hatchet.utils.Supporting import log, logArgs, error, download, checksum, to_tuple
from hatchet import config


def main(args=None):
    log(msg="# log notes\n", level="STEP")
    args = ap.parse_download_panel_arguments(args)
    logArgs(args)

    os.makedirs(args["refpaneldir"], exist_ok=True)

    # download reference panel, prepare files for liftover
    if args["refpanel"] == "1000GP_Phase3":
        # download 1000GP ref panel
        download(
            url=config.urls.onekgp,
            dirpath=args["refpaneldir"],
            sentinel_file=os.path.join("1000GP_Phase3", "1000GP_Phase3.sample"),
        )
    else:
        error(
            'Currently, only the 1000 genome panel aligned to GRCh37 without "chr" prefix is supported\n',
            raise_exception=True,
        )

    # download necessary liftover files; 1000GP in <refpanel_genome_refversion> coordinates
    # if users aligned reads to the <refver> other than <refpanel_genome_refversion>,
    # we need to liftover coordinates to the reference panel (<refver> -> <refpanel_genome_refversion>)
    # since the 1000GP panel is in <refpanel_genome_refversion> coordinates, we need to download
    # (1) <refpanel_genome_refversion> genome
    # (2) chain files for liftover via picard
    dwnld_refpanel_genome(path=args["refpaneldir"])
    dwnld_chains(dirpath=args["refpaneldir"])

    # if users aligned reads to the same reference genome as used in the reference panel, liftover isn't required, but
    # there could be different naming conventions of chromosomes, with or without the 'chr' prefix. The 1000GP reference
    # panel does NOT use 'chr' prefix, so input into shapeit also should not have this
    mk_rename_file(path=args["refpaneldir"])


def dwnld_chains(dirpath):
    """
    Download liftover files for all supported reference versions.
    """

    def mod_chain(infile, sample_chr, refpanel_index, sample_index):
        if sample_chr:
            name = infile.strip(".gz").replace("over", "chr")
        else:
            name = infile.strip(".gz").replace("over", "no_chr")

        with open(name, "w") as new:
            with gzip.open(infile, "rt") as f:
                for line in f:
                    if line.startswith("chain"):
                        line = line.split()
                        if not config.urls.refpanel_genome_chr_notation:
                            line[refpanel_index] = line[refpanel_index].replace(
                                "chr", ""
                            )
                        if not sample_chr:
                            line[sample_index] = line[sample_index].replace("chr", "")
                        new.write(" ".join(line) + "\n")
                    else:
                        new.write(line)
        return name

    supported_refvers = to_tuple(config.genotype_snps.builtin_refvers, n=None, typ=str)
    panel_refver = config.urls.refpanel_genome_refversion
    for refver in supported_refvers:
        if refver == panel_refver:
            continue
        log(msg=f"Download chain file for {refver}\n", level="STEP")

        to_panel = download(
            url=config.urls[f"refpanel_{refver}to{panel_refver}"],
            dirpath=dirpath,
            overwrite=False,
            extract=False,
        )
        from_panel = download(
            url=config.urls[f"refpanel_{panel_refver}to{refver}"],
            dirpath=dirpath,
            overwrite=False,
            extract=False,
        )

        # make all necessary chain files to convert from <refver> (w/ or w/out chr notation) 
        # to <refpanel_genome_refversion> (no chr notation),
        # and also to lift back over from <refpanel_genome_refversion> (no chr notation) 
        # to <refver> (w/ or w/out chr notation).

        # modify chr notation of <refver>To<refpanel_genome_refversion>, 
        # ref panel chr in 7th field, sample chr in 2nd field
        mod_chain(to_panel, sample_chr=True, refpanel_index=7, sample_index=2)
        mod_chain(to_panel, sample_chr=False, refpanel_index=7, sample_index=2)

        # modify chr notation of <refpanel_genome_refversion>To<refver>, 
        # ref panel chr in 2nd field, sample chr in 7th field
        mod_chain(from_panel, sample_chr=True, refpanel_index=2, sample_index=7)
        mod_chain(from_panel, sample_chr=False, refpanel_index=2, sample_index=7)
    return


def dwnld_refpanel_genome(path):
    """
    Download <refpanel_genome_refversion> reference with no-chr notation, used in 1000 genome panel.
    """
    panel_refver = config.urls.refpanel_genome_refversion
    ref_file = os.path.join(path, f"{panel_refver}_no_chr.fa")
    if not os.path.isfile(ref_file):
        usr_ref_file = config.paths.reference
        if (
            os.path.isfile(usr_ref_file)
            and checksum(usr_ref_file) == config.urls.refpanel_genome_checksum
        ):
            tmp_ref_file = usr_ref_file
        else:
            tmp_ref_file = download(
                config.urls.refpanel_genome, dirpath=path, extract=False
            )

        if tmp_ref_file.endswith(".gz"):
            tmp_fd = gzip.open(tmp_ref_file, "rt")
        else:
            tmp_fd = open(tmp_ref_file, "r")
        with open(ref_file, "w") as ref_fd:
            for line in tmp_fd:
                if line.startswith(">"):
                    ref_fd.write(line.replace("chr", ""))
                else:
                    ref_fd.write(line)
            ref_fd.close()
        tmp_fd.close()

    dict_file = os.path.join(path, f"{panel_refver}_no_chr.dict")
    if not os.path.isfile(dict_file):
        samtools = os.path.join(config.paths.samtools, "samtools")
        cmd = f"{samtools} dict {ref_file} > {dict_file}"
        errname = os.path.join(path, "samtools.log")
        with open(errname, "w") as err:
            run = pr.run(
                cmd,
                stdout=err,
                stderr=err,
                shell=True,
                universal_newlines=True,
            )
            err.close()
        if run.returncode != 0:
            raise ValueError(
                error(
                    f"Samtools dict creation failed, please check errors in {errname}!"
                )
            )
        else:
            os.remove(errname)
    return ref_file, dict_file


def mk_rename_file(path):
    """
    makes rename_chrs1.txt for removing "chr", rename_chrs2.txt for adding "chr"
    """

    names = [
        os.path.join(path, "rename_chrs1.txt"),
        os.path.join(path, "rename_chrs2.txt"),
    ]
    fd1 = open(names[0], "w")
    fd2 = open(names[1], "w")
    for j in range(1, 23):
        fd1.write(f"chr{j} {j}\n")
        fd2.write(f"{j} chr{j}\n")
    fd1.close()
    fd2.close()
    return names


if __name__ == "__main__":
    main()
