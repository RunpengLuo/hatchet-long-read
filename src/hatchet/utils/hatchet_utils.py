import os
import gzip
import subprocess

import pandas as pd
import numpy as np

from hatchet.utils.Supporting import (
    ensure,
    log,
    error,
    which,
    url_exists,
    bcolors,
    numericOrder,
)

def sort_chroms(chromosomes: list):
    assert len(chromosomes) != 0
    ch = "chr" if str(chromosomes[0]).startswith("chr") else ""
    chr2ord = {}
    for i in range(1,23):
        chr2ord[f"{ch}{i}"] = i
    chr2ord[f"{ch}X"] = 23
    chr2ord[f"{ch}Y"] = 24
    if any(x not in chr2ord for x in chromosomes):
        return chromosomes
    return sorted(chromosomes, key=lambda x: chr2ord[x])

"""
    check if all chromosomes with prefix chr or not
    return True if all prefix with chr
    return False if all prefix without chr, or the input is empty
"""
def use_chr_prefix(chromosomes: list):
    # Check that chr notation is consistent across chromosomes
    using_chr = [a.startswith("chr") for a in chromosomes]
    if any(using_chr):
        ensure(
            all(using_chr),
            "Some chromosomes use chr notation while others do not.",
        )
        return True
    return False

def has_header(file: str, header_prefix=["#CHR", "CHR", "Chr", "CHROMOSOME", "#ID"]):
    """
    check if the file has header.
    """
    fd = gzip.open(file, "rt") if file.endswith(".gz") else open(file, "r")
    first_line = fd.readline()
    fd.close()
    for hp in header_prefix:
        if first_line.startswith(hp):
            return True
    return False

def load_seg_file(seg_file: str, convert_1based=False, additional_columns=[]):
    """
    load segment file in BED format with 0-indexed and left-close right-open format.
    If convert_1based is set, 1-indexed and left-close right-close format will be used.
    CHR column will always have chr-prefix.
    CHR\tSTART\tEND\t...
    """
    num_columns = 3 + len(additional_columns)
    if has_header(seg_file):
        seg_df = pd.read_csv(seg_file, sep="\t")
    else:
        seg_df = pd.read_csv(seg_file, sep="\t", header=None,
                             usecols=range(num_columns),
                             names=["CHR", "START", "END"] + additional_columns)

    seg_df["CHR"] = seg_df["CHR"].astype(str)
    if not use_chr_prefix(seg_df["CHR"].tolist()):
        seg_df["CHR"] = seg_df["CHR"].apply(lambda s: "chr" + s)
    
    if convert_1based:
        seg_df.loc[:, "START"] = seg_df.loc[:, "START"] + 1

    chs = seg_df["CHR"].unique().tolist()
    return seg_df, chs

def load_cent_file(cent_file: str):
    """
    load standard centromeres file using 1-based index and left-right-closed format.
    CHR column will always have chr-prefix.
    """
    centromeres = pd.read_table(
        cent_file,
        header=None,
        names=["CHR", "START", "END", "NAME", "gieStain"],
    )
    assert (centromeres.gieStain == "acen").all()

    if not use_chr_prefix(centromeres["CHR"].tolist()):
        centromeres["CHR"] = centromeres["CHR"].apply(lambda s: "chr" + s)

    return centromeres

def get_centromeres(filename: str, use_chr: bool):
    """
    load as segment file if it has BED suffix.
        - two segments per chromosome denotes regions before/after centromeres.
    otherwise, load as centromere file if it has TXT suffix.
        - two segments per chromosome denotes centromere p and q.
    """
    assert filename.endswith("bed") or filename.endswith("txt")
    chr2centro = {}
    if filename.endswith("bed"):
        centromeres, chs = load_seg_file(filename, convert_1based=True)
        for ch in chs:
            cent_df_ = centromeres[centromeres.CHR == ch]
            assert len(cent_df_) == 2, f">2 region detected in {filename}."
            chkey = ch if use_chr else ch[3:]
            # Each centromere should consist of 2 adjacent segments
            chr2centro[chkey] = cent_df_.END.min(), cent_df_.START.max()
    else:
        centromeres = load_cent_file(filename)
        chs = centromeres.CHR.unique().tolist()
        for ch in chs:
            cent_df_ = centromeres[centromeres.CHR == ch]
            assert len(cent_df_) == 2, f">2 region detected in {filename}."
            chkey = ch if use_chr else ch[3:]
            # Each centromere should consist of 2 adjacent segments
            chr2centro[chkey] = cent_df_.START.min(), cent_df_.END.max()
    return chr2centro
