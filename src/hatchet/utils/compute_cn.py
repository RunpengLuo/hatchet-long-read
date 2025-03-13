import os
import sys
import shutil

import numpy as np
import pandas as pd

import hatchet.utils.Supporting as sp
from hatchet.utils.ArgParsing import parse_compute_cn_args
from hatchet.utils.compute_cn_utils import (
    get_scaling_factor_no_WGD,
    get_scaling_factor_WGD
)
from hatchet.utils.solve import *
from hatchet.utils.solve.utils import *
from hatchet.utils.solve.segmentation import segmentation

def main(args=None):
    sp.log(msg="# Parsing and checking input arguments\n", level="STEP")
    args = parse_compute_cn_args(args)
    sp.logArgs(args, 80)

    out_dir = args["x"]
    os.makedirs(out_dir, exist_ok=True)

    bbc = pd.read_table(args["bbc"])
    seg = pd.read_table(args["seg"])

    solver = args["solver"]
    solver_path = args["solver_path"]
    if not solver_available(solver):
        raise ValueError(sp.error(f"{solver} is not supported/available"))

    solve_func = execute_cpp if solver == "cpp" else execute_python

    samples = sorted(bbc["SAMPLE"].unique().tolist())
    clusters = sorted(seg["#ID"].unique().tolist())

    if args["purities"] != None:
        if not all(sample in samples for sample in args["purities"].keys()):
            raise ValueError(sp.error(f"purities contains sample name error"))

    cluster_sizes = {}
    for cid in clusters:
        seg_rows = seg.loc[seg["#ID"] == cid]
        if len(seg_rows["#BINS"].unique()) != 1:
            raise ValueError(
                f"Bin sizes for cluster {cid} across tumor samples are not identical!"
            )
        bbc_rows = bbc.loc[(bbc["CLUSTER"] == cid) & (bbc["SAMPLE"] == samples[0])]
        cluster_sizes[cid] = bbc_rows.apply(
            func=lambda r: r["END"] - r["START"], axis=1
        ).sum()

    fbbc, fseg, fclusters = filtering(
        bbc=bbc,
        seg=seg,
        samples=samples,
        cluster_sizes=cluster_sizes,
        ts=args["ts"],
        tc=args["tc"],
        mB=args["mB"],
        mR=args["mR"],
        v=args["v"],
    )

    if len(fseg) != len(seg):
        fseg_path = os.path.join(out_dir, "bulk.fil.seg")
        fseg.to_csv(fseg_path, header=True, index=False, sep="\t")
        args["seg"] = fseg_path

        fbbc_path = os.path.join(out_dir, "bulk.fil.bbc")
        fbbc.to_csv(fbbc_path, header=True, index=False, sep="\t")
        args["bbc"] = fbbc_path
        clusters = fclusters
        sp.log(msg=f"Clusters after filtering is stored in {args["seg"]} and {args["bbv"]}\n", level="STEP")
    else:
        sp.log(msg="No cluster is filtered\n", level="STEP")
    
    sp.log(msg="General cluster statistics\n", level="INFO")
    sp.log(msg="#ID\tSIZE(bp)\n")
    for cid, csize in cluster_sizes.items():
        sp.log(msg=f"{cid}\t{csize}\n", level="INFO")

    s, gammas_dip = get_scaling_factor_no_WGD(
        seg=fseg,
        samples=samples,
        cluster_sizes=cluster_sizes,
        tol_baf=args["td"],
        v=args["v"],
    )
    sp.log(msg=f"Inferred diploid netural cluster={s}\n", level="INFO")
    sp.log(msg="Inferred diploid RD scaling factor gamma per sample:\n", level="INFO")
    for sname, gamma in gammas_dip.items():
        sp.log(msg=f"{sname}\tgamma={gamma}\n", level="INFO")

    # optimization step
    first_n, last_n = args["ln"], args["un"] + 1

    diploid_sols = {}
    if args["diploid"]:
        clonal_dip = {s: (1, 1)}
        sp.log(msg=f"running diploid with clonal clusters={str(clonal_dip)}\n", level="STEP")
        for n in range(first_n, last_n):
            diploid_sols[n] = solve_func(
                n, clonal_dip, gammas_dip, cluster_sizes, args, "diploid"
            )

    tetraploid_sols = {}
    if args["tetraploid"]:
        zid, cz, gammas_wgd = get_scaling_factor_WGD(
            seg=fseg,
            sid=s,
            cluster_sizes=cluster_sizes,
            max_cn=args["eT"],
            lb_purity=args["mP"],
            rd_tol=args["tR"],
            baf_tol=args["tB"],
            v=args["v"],
        )
        if zid == None:
            sp.log(f"Cannot infer WGD clonal cluster, try increase <baf_tol> or <lb_purity>.\n", level="WARN")
        else:
            sp.log(msg=f"Inferred tetraploid clonal cluster={zid}\n", level="INFO")
            sp.log(msg="Inferred tetraploid RD scaling factor gamma per sample:\n", level="INFO")
            for sname, gamma in gammas_wgd.items():
                sp.log(msg=f"{sname}\tgamma={gamma}\n", level="INFO")
            clonal_tet = {s: (2, 2), zid: cz}
            sp.log(msg=f"running tetraploid with clonal clusters={str(clonal_tet)}\n", level="STEP")
            for n in range(first_n, last_n):
                tetraploid_sols[n] = solve_func(
                    n, clonal_tet, gammas_wgd, cluster_sizes, args, "tetraploid"
                )

    # final model selection between diploid and tetraploid with varying n.
    n_dip, n_tet, best_type = model_selection(diploid_sols, tetraploid_sols, args["v"])

    # save model selected result here
    if n_dip > 0:
        shutil.copy2(
            os.path.join(out_dir, f"results.diploid.n{n_dip}.bbc.ucn.tsv"),
            os.path.join(out_dir, "chosen.diploid.bbc.ucn"),
        )
        shutil.copy2(
            os.path.join(out_dir, f"results.diploid.n{n_dip}.seg.ucn.tsv"),
            os.path.join(out_dir, "chosen.diploid.seg.ucn"),
        )

    if n_tet > 0:
        shutil.copy2(
            os.path.join(out_dir, f"results.tetraploid.n{n_tet}.bbc.ucn.tsv"),
            os.path.join(out_dir, "chosen.tetraploid.bbc.ucn"),
        )
        shutil.copy2(
            os.path.join(out_dir, f"results.tetraploid.n{n_tet}.seg.ucn.tsv"),
            os.path.join(out_dir, "chosen.tetraploid.seg.ucn"),
        )

    if best_type != None:
        shutil.copy2(
            os.path.join(out_dir, f"chosen.{best_type}.bbc.ucn"),
            os.path.join(out_dir, "best.bbc.ucn"),
        )
        shutil.copy2(
            os.path.join(out_dir, f"chosen.{best_type}.seg.ucn"),
            os.path.join(out_dir, "best.seg.ucn"),
        )

    return


def filtering(
    bbc: pd.DataFrame,
    seg: pd.DataFrame,
    samples: list,
    cluster_sizes: dict,
    ts: float,
    tc: float,
    mB: float,
    mR: float,
    v=1,
):
    """
    filter clusters before optimization step
    TODO we can also check variance of cluster and filter sparse ones
    """
    chrs_per_cluster = {}
    for cid in cluster_sizes.keys():
        chrs_per_cluster[cid] = bbc.loc[bbc["CLUSTER"] == cid, "#CHR"].unique().tolist()

    # merge clusters
    sp.log(msg="filtering is not implemented yet!", level="INFO")

    # for cid, csize in sorted(cluster_sizes.items(), key=lambda tp: tp[1], reverse=True):
    return bbc, seg, cluster_sizes


def execute_python(
    n: int,
    clonal: dict,
    gammas: dict,
    cluster_sizes: dict,
    args: dict,
    problem_type: str,
):
    """
    execute optimization prog
    return:
    obj
    """
    solver_mode = ("both", "ilp", "cd")[args["M"]]
    out_dir = args["x"]
    sol_dir = os.path.join(out_dir, f"sols/{problem_type}_n{n}")
    out_bbc = os.path.join(out_dir, f"results.{problem_type}.n{n}.bbc.ucn.tsv")
    out_seg = os.path.join(out_dir, f"results.{problem_type}.n{n}.seg.ucn.tsv")

    # bbc = pd.read_table(args["bbc"])
    seg = pd.read_table(args["seg"]).sort_values(["#ID", "SAMPLE"])
    sample_ids = np.sort(seg["SAMPLE"].unique())

    rdr = seg.pivot(index="#ID", columns="SAMPLE", values="RD")
    baf = seg.pivot(index="#ID", columns="SAMPLE", values="BAF")

    gammas_ = pd.Series(gammas).sort_index()
    fcn = rdr * gammas_
    f_b = fcn * baf
    f_a = fcn - f_b

    bins = pd.Series(cluster_sizes)
    weights = 100 * bins / sum(bins)

    # check all user-defined fixed clonal states & fixed clone proportions. TODO
    copy_number_fixed = None
    purities_fixed = None

    # check user-defined regularization terms
    # [pname, num_steps, step_size] = ["RAW", 0, 0.0]
    [pname, num_steps, step_size] = args["reg_term"]

    store_solve_input(
        os.path.join(sol_dir, "input.tsv"),
        sample_ids,
        cluster_ids,
        baf,
        rdr,
        fcn,
        f_a,
        f_b,
        weights,
    )

    best_instance = None
    if args["bin_wise"]:
        assert False, "bin_wise mode is unsupported"
    else:
        instances = {}
        os.makedirs(os.path.join(sol_dir, "instances"), exist_ok=True)
        for i0 in range(0, num_steps + 1):
            param = step_size * i0
            instance_dir = os.path.join(sol_dir, f"instances/solve_{pname}_{param}")
            os.makedirs(instance_dir, exist_ok=True)
            instances[param] = solve_instance(
                f_a=f_a,
                f_b=f_b,
                n=n,
                mu=args["u"],
                d=-1 if args["d"] == None else args["d"],
                cn_max=args["e"],
                weights=weights,
                ampdel=args["ampdel"],
                clonal=clonal,
                purities=args["purities"],
                baf=baf,
                copy_numbers_fixed=copy_number_fixed,
                purities_fixed=purities_fixed,
                penalty_param=[pname, param],
                solver=args["solver"],
                max_iters=10 if args["f"] == None else args["f"],
                n_seed=args["p"],
                n_worker=args["j"],
                random_seed=args["r"],
                timelimit=args["s"],
                instance_dir=instance_dir,
                solve_mode=solver_mode,
            )
        best_instance = model_selection_instance(
            f_a, f_b, weights, instances, pname, sol_dir
        )

    assert best_instance != None, f"no solution for {problem_type} and n={n}"
    [obj, cA, cB, u, cluster_ids, sample_ids] = best_instance
    segmentation(
        cA,
        cB,
        u,
        cluster_ids,
        sample_ids,
        bbc_file=args["bbc"],
        bbc_out_file=out_bbc,
        seg_out_file=out_seg,
    )
    return obj


def execute_cpp(
    n: int,
    clonal: dict,
    gammas: dict,
    cluster_sizes: dict,
    args: dict,
    problem_type: str,
):
    sp.log(msg="cpp optimization is not implemented yet!", level="INFO")
    return -1

def model_selection(diploid_sols: dict, tetraploid_sols: dict, v=1):
    sp.log(msg="model selection is not implemented yet!", level="INFO")
    # TODO
    return 2, 2, "diploid"
