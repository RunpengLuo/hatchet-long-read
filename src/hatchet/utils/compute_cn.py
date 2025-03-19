import os
import sys
import shutil

import numpy as np
import pandas as pd
import kneed
import matplotlib.pyplot as plt

import hatchet.utils.Supporting as sp
from hatchet.utils.ArgParsing import parse_compute_cn_args
from hatchet.utils.compute_cn_utils import (
    get_scaling_factor_no_WGD,
    get_scaling_factor_WGD,
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
        sp.log(
            msg=f"Clusters after filtering is stored in {fseg_path} and {fbbc_path}\n",
            level="STEP",
        )
    else:
        sp.log(msg="No cluster is filtered\n", level="STEP")

    sp.log(msg="General cluster statistics\n", level="INFO")
    sp.log(msg="#ID\tSIZE(bp)\n", level="INFO")
    for cid, csize in cluster_sizes.items():
        sp.log(msg=f"#{cid}\t{csize}\n", level="INFO")

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
        for n in range(first_n, last_n):
            sp.log(
                msg=f"running diploid with n={n} and clonal clusters={str(clonal_dip)}\n",
                level="STEP",
            )
            (obj, imf_obj) = solve_func(
                n, clonal_dip, gammas_dip, cluster_sizes, args, "diploid"
            )
            diploid_sols[n] = (obj, imf_obj)
            sp.log(
                msg=f"diploid n={n} objective={obj} imf-objective={imf_obj}\n",
                level="STEP",
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
            sp.log(
                f"Cannot infer WGD clonal cluster, try increase <baf_tol> or <lb_purity>.\n",
                level="WARN",
            )
        else:
            sp.log(msg=f"Inferred tetraploid clonal cluster={zid}\n", level="INFO")
            sp.log(
                msg="Inferred tetraploid RD scaling factor gamma per sample:\n",
                level="INFO",
            )
            for sname, gamma in gammas_wgd.items():
                sp.log(msg=f"{sname}\tgamma={gamma}\n", level="INFO")
            clonal_tet = {s: (2, 2), zid: cz}
            for n in range(first_n, last_n):
                sp.log(
                    msg=f"running tetraploid with n={n} and clonal clusters={str(clonal_tet)}\n",
                    level="STEP",
                )
                (obj, imf_obj) = solve_func(
                    n, clonal_tet, gammas_wgd, cluster_sizes, args, "tetraploid"
                )
                tetraploid_sols[n] = (obj, imf_obj)
                sp.log(
                    msg=f"tetraploid n={n} objective={obj} imf-objective={imf_obj}\n",
                    level="STEP",
                )

    # final model selection between diploid and tetraploid with varying n.
    n_dip, n_tet, best_type = model_selection_final(
        diploid_sols, tetraploid_sols, out_dir, args["v"]
    )

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
    filter&merge clusters before optimization step
    TODO we can also check variance of cluster and filter sparse ones
    """
    cluster_maps = {}
    for cid in cluster_sizes.keys():
        cluster_maps[cid] = [cid]

    # check sparsity for each cluster
    # for cid in cluster_sizes.keys():
    #     bbc_ = bbc[bbc["CLUSTER"] == cid]

    chrs_per_cluster = {}
    for cid in cluster_sizes.keys():
        chrs_per_cluster[cid] = bbc.loc[bbc["CLUSTER"] == cid, "#CHR"].unique().tolist()

    # merge clusters
    sp.log(msg="filtering is not implemented yet!\n", level="INFO")

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
    solve_mode = ("both", "ilp", "cd", "cvx")[args["M"]]
    out_dir = args["x"]

    sol_dir = os.path.join(out_dir, f"sols/{problem_type}_n{n}")
    os.makedirs(sol_dir, exist_ok=True)

    instances_dir = os.path.join(sol_dir, "instances")
    os.makedirs(instances_dir, exist_ok=True)

    out_bbc = os.path.join(out_dir, f"results.{problem_type}.n{n}.bbc.ucn.tsv")
    out_seg = os.path.join(out_dir, f"results.{problem_type}.n{n}.seg.ucn.tsv")

    # bbc = pd.read_table(args["bbc"])
    seg = pd.read_table(args["seg"]).sort_values(["#ID", "SAMPLE"])

    rdr = seg.pivot(index="#ID", columns="SAMPLE", values="RD")
    baf = seg.pivot(index="#ID", columns="SAMPLE", values="BAF")

    gammas_ = pd.Series(gammas).sort_index()
    fcn = rdr * gammas_
    f_b = fcn * baf
    f_a = fcn - f_b

    cluster_ids = f_a.index.tolist()
    sample_ids = f_a.columns.tolist()

    # sort by cluster ID
    bins = pd.Series(cluster_sizes).sort_index()
    weights = 100 * bins / sum(bins)

    # check all user-defined fixed clonal states & fixed clone proportions. TODO
    copy_number_fixed = None
    purities_fixed = None

    # get user-defined regularization term / second objective
    [pname, _, _] = args["reg_term"]

    verbose = args["v"] >= 2

    store_solve_input(
        os.path.join(sol_dir, "input.tsv"),
        baf,
        rdr,
        fcn,
        f_a,
        f_b,
        weights,
    )

    # pre-process some args
    if problem_type == "diploid":
        cn_max = args["eD"]
    else:
        cn_max = args["eT"]
    max_ncns_seg = -1 if args["d"] == None else args["d"]
    max_iters = 10 if args["f"] == None else args["f"]

    best_instance = None
    imf_obj = 0.0
    if args["binwise"]:
        assert False, "binwise mode is unsupported"
    else:
        instances = solve(
            f_a=f_a,
            f_b=f_b,
            n=n,
            minprop=args["u"],
            max_ncns_seg=max_ncns_seg,
            cn_max=cn_max,
            weights=weights,
            ampdel=args["ampdel"],
            clonal=clonal,
            purities=args["purities"],
            baf=baf,
            copy_numbers_fixed=copy_number_fixed,
            purities_fixed=purities_fixed,
            reg_term=args["reg_term"],
            solver_type=args["solver"],
            max_iters=max_iters,
            n_seed=args["p"],
            n_worker=args["j"],
            random_seed=args["r"],
            timelimit=args["s"],
            instances_dir=instances_dir,
            solve_mode=solve_mode,
            verbose=verbose,
        )
        best_instance, imf_obj = model_selection_instance(
            f_a, f_b, weights, instances, pname, solve_mode, sol_dir, verbose
        )

    assert best_instance != None, f"no solution for {problem_type} and n={n}"
    [obj, cA, cB, u] = best_instance
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
    return obj, imf_obj


def execute_cpp(
    n: int,
    clonal: dict,
    gammas: dict,
    cluster_sizes: dict,
    args: dict,
    problem_type: str,
):
    sp.log(msg="cpp optimization is not implemented yet!\n", level="INFO")
    return -1, -1


def model_selection_final(diploid_sols: dict, tetraploid_sols: dict, out_dir: str, v=1):
    """
    1. select n based on elbow criterion for either WGD/no WGD
    2. then select the final solution based on principle of parsimony (lowest n)
    """

    def select_best_n(data: list, problem_type: str):
        sp.log(msg=f"running model selection for {problem_type}\n", level="INFO")
        sorted_data = sorted(data, key=lambda elem: elem[1])
        df = pd.DataFrame(data=sorted_data, columns=["n", "IMF-objective"])
        df, sol_index = model_select(
            df,
            "n",
            "IMF-objective",
            os.path.join(out_dir, f"pareto_curve.{problem_type}.png"),
            verbose=True,
        )
        df.to_csv(
            os.path.join(out_dir, f"model_selections.{problem_type}.tsv"),
            sep="\t",
            header=True,
            index=False,
        )
        return df.loc[sol_index, "n"], df.loc[sol_index, "IMF-objective"]

    if len(diploid_sols) == 0 and len(tetraploid_sols) == 0:
        sp.log(
            msg="ERROR! no solution found for either diploid or tetraploid setting!\n",
            level="ERROR",
        )
        raise ValueError(sp.error(f"final model selection error"))

    data_diploid = [[n, imf_obj] for n, (_, imf_obj) in diploid_sols.items()]
    (n2, obj2) = select_best_n(sorted(data_diploid, key=lambda a: a[0]), "diploid")
    sp.log(
        msg=f"best diploid solution is n={n2} with IMF-objective={obj2}\n", level="INFO"
    )

    data_tetraploid = [[n, imf_obj] for n, (_, imf_obj) in tetraploid_sols.items()]
    (n4, obj4) = select_best_n(
        sorted(data_tetraploid, key=lambda a: a[0]), "tetraploid"
    )
    sp.log(
        msg=f"best tetraploid solution is n={n4} with IMF-objective={obj4}\n",
        level="INFO",
    )

    # pick best solution by principle of parsimony
    final_selection = "diploid" if n2 <= n4 else "tetraploid"
    sp.log(msg=f"final selection: {final_selection}\n", level="INFO")
    return n2, n4, final_selection
