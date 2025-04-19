import os
import sys
import shutil

import numpy as np
import pandas as pd

import hatchet.utils.Supporting as sp
from hatchet.utils.ArgParsing import parse_compute_cn_args
from hatchet.utils.compute_cn_utils import (
    locate_balanced_clusters,
    pairwise_merge,
    get_scaling_factor,
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

    for cid in clusters:
        seg_rows = seg.loc[seg["#ID"] == cid]
        if len(seg_rows["#BINS"].unique()) != 1:
            raise ValueError(
                f"Bin sizes for cluster {cid} across tumor samples are not identical!"
            )

    cluster_refined = False
    good_clusters = filtering(
        bbc=bbc,
        seg=seg,
        samples=samples,
        clusters=clusters,
        fstd=args["fstd"],
        v=args["v"],
    )
    if len(good_clusters) != len(clusters):
        cluster_refined |= True
        seg = seg[seg["#ID"].isin(good_clusters)]
        bbc = bbc[bbc["CLUSTER"].isin(good_clusters)]

    balanced_s, unbalanced_z = locate_balanced_clusters(seg, args["mB"])
    if len(balanced_s) <= 0:
        raise ValueError(f"no balanced cluster was found, try increase <mB>")
    if args["merge"]:
        bbc, seg, _balanced_s = pairwise_merge(
            samples, seg, bbc, balanced_s, args["mR"], args["v"]
        )
        cluster_refined |= len(_balanced_s) != len(balanced_s)
        balanced_s = _balanced_s

    clusters = balanced_s + unbalanced_z
    if cluster_refined:
        fseg_path = os.path.join(out_dir, "bulk.good.seg")
        seg.to_csv(fseg_path, header=True, index=False, sep="\t")
        args["seg"] = fseg_path
        fbbc_path = os.path.join(out_dir, "bulk.good.bbc")
        bbc.to_csv(fbbc_path, header=True, index=False, sep="\t")
        args["bbc"] = fbbc_path
        sp.log(
            msg=f"Clusters after filtering&merging is stored in {fseg_path} and {fbbc_path}\n",
            level="STEP",
        )
    else:
        sp.log(msg="No cluster is filtered&merged\n", level="STEP")

    sp.log(msg="General cluster statistics\n", level="INFO")
    sp.log(msg="#ID\tSIZE(bp)\n", level="INFO")
    cluster_sizes = {}
    for cid in clusters:
        bbc_rows = bbc.loc[(bbc["CLUSTER"] == cid) & (bbc["SAMPLE"] == samples[0])]
        cluster_sizes[cid] = bbc_rows.apply(
            func=lambda r: r["END"] - r["START"], axis=1
        ).sum()
        sp.log(msg=f"#{cid}\t{cluster_sizes[cid]}\n", level="INFO")

    # compute RD scaling factor
    ret_scaling = get_scaling_factor(
        samples,
        seg,
        bbc,
        balanced_s,
        unbalanced_z,
        args["tR"],
        args["tB"],
        args["eD"],
        args["eT"],
    )
    s0, pair_noWGD, gammas_noWGD, pair_WGD, gammas_WGD = ret_scaling

    # optimization step
    first_n, last_n = args["ln"], args["un"] + 1

    diploid_sols = {}
    if args["diploid"]:
        if len(gammas_noWGD) != len(samples):
            sp.log(
                f"Failed to infer scaling factor for noWGD, try increase <baf_tol> or <rd_tol>.\n",
                level="WARN",
            )
        else:
            sp.log(msg=f"Inferred (1,1) balanced cluster={s0}\n", level="INFO")
            clonal_dip = {s0: (1, 1)}
            if pair_noWGD != None:
                (s, z, (sa, sb), (za, zb)) = pair_noWGD
                sp.log(
                    msg=f"Inferred clonal pair: {s}:({sa},{sb}), {z}:({za},{zb})\n",
                    level="INFO",
                )
                if args["fixc_nowgd"]:
                    clonal_dip = {s: (sa, sb), z: (za, zb)}
                    sp.log(msg=f"fixed clonal pair\n", level="INFO")
            sp.log(
                msg="Inferred diploid RD scaling factor gamma per sample:\n",
                level="INFO",
            )
            for sample, gamma in gammas_noWGD.items():
                sp.log(msg=f"{sample}\tgamma={gamma}\n", level="INFO")

            for n in range(first_n, last_n):
                sp.log(
                    msg=f"running diploid with n={n} and clonal clusters={str(clonal_dip)}\n",
                    level="STEP",
                )
                (obj, imf_obj) = solve_func(
                    n, clonal_dip, gammas_noWGD, cluster_sizes, args, "diploid"
                )
                diploid_sols[n] = (obj, imf_obj)
                sp.log(
                    msg=f"diploid n={n} objective={obj} imf-objective={imf_obj}\n",
                    level="STEP",
                )

    tetraploid_sols = {}
    if args["tetraploid"]:
        if len(gammas_WGD) != len(samples) or pair_WGD == None:
            sp.log(
                f"Failed to infer scaling factor for WGD, try increase <baf_tol> or <rd_tol>.\n",
                level="WARN",
            )
        else:
            (s, z, (sa, sb), (za, zb)) = pair_WGD
            sp.log(
                msg=f"Inferred clonal pair: {s}:({sa},{sb}), {z}:({za},{zb})\n",
                level="INFO",
            )
            sp.log(
                msg="Inferred tetraploid RD scaling factor gamma per sample:\n",
                level="INFO",
            )
            for sample, gamma in gammas_WGD.items():
                sp.log(msg=f"{sample}\tgamma={gamma}\n", level="INFO")

            clonal_tet = {s: (sa, sb), z: (za, zb)}
            for n in range(first_n, last_n):
                sp.log(
                    msg=f"running tetraploid with n={n} and clonal clusters={str(clonal_tet)}\n",
                    level="STEP",
                )
                (obj, imf_obj) = solve_func(
                    n, clonal_tet, gammas_WGD, cluster_sizes, args, "tetraploid"
                )
                tetraploid_sols[n] = (obj, imf_obj)
                sp.log(
                    msg=f"tetraploid n={n} objective={obj} imf-objective={imf_obj}\n",
                    level="STEP",
                )

    if len(diploid_sols) == 0 and len(tetraploid_sols) == 0:
        raise ValueError("No solutions found for either noWGD or WGD case, exit..\n")

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
    clusters: list,
    fstd=2.0,
    v=1,
):
    """
    filter&merge clusters before optimization step
    1. compute per-sample per-cluster variance SCV,
    2. compute per-sample MV and STDV
    3. filter a cluster if it has |SCV - MV| >= 2 * STDV for all samples.

    Returns:
    1. list of remaining cluster IDs
    """
    sp.log(f"preprocessing, filtering clusters\n", level="STEP")

    var_rd_matrix = np.zeros((len(clusters), len(samples)), dtype=np.float64)
    var_baf_matrix = np.zeros((len(clusters), len(samples)), dtype=np.float64)

    for i, cluster in enumerate(clusters):
        for j, sample in enumerate(samples):
            bbc_ = bbc[(bbc["SAMPLE"] == sample) & (bbc["CLUSTER"] == cluster)]
            var_rd_matrix[i, j] = np.linalg.norm(
                bbc_["RD"] - np.mean(bbc_["RD"]), 2
            ) / len(bbc_)
            var_baf_matrix[i, j] = np.linalg.norm(
                bbc_["BAF"] - np.mean(bbc_["BAF"]), 2
            ) / len(bbc_)
    mv_rd = np.mean(var_rd_matrix, axis=0)
    stdv_rd = np.std(var_rd_matrix, axis=0, ddof=1)
    mv_baf = np.mean(var_baf_matrix, axis=0)
    stdv_baf = np.std(var_baf_matrix, axis=0, ddof=1)
    if v >= 1:
        for j, sample in enumerate(samples):
            lb_rd = mv_rd[j] - fstd * stdv_rd[j]
            ub_rd = mv_rd[j] + fstd * stdv_rd[j]
            lb_baf = mv_baf[j] - fstd * stdv_baf[j]
            ub_baf = mv_baf[j] + fstd * stdv_baf[j]
            sp.log(
                msg=f"{sample}\tRD-variance bound={(lb_rd, ub_rd)}\tBAF-variance bound={(lb_baf, ub_baf)}\n",
                level="INFO",
            )

    ret_clusters = []
    for i, cluster in enumerate(clusters):
        dv_rd = np.abs(var_rd_matrix[i, :] - mv_rd)
        dv_baf = np.abs(var_baf_matrix[i, :] - mv_baf)
        if v >= 1:
            sp.log(
                msg=f"{cluster}\tRD-variance={var_rd_matrix[i, :]}\tBAF-variance={var_baf_matrix[i, :]}\n",
                level="INFO",
            )
        if np.all(dv_rd > (fstd * stdv_rd)) and np.all(dv_baf > (fstd * stdv_baf)):
            sp.log(msg=f"cluster {cluster} is outlier, removed\n", level="INFO")
            continue
        if v >= 1:
            sp.log(msg=f"cluster {cluster} Z(RD)={dv_rd / stdv_rd}\tZ(RD)={dv_baf / stdv_baf}\n", level="INFO")
        ret_clusters.append(cluster)

    sp.log(msg=f"remaining clusters: {ret_clusters}\n", level="INFO")
    return ret_clusters


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
    TODO handle the case when elbow criterion failed
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

    n2 = 0
    obj2 = 0
    if len(diploid_sols) > 0:
        data_diploid = [[n, imf_obj] for n, (_, imf_obj) in diploid_sols.items()]
        (n2, obj2) = select_best_n(data_diploid, "diploid")
        sp.log(
            msg=f"best diploid solution is n={n2} with IMF-objective={obj2}\n",
            level="INFO",
        )

    n4 = 0
    obj4 = 0
    if len(tetraploid_sols) > 0:
        data_tetraploid = [[n, imf_obj] for n, (_, imf_obj) in tetraploid_sols.items()]
        (n4, obj4) = select_best_n(data_tetraploid, "tetraploid")
        sp.log(
            msg=f"best tetraploid solution is n={n4} with IMF-objective={obj4}\n",
            level="INFO",
        )

    # pick best solution by principle of parsimony
    if len(tetraploid_sols) == 0:
        final_selection = "diploid"
    elif len(diploid_sols) == 0:
        final_selection = "tetraploid"
    else:
        final_selection = "diploid" if n2 <= n4 else "tetraploid"
    sp.log(msg=f"final selection: {final_selection}\n", level="INFO")
    return n2, n4, final_selection
