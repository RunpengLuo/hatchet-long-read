import os
import numpy as np
import pandas as pd
from pyomo import environ as pe
from collections import OrderedDict

from hatchet.utils.solve.ilp_subset import ILPSubset, ILPSubsetSplit
from hatchet.utils.solve.cd import CoordinateDescent, CoordinateDescentSplit
from hatchet.utils.solve.utils import parse_clonal, scale_rdr, store_temp_result
from hatchet import config
import hatchet.utils.Supporting as sp


def solver_available(solver=None):
    solver = solver or config.compute_cn.solver
    if solver == "cpp":
        return os.getenv("GRB_LICENSE_FILE") is not None
    elif solver == "gurobipy":
        return pe.SolverFactory("gurobi", solver_io="python").available(
            exception_flag=False
        )
    return pe.SolverFactory(solver).available(exception_flag=False)


def solve(
    clonal,
    bbc_file,
    seg_file,
    n, # n_clones
    solver="gurobi",
    solve_mode="cd",
    d=-1, # number of distinct states per segment
    cn_max=-1, # threshold, max cA + cB per segment
    mu=0.01, #u_min
    diploid_threshold=0.1, # constant TODO
    ampdel=True, # all amp or all del
    n_seed=400,
    n_worker=8,
    random_seed=None,
    max_iters=None, # cd
    timelimit=None,
    binwise=False,
    purities=None,
    tempdir=None,
):
    assert solve_mode in ("ilp", "cd", "both"), "Unrecognized solve_mode"
    assert solver_available(solver), f"Solver {solver} not available or not licensed"

    if max_iters is None:
        max_iters = 10

    df = pd.read_csv(seg_file, sep="\t")
    df = df.sort_values(["#ID", "SAMPLE"])

    bbc = pd.read_table(bbc_file)
    bbc = bbc.sort_values(by=["#CHR", "START", "SAMPLE"])

    # sanity-check
    sample_ids = np.sort(df["SAMPLE"].unique())
    for _cluster_id, _df in df.groupby("#ID"):
        _sample_ids = _df["SAMPLE"].values
        if not np.all(_sample_ids == sample_ids):
            raise ValueError(
                f"Sample IDs for cluster {_cluster_id} do not match {sample_ids}"
            )

    rdr = df.pivot(index="#ID", columns="SAMPLE", values="RD")
    baf = df.pivot(index="#ID", columns="SAMPLE", values="BAF")

    bins = {}  # cluster_id => no. of bins
    for cluster_id, _df in df.groupby("#ID"):
        if len(_df["#BINS"].unique()) != 1:
            raise ValueError(
                f"Bin sizes for cluster {cluster_id} across tumor samples are not identical!"
            )
        bins[cluster_id] = _df.iloc[0]["#BINS"]
    bins = pd.Series(bins)

    weights = 100 * bins / sum(bins)

    if clonal is None:
        _candidate_cluster_ids = np.all(rdr > 0.5 - diploid_threshold, axis=1)
        if not np.any(_candidate_cluster_ids):
            raise RuntimeError(
                f"Unable to determine cluster with diploid RDR threshold {diploid_threshold}"
            )
        # get cluster with BAF close to 0.5 and has most clustered bins
        dipoid_cluster_id = (_candidate_cluster_ids * weights).idxmax()
        copy_numbers = {dipoid_cluster_id: (1, 1)}
    else:
        copy_numbers = parse_clonal(clonal)

    gamma = scale_rdr(rdr, copy_numbers)
    sp.log(
        msg="Computed scaling factor gamma = \n" + str(gamma) + "\n",
        level="INFO",
    )

    fcn = rdr * gamma
    rdr = rdr * gamma
    f_b = rdr * baf
    f_a = rdr - f_b

    if not binwise:
        # TODO HT941
        if sample_ids[0] == "HT941" and n == 3:
            copy_numbers_fixed = {1: [(1,1),(1,1)], 
                                  6: [(1,0),(1,0)]}
            for cluster_id in list(copy_numbers_fixed.keys()):
                if cluster_id in copy_numbers:
                    copy_numbers_fixed.pop(cluster_id)
        else:
            copy_numbers_fixed = None
        # store detailed config
        with open(f"{tempdir}/config.txt", 'w') as fd:
            fd.write("========================================\n")
            fd.write(f"mode={solve_mode}\nsolver={solver}\nbinwise={binwise}\n")
            fd.write(f"max_iters={max_iters}\ntimelimit={timelimit}\n")
            fd.write(f"random_seed={random_seed}\nnum_seeds={n_seed}\n")
            fd.write("========================================\n")
            fd.write(f"clonal={list(copy_numbers.items())}\n")
            if copy_numbers_fixed != None:
                fd.write(f"cn_fixed={list(copy_numbers_fixed.items())}\n")
            fd.write(f"purity={purities}\n")
            fd.write("========================================\n")
            fd.write(f"(m,n,k)=({f_a.shape[0]},{n},{f_a.shape[1]})\n")
            fd.write(f"u_min={mu}\ncn_max={cn_max}\nampdel={ampdel}\n")
            fd.write(f"base={min(2, len(copy_numbers))}\n")
            fd.write(f"d={d}\n")
            fd.write("========================================\n")
            for sample in sample_ids:
                fd.write(f"{sample}\tgamma={gamma[sample].tolist()}\n")
            fd.close()

            with open(f"{tempdir}/input.tsv", 'w') as fd:
                fd.write("CLUSTER\tSAMPLE\tBAF\tRDR\tFCN\tF_A\tF_B\tweight\n")
                for sample in sample_ids:
                    for cID in df["#ID"].unique().tolist():
                        fd.write('\t'.join(
                            [str(cID), str(sample),
                             str(baf.loc[cID, sample]),
                             str(rdr.loc[cID, sample]),
                             str(fcn.loc[cID, sample]),
                             str(f_a.loc[cID, sample]),
                             str(f_b.loc[cID, sample]),
                             str(weights[cID])
                            ]) + '\n')
                fd.close()

        if solve_mode == "cd" or solve_mode == "both":
            cd = CoordinateDescent(
                f_a=f_a,
                f_b=f_b,
                n=n,
                mu=mu,
                d=d,
                cn_max=cn_max,
                w=weights,
                ampdel=ampdel,
                cn=copy_numbers,
                purities=purities,
                baf=baf,
                copy_numbers_fixed=copy_numbers_fixed
            )
            obj, cA, cB, u, cluster_ids, sample_ids = cd.run(
                solver_type=solver,
                max_iters=max_iters,
                n_seed=n_seed,
                j=n_worker,
                random_seed=random_seed,
                timelimit=timelimit,
                tempdir=tempdir,
            )
        
        if solve_mode == "ilp" or solve_mode == "both":
            ilp = ILPSubset(
                n,
                cn_max,
                d=d,
                mu=mu,
                ampdel=ampdel,
                copy_numbers=copy_numbers,
                f_a=f_a,
                f_b=f_b,
                w=weights,
                purities=purities,
                baf=baf,
                copy_numbers_fixed=copy_numbers_fixed
            )
            if solve_mode == "ilp":
                ilp.create_model(pprint=True)
            else:
                # run coordinate-descent first to get local-opt cA and cB
                # use cA and cB to hot start the model.
                ilp.create_model()
                ilp.hot_start(cA, cB)

            obj, cA, cB, u, cluster_ids, sample_ids = ilp.run(
                solver_type=solver, timelimit=timelimit)
            store_temp_result({obj: [cA, cB, u]}, cluster_ids, sample_ids, f_a, f_b, 
                              baf, tempdir, solve_mode, n)
        return obj, cA, cB, u, cluster_ids, sample_ids

    else:
        bins = OrderedDict()  # cluster_id => RDR for cluster
        for cluster_id, _df in df.groupby("#ID"):
            if len(_df["#BINS"].unique()) != 1:
                raise ValueError(
                    f"Bin sizes for cluster {cluster_id} across tumor samples are not identical!"
                )
            my_bbc = bbc[bbc.CLUSTER == cluster_id]
            if _df.iloc[0]["#BINS"] * len(_df) != len(
                my_bbc
            ):  # seg file should have 1 row per sample
                raise ValueError(
                    f"BBC and SEG files describe inconsisitent # bins for cluster {cluster_id}!"
                )
            bins[cluster_id] = my_bbc.RD

        binned_length = (
            bbc[bbc.SAMPLE == bbc.iloc[0].SAMPLE].END
            - bbc[bbc.SAMPLE == bbc.iloc[0].SAMPLE].START
        ).sum()

        bins_rdr = {
            k: my_df.pivot(index=["#CHR", "START"], columns="SAMPLE", values="RD")
            for k, my_df in bbc.groupby("CLUSTER")
        }
        bins_baf = {
            k: my_df.pivot(index=["#CHR", "START"], columns="SAMPLE", values="BAF")
            for k, my_df in bbc.groupby("CLUSTER")
        }
        bins_length = {
            k: (
                (
                    my_df.pivot(index=["#CHR", "START"], columns="SAMPLE", values="END")
                    - my_df.pivot(
                        index=["#CHR", "START"],
                        columns="SAMPLE",
                        values="START",
                    )
                ).values[:, 0]
                * 100
            )
            / binned_length
            for k, my_df in bbc.groupby("CLUSTER")
        }

        binsA = {}
        binsB = {}
        for k in bins_rdr.keys():
            bins_rdr[k] = bins_rdr[k] * gamma
            binsB[k] = bins_rdr[k] * bins_baf[k]
            binsA[k] = bins_rdr[k] - binsB[k]

        if solve_mode == "ilp":
            ilp = ILPSubsetSplit(
                n,
                cn_max,
                d=d,
                mu=mu,
                ampdel=ampdel,
                copy_numbers=copy_numbers,
                f_a=f_a,
                f_b=f_b,
                binsA=binsA,
                binsB=binsB,
                lengths=bins_length,
                purities=purities,
            )
            ilp.create_model(pprint=True)
            return ilp.run(solver_type=solver, timelimit=timelimit)
        elif solve_mode == "cd":
            cd = CoordinateDescentSplit(
                f_a=f_a,
                f_b=f_b,
                n=n,
                mu=mu,
                d=d,
                cn_max=cn_max,
                ampdel=ampdel,
                cn=copy_numbers,
                binsA=binsA,
                binsB=binsB,
                lengths=bins_length,
            )
            return cd.run(
                solver_type=solver,
                max_iters=max_iters,
                n_seed=n_seed,
                j=n_worker,
                random_seed=random_seed,
                timelimit=timelimit,
            )
        else:
            cd = CoordinateDescentSplit(
                f_a=f_a,
                f_b=f_b,
                n=n,
                mu=mu,
                d=d,
                cn_max=cn_max,
                ampdel=ampdel,
                cn=copy_numbers,
                binsA=binsA,
                binsB=binsB,
                lengths=bins_length,
            )
            _, cA, cB, _, _, _ = cd.run(
                solver_type=solver,
                max_iters=max_iters,
                n_seed=n_seed,
                j=n_worker,
                random_seed=random_seed,
                timelimit=timelimit,
            )

            ilp = ILPSubsetSplit(
                n,
                cn_max,
                d=d,
                mu=mu,
                ampdel=ampdel,
                copy_numbers=copy_numbers,
                f_a=f_a,
                f_b=f_b,
                cn=copy_numbers,
                binsA=binsA,
                binsB=binsB,
                lengths=bins_length,
            )
            ilp.create_model()
            ilp.hot_start(cA, cB)
            return ilp.run(solver_type=solver, timelimit=timelimit)
