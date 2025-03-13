import os
import numpy as np
import pandas as pd
from pyomo import environ as pe
from collections import OrderedDict
import kneed
import matplotlib.pyplot as plt

from hatchet.utils.solve.utils import *
from hatchet.utils.solve.ilp_subset import ILPSubset, ILPSubsetSplit
from hatchet.utils.solve.cd import CoordinateDescent, CoordinateDescentSplit

def solver_available(solver: str):
    if solver == "cpp":
        return os.getenv("GRB_LICENSE_FILE") is not None

    if solver == "gurobipy":
        return pe.SolverFactory("gurobi", solver_io="python").available(
            exception_flag=False
        )

    return pe.SolverFactory(solver).available(exception_flag=False)


def solve_instance(
    f_a,
    f_b,
    n,
    mu,
    d,
    cn_max,
    weights,
    ampdel,
    clonal,
    purities,
    baf,
    copy_numbers_fixed,
    purities_fixed,
    penalty_param,
    solver,
    max_iters,
    n_seed,
    n_worker,
    random_seed,
    timelimit,
    instance_dir,
    solve_mode,
):
    """
    solve optimization with specific problem parameter setting
    """
    assert solve_mode in ("ilp", "cd", "both"), "Unrecognized solve_mode"
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
            cn=clonal,
            purities=purities,
            baf=baf,
            copy_numbers_fixed=copy_numbers_fixed,
            purities_fixed=purities_fixed,
            penalty_param=penalty_param,
        )
        obj, cA, cB, u, cluster_ids, sample_ids = cd.run(
            solver_type=solver,
            max_iters=max_iters,
            n_seed=n_seed,
            j=n_worker,
            random_seed=random_seed,
            timelimit=timelimit,
            tempdir=instance_dir,
        )

    if solve_mode == "ilp" or solve_mode == "both":
        ilp = ILPSubset(
            n,
            cn_max,
            d=d,
            mu=mu,
            ampdel=ampdel,
            copy_numbers=clonal,
            f_a=f_a,
            f_b=f_b,
            w=weights,
            purities=purities,
            baf=baf,
            copy_numbers_fixed=copy_numbers_fixed,
            purities_fixed=purities_fixed,
            penalty_param=penalty_param,
        )
        if solve_mode == "ilp":
            ilp.create_model(pprint=False)
        else:
            # run coordinate-descent first to get local-opt cA and cB
            # use cA and cB to hot start the model.
            ilp.create_model()
            ilp.hot_start(cA, cB)

        obj, cA, cB, u, cluster_ids, sample_ids = ilp.run(
            solver_type=solver, timelimit=timelimit
        )
        store_instance_tofile(
            {obj: [cA, cB, u]},
            cluster_ids,
            sample_ids,
            f_a,
            f_b,
            baf,
            instance_dir,
            solve_mode,
            n,
        )
    return obj, cA, cB, u, cluster_ids, sample_ids


def model_selection_instance(
    f_a: pd.DataFrame,
    f_b: pd.DataFrame,
    weights: pd.Series,
    instances: dict,
    pname: str,
    outdir: str,
):
    """
    select best instance among all scalarized solutions
    """
    assert len(instances) > 0, "ERROR! there is no solution to be selected"

    if pname == "RAW" or len(instances) == 1:
        return instances[0]
    data = []
    for param, [tobj, cA, cB, u, _, _] in sorted(
        instances.items(), key=lambda tp: tp[0]
    ):
        [imf_obj, reg_obj] = compute_individual_objs(
            pname, weights, f_a, f_b, cA, cB, u
        )
        errv = tobj - (imf_obj + param * reg_obj)
        data.append([param, tobj, imf_obj, reg_obj, errv])

    df = pd.DataFrame(
        data=data,
        columns=[
            "Lambda",
            "Objective",
            "IMF-objective",
            f"{pname}-objective",
            "float-error",
        ],
    )

    # find elbow point
    xs = df[f"{pname}-objective"].to_numpy()
    ys = df["IMF-objective"].to_numpy()
    kl = kneed.KneeLocator(x=xs, y=ys, curve="convex", direction="decreasing")
    elbow_x, elbow_y = kl.elbow, kl.elbow_y
    kl.plot_knee(
        title="Model Selection Pareto Curve",
        xlabel=f"{pname}-objective",
        ylabel="IMF-objective",
    )
    plt.savefig(f"{outdir}/pareto_curve.{pname}.png", dpi=300)

    sol_index = 0
    if elbow_x == None:
        print(
            f"Failed to identify elbow in model selection step, use result without penalty."
        )
    else:
        sol_indices = np.where(ys >= elbow_y)[0]
        if len(sol_indices) == 0:
            print(
                f"Failed to locate result in model selection step, use result without penalty."
            )
        else:
            # multiple instance may yield same objective values, pick the one with minimum penalty
            sol_index = sol_indices[0]
            print(f"Model selection found solution with index={sol_index}")
    df.loc[:, "selected"] = ""
    df.loc[sol_index, "selected"] = "*"
    df.to_csv(f"{outdir}/model_selections.tsv", sep="\t", header=True, index=False)

    return instances[df.loc[sol_index, "Lambda"]]
