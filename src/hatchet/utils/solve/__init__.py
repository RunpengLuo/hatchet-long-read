import os
import numpy as np
import pandas as pd
from pyomo import environ as pe
from collections import OrderedDict
import kneed
import matplotlib.pyplot as plt

import hatchet.utils.Supporting as sp

from hatchet.utils.solve.utils import *
from hatchet.utils.solve.ilp_subset import ILPSubset
from hatchet.utils.solve.ilp_subset_split import ILPSubsetSplit
from hatchet.utils.solve.cd import CoordinateDescent, CoordinateDescentSplit
from hatchet.utils.solve.cvx_subset import CVXSubset


def solver_available(solver_type: str):
    if solver_type == "cpp":
        return os.getenv("GRB_LICENSE_FILE") is not None

    if solver_type == "gurobipy":
        return pe.SolverFactory("gurobi", solver_io="python").available(
            exception_flag=False
        )

    return pe.SolverFactory(solver_type).available(exception_flag=False)


def solve(
    f_a: pd.DataFrame,
    f_b: pd.DataFrame,
    n: int,
    minprop: float,
    max_ncns_seg: int,
    cn_max: int,
    weights: pd.Series,
    ampdel: bool,
    clonal: dict,
    purities: dict,
    baf: pd.DataFrame,
    copy_numbers_fixed: dict,
    purities_fixed: dict,
    reg_term: list,
    solver_type: str,
    max_iters: int,
    n_seed: int,
    n_worker: int,
    random_seed: int,
    timelimit: int,
    instances_dir: str,
    solve_mode: str,
    verbose=False,
):
    cd_instances = None
    if solve_mode in ["cd", "both"]:
        cd = CoordinateDescent(
            f_a=f_a,
            f_b=f_b,
            n=n,
            minprop=minprop,
            max_ncns_seg=max_ncns_seg,
            cn_max=cn_max,
            w=weights,
            ampdel=ampdel,
            cn=clonal,
            purities=purities,
            baf=baf,
            copy_numbers_fixed=copy_numbers_fixed,
            purities_fixed=purities_fixed,
            penalty_param=["RAW", 0.0],
        )

        # obj. value => (cA, cB, u) mapping
        cd_instances = cd.run(
            solver_type=solver_type,
            max_iters=max_iters,
            n_seed=n_seed,
            j=n_worker,
            random_seed=random_seed,
            timelimit=timelimit,
        )
        if instances_dir != None:
            store_instance_tofile(
                cd_instances,
                f_a,
                f_b,
                baf,
                instances_dir,
                "cd",
                n,
            )

    sol_instances = None
    sol_class = {"ilp": ILPSubset, "both": ILPSubset, "cvx": CVXSubset}
    if solve_mode in ["ilp", "both", "cvx"]:
        sol_instances = {}
        [pname, num_steps, step_size] = reg_term
        solver = sol_class[solve_mode](
            n,
            cn_max,
            max_ncns_seg=max_ncns_seg,
            minprop=minprop,
            ampdel=ampdel,
            copy_numbers=clonal,
            f_a=f_a,
            f_b=f_b,
            w=weights,
            purities=purities,
            baf=baf,
            copy_numbers_fixed=copy_numbers_fixed,
            purities_fixed=purities_fixed,
            penalty_param=[pname, 0.0],
        )
        solver.create_model(pprint=verbose)
        if solve_mode == "both":
            # select local-opt from coordinate-descent instances
            # TODO does the starting point be more useful to do additional model selection?
            _, [obj, cA, cB, _] = min(cd_instances.items(), key=lambda tp: tp[1][0])
            sp.log(
                msg=f"use CD local opt with obj={obj} to initialize ILP model\n",
                level="STEP",
            )
            solver.hot_start(cA, cB)

        for i0 in range(0, num_steps + 1):
            if verbose:
                sp.log(
                    msg=f"running instance {i0}/{num_steps}\n",
                    level="STEP",
                )
            pparam = step_size * i0
            solver.model.pparam = pparam
            if i0 > 0:
                cA, cB = sol_instances[0][1:3]
                solver.hot_start(cA, cB)
            sol_instances[pparam] = solver.run(solver_type=solver_type, timelimit=timelimit)
            assert sol_instances[pparam] != None, f"ERROR! optimization failed."

        if instances_dir != None:
            store_instance_tofile(
                sol_instances,
                f_a,
                f_b,
                baf,
                instances_dir,
                solve_mode,
                n,
            )

    if solve_mode == "cd":
        return cd_instances
    else:
        return sol_instances


def model_selection_instance(
    f_a: pd.DataFrame,
    f_b: pd.DataFrame,
    weights: pd.Series,
    instances: dict,
    pname: str,
    solve_mode: str,
    outdir: str,
):
    """
    use elbow criterion to select best instance from either
    1) ILP or CD+ILP with scalaried solutions, or
    2) CD only solutions

    if solve_mode != cd, float-number error with be estimated.
    """
    assert len(instances) > 0, "ERROR! there is no solution to be selected"

    if pname == "RAW" or len(instances) == 1:
        return instances[0], instances[0][1]

    data = []
    errv = 0.0
    for param, [tobj, cA, cB, u] in sorted(instances.items(), key=lambda tp: tp[0]):
        [imf_obj, reg_obj] = compute_individual_objs(
            pname, weights, f_a, f_b, cA, cB, u
        )
        if solve_mode != "cd":
            errv = tobj - (imf_obj + param * reg_obj)
        else:
            errv = tobj - imf_obj
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
    plt.savefig(f"{outdir}/pareto_curve.{solve_mode}.{pname}.png", dpi=300)

    sol_index = 0
    if elbow_x == None:
        sp.log(
            msg=f"Failed to identify elbow in model selection step, use result without penalty.\n",
            level="WARN",
        )
    else:
        sol_indices = np.where(ys >= elbow_y)[0]
        if len(sol_indices) == 0:
            sp.log(
                msg=f"Failed to locate result in model selection step, use result without penalty.\n",
                level="WARN",
            )
        else:
            # multiple instance may yield same objective values, pick the one with minimum penalty
            sol_index = sol_indices[0]
            sp.log(
                msg=f"Model selection found solution with index={sol_index}!\n",
                level="INFO",
            )
    df.loc[:, "selected"] = ""
    df.loc[sol_index, "selected"] = "*"
    df.to_csv(
        f"{outdir}/model_selections.{solve_mode}.{pname}.tsv",
        sep="\t",
        header=True,
        index=False,
    )

    return instances[df.loc[sol_index, "Lambda"]], df.loc[sol_index, "IMF-objective"]
