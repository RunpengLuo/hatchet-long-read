import os
import numpy as np
import pandas as pd
from pyomo import environ as pe
from collections import OrderedDict

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
    [pname, num_steps, step_size] = reg_term

    cd_instances = None
    if solve_mode in ("cd", "both"):
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
            copy_numbers_fixed=copy_numbers_fixed,
            purities_fixed=purities_fixed,
            reg_term=reg_term,
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
    if solve_mode in ("ilp", "both", "cvx"):
        sol_instances = {}
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
            sol_instances[pparam] = solver.run(
                solver_type=solver_type, timelimit=timelimit
            )
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
