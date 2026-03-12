import textwrap
import math
import numpy as np
import pandas as pd
from pyomo import environ as pe
from pyomo.opt import SolverStatus, TerminationCondition


class CVXSubset:
    def __init__(
        self,
        n: int,
        cn_max: int,
        max_ncns_seg: int,
        minprop: float,
        ampdel: bool,
        copy_numbers: dict,
        f_a: pd.DataFrame,
        f_b: pd.DataFrame,
        w: pd.Series,
        purities: dict,
        baf: pd.DataFrame,
        copy_numbers_fixed: dict,
        purities_fixed: dict,
        penalty_param: list,
    ):
        # Each CVXSubset maintains its own data, so make a deep-copy of passed-in DataFrames
        f_a, f_b = f_a.copy(deep=True), f_b.copy(deep=True)

        assert f_a.shape == f_b.shape
        assert np.all(f_a.index == f_b.index)
        assert np.all(f_a.columns == f_b.columns)

        self.m, self.k = f_a.shape
        self.f_a = f_a
        self.f_b = f_b
        self.cluster_ids = f_a.index
        self.sample_ids = f_a.columns

        self.n = n
        self.cn_max = cn_max
        self.max_ncns_seg = max_ncns_seg
        self.minprop = minprop
        self.ampdel = ampdel
        self.copy_numbers = copy_numbers
        self.w = w
        self.purities = purities

        self.copy_numbers_fixed = copy_numbers_fixed  # TODO
        self.purities_fixed = purities_fixed
        self.penalty_param = penalty_param
        self.baf = baf

        self.tol = 0.001  # TODO make as argument?

        self.mode = "FULL"

        # Values we want to optimize for, as dataframes
        self.cA = [[np.nan for _ in range(self.n)] for _ in range(self.m)]
        self.cB = [[np.nan for _ in range(self.n)] for _ in range(self.m)]
        self.u = [[np.nan for _ in range(self.k)] for _ in range(self.n)]

        # Fixed values of cA/cB/u
        self._fixed_cA = [[np.nan for _ in range(self.n)] for _ in range(self.m)]
        self._fixed_cB = [[np.nan for _ in range(self.n)] for _ in range(self.m)]
        self._fixed_u = [[np.nan for _ in range(self.k)] for _ in range(self.n)]
        self.warmstart = False  # set on hot_start()
        self.model = None  # initialized on create_model()

    def __str__(self):
        # Pyomo pprint gives us too much information - too unwieldy for large models
        # This method is implemented to supply the bare-minimum but useful model information.
        if self.model is None:
            return ""
        else:
            return textwrap.dedent(
                f"""
                # ------------------------------------------
                #   Problem Information
                # ------------------------------------------
                #     Number of constraints: {self.model.nconstraints()}
                #     Number of variables: {self.model.nvariables()}
                # ------------------------------------------
            """
            )

    def create_model(self, pprint=False):
        m, n, k = self.m, self.n, self.k
        f_a, f_b = self.f_a, self.f_b
        cn_max = self.cn_max
        copy_numbers = self.copy_numbers
        # copy_numbers_fixed = self.copy_numbers_fixed
        # purities_fixed = self.purities_fixed
        # mode_t = self.mode
        # max_ncns_seg = self.max_ncns_seg
        # purities = self.purities

        model = pe.ConcreteModel()
        model.constraints = pe.ConstraintList()

        # A and B matrix
        for _m in range(m):
            cluster_id = f_a.index[_m]

            # upper bound for solver
            ub = max(sum(copy_numbers.get(cluster_id, (0, 0))), cn_max)

            for _n in range(n):
                self.cA[_m][_n] = pe.Var(bounds=(0, ub), domain=pe.Integers)
                model.add_component(f"cA_{_m + 1}_{_n + 1}", self.cA[_m][_n])
                self.cB[_m][_n] = pe.Var(bounds=(0, ub), domain=pe.Integers)
                model.add_component(f"cB_{_m + 1}_{_n + 1}", self.cB[_m][_n])

        # U matrix
        for _n in range(n):
            for _k in range(k):
                self.u[_n][_k] = pe.Var(bounds=(0, 1), domain=pe.Reals)
                model.add_component(f"u_{_n + 1}_{_k + 1}", self.u[_n][_k])

        u_aux = {}
        for _n in range(n):
            for _k in range(k):
                u_aux[(_n, _k)] = pe.Var(domain=pe.Binary)
                model.add_component(f"u_aux_{_n + 1}_{_k + 1}", u_aux[(_n, _k)])

        # fix copy-numbers
        for _m, (a, b) in copy_numbers.items():
            cluster_id = f_a.index[_m]
            model.constraints.add(self.cA[_m][0] == 1)
            model.constraints.add(self.cB[_m][0] == 1)
            for _n in range(n):
                model.constraints.add(self.cA[_m][_n] == a)
                model.constraints.add(self.cB[_m][_n] == b)

        # copy-number constraints
        for _m in range(m):
            cluster_id = f_a.index[_m]
            for _n in range(n):
                model.constraints.add(self.cA[_m][_n] + self.cB[_m][_n] <= cn_max)

        # purity constraint
        for _k in range(k):
            model.constraints.add(sum(self.u[_n][_k] for _n in range(n)) == 1)

        for _n in range(n):
            for _k in range(k):
                model.constraints.add(u_aux[(_n, _k)] >= self.u[_n][_k])
                model.constraints.add(self.u[_n][_k] >= self.minprop * u_aux[(_n, _k)])

        # set objectives
        yA = {}
        yB = {}
        for _m in range(m):
            cluster_id = f_a.index[_m]
            f_a_values = f_a.loc[cluster_id].values
            f_b_values = f_b.loc[cluster_id].values
            for _k in range(k):
                yA[(_m, _k)] = pe.Var(bounds=(0, np.inf), domain=pe.Reals)
                model.add_component(f"yA_{_m + 1}_{_k + 1}", yA[(_m, _k)])
                yB[(_m, _k)] = pe.Var(bounds=(0, np.inf), domain=pe.Reals)
                model.add_component(f"yB_{_m + 1}_{_k + 1}", yB[(_m, _k)])

                a_dot = sum(self.cA[_m][_n] * self.u[_n][_k] for _n in range(n))
                b_dot = sum(self.cB[_m][_n] * self.u[_n][_k] for _n in range(n))
                model.constraints.add(float(f_a_values[_k]) - a_dot <= yA[(_m, _k)])
                model.constraints.add(a_dot - float(f_a_values[_k]) <= yA[(_m, _k)])
                model.constraints.add(float(f_b_values[_k]) - b_dot <= yB[(_m, _k)])
                model.constraints.add(b_dot - float(f_b_values[_k]) <= yB[(_m, _k)])

        objective = 0
        for _m in range(m):
            for _k in range(k):
                objective += (yA[(_m, _k)] + yB[(_m, _k)]) * self.w[
                    self.cluster_ids[_m]
                ]

        # set penalties
        model.obj = pe.Objective(expr=objective, sense=pe.minimize)
        self.model = model

        if pprint:
            print(str(self))
        return

    def run(self, solver_type="gurobi", timelimit=None, write_path=None):
        if solver_type == "gurobipy":
            solver = pe.SolverFactory("gurobi", solver_io="python")
        else:
            solver = pe.SolverFactory(solver_type)

        kwargs = {"report_timing": False}
        if timelimit is not None:
            kwargs["timelimit"] = int(timelimit)
        if solver.warm_start_capable():
            kwargs["warmstart"] = self.warmstart

        results = solver.solve(self.model, **kwargs)
        if (
            (results.solver.status == SolverStatus.ok)
            and (
                results.solver.termination_condition
                in (
                    TerminationCondition.optimal,
                    TerminationCondition.feasible,
                )
            )
            or (
                results.solver.status == SolverStatus.aborted
                and results.solver.termination_condition
                == TerminationCondition.maxTimeLimit
            )
        ):
            pass
        else:
            return None

        if write_path is not None:
            self.model.write(write_path)

        return (
            self.model.obj(),
            [[int(getattr(x, "value", x)) for x in row] for row in self.cA],
            [[int(getattr(x, "value", x)) for x in row] for row in self.cB],
            [[getattr(x, "value", x) for x in row] for row in self.u],
        )
