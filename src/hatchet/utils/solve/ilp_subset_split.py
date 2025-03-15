import textwrap
import math
import numpy as np
from pyomo import environ as pe
from pyomo.opt import SolverStatus, TerminationCondition

from hatchet.utils.solve.utils import Random
from hatchet.utils.solve.ilp_subset import ILPSubset


class ILPSubsetSplit(ILPSubset):
    def __init__(
        self,
        n,
        cn_max,
        d,
        mu,
        ampdel,
        copy_numbers,
        f_a,
        f_b,
        binsA,
        binsB,
        lengths,
    ):
        # Each ILPSubset maintains its own data, so make a deep-copy of passed-in DataFrames
        f_a, f_b = f_a.copy(deep=True), f_b.copy(deep=True)
        self.binsA = {k: v.copy(deep=True) for k, v in binsA.items()}
        self.binsB = {k: v.copy(deep=True) for k, v in binsB.items()}
        self.lengths = {k: np.copy(v) for k, v in lengths.items()}

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
        self.d = d
        self.mu = mu
        self.ampdel = ampdel
        self.copy_numbers = copy_numbers

        self.tol = 0.001

        self.mode = "FULL"

        # Values we want to optimize for, as dataframes
        self.cA = [[np.nan for _ in range(self.n)] for _ in range(self.m)]
        self.cB = [[np.nan for _ in range(self.n)] for _ in range(self.m)]
        self.u = [[np.nan for _ in range(self.k)] for _ in range(self.n)]

        # Fixed values of cA/cB/u
        self._fixed_cA = [[np.nan for _ in range(n)] for _ in range(self.m)]
        self._fixed_cB = [[np.nan for _ in range(n)] for _ in range(self.m)]
        self._fixed_u = [[np.nan for _ in range(self.k)] for _ in range(self.n)]

        self.warmstart = False  # set on hot_start()
        self.model = None  # initialized on create_model()

    def __copy__(self):
        return ILPSubsetSplit(
            n=self.n,
            cn_max=self.cn_max,
            d=self.d,
            mu=self.mu,
            ampdel=self.ampdel,
            copy_numbers=self.copy_numbers,
            f_a=self.f_a,
            f_b=self.f_b,
            binsA=self.binsA,
            binsB=self.binsB,
            lengths=self.lengths,
        )

    def create_model(self, pprint=False):
        m, n, k = self.m, self.n, self.k
        f_a = self.f_a
        cn_max = self.cn_max
        ampdel = self.ampdel
        copy_numbers = self.copy_numbers
        mode_t = self.mode
        d = self.d
        _M = self.M
        _base = self.base

        model = pe.ConcreteModel()

        fA = {}
        fB = {}
        yA = {}
        yB = {}
        adA = {}
        adB = {}

        for _m in range(m):
            cluster_id = f_a.index[_m]

            # upper bound for solver
            ub = max(sum(copy_numbers.get(cluster_id, (0, 0))), cn_max)

            for _k in range(k):
                fA[(_m, _k)] = pe.Var(bounds=(0, ub), domain=pe.Reals)
                model.add_component(f"fA_{_m + 1}_{_k + 1}", fA[(_m, _k)])
                fB[(_m, _k)] = pe.Var(bounds=(0, ub), domain=pe.Reals)
                model.add_component(f"fB_{_m + 1}_{_k + 1}", fB[(_m, _k)])

                # Need an objective function term for each bin for absolute vlaue
                for _i in range(len(self.binsA[cluster_id])):
                    yA[(_m, _k, _i)] = pe.Var(bounds=(0, np.inf), domain=pe.Reals)
                    model.add_component(
                        f"yA_{_m + 1}_{_k + 1}_{_i + 1}", yA[(_m, _k, _i)]
                    )
                    yB[(_m, _k, _i)] = pe.Var(bounds=(0, np.inf), domain=pe.Reals)
                    model.add_component(
                        f"yB_{_m + 1}_{_k + 1}_{_i + 1}", yB[(_m, _k, _i)]
                    )

        if mode_t in ("FULL", "CARCH"):
            for _m in range(m):
                cluster_id = f_a.index[_m]

                # upper bound for solver
                ub = max(sum(copy_numbers.get(cluster_id, (0, 0))), cn_max)

                for _n in range(n):
                    self.cA[_m][_n] = pe.Var(bounds=(0, ub), domain=pe.Integers)
                    model.add_component(f"cA_{_m + 1}_{_n + 1}", self.cA[_m][_n])
                    self.cB[_m][_n] = pe.Var(bounds=(0, ub), domain=pe.Integers)
                    model.add_component(f"cB_{_m + 1}_{_n + 1}", self.cB[_m][_n])

            if ampdel:
                for _m in range(m):
                    cluster_id = f_a.index[_m]
                    if cluster_id not in copy_numbers:
                        adA[_m] = pe.Var(bounds=(0, 1), domain=pe.Binary)
                        model.add_component(f"adA_{_m + 1}", adA[_m])
                        adB[_m] = pe.Var(bounds=(0, 1), domain=pe.Binary)
                        model.add_component(f"adB_{_m + 1}", adB[_m])

        bitcA = {}
        bitcB = {}
        if (mode_t == "FULL") or (d > 0 and mode_t == "CARCH"):
            for _b in range(_M):
                for _m in range(m):
                    for _n in range(n):
                        bitcA[(_b, _m, _n)] = pe.Var(bounds=(0, 1), domain=pe.Binary)
                        model.add_component(
                            f"bitcA_{_b + 1}_{_m + 1}_{_n + 1}",
                            bitcA[(_b, _m, _n)],
                        )
                        bitcB[(_b, _m, _n)] = pe.Var(bounds=(0, 1), domain=pe.Binary)
                        model.add_component(
                            f"bitcB_{_b + 1}_{_m + 1}_{_n + 1}",
                            bitcB[(_b, _m, _n)],
                        )

        if mode_t in ("FULL", "UARCH"):
            for _n in range(n):
                for _k in range(k):
                    self.u[_n][_k] = pe.Var(bounds=(0, 1), domain=pe.Reals)
                    model.add_component(f"u_{_n + 1}_{_k + 1}", self.u[_n][_k])

        vA = {}
        vB = {}
        if mode_t == "FULL":
            for _b in range(_M):
                for _m in range(m):
                    for _n in range(n):
                        for _k in range(k):
                            vA[(_b, _m, _n, _k)] = pe.Var(
                                bounds=(0, 1), domain=pe.Reals
                            )
                            model.add_component(
                                f"vA_{_b + 1}_{_m + 1}_{_n + 1}_{_k + 1}",
                                vA[(_b, _m, _n, _k)],
                            )
                            vB[(_b, _m, _n, _k)] = pe.Var(
                                bounds=(0, 1), domain=pe.Reals
                            )
                            model.add_component(
                                f"vB_{_b + 1}_{_m + 1}_{_n + 1}_{_k + 1}",
                                vB[(_b, _m, _n, _k)],
                            )

        x = {}
        if (mode_t in ("FULL", "UARCH")) and (self.mu > 0):
            for _n in range(n):
                for _k in range(k):
                    x[(_n, _k)] = pe.Var(domain=pe.Binary)
                    model.add_component(f"x_{_n + 1}_{_k + 1}", x[(_n, _k)])

        # buildOptionalVariables
        z = {}
        if (mode_t in ("FULL", "CARCH")) and d > 0:
            for _m in range(self.m):
                for _n in range(1, self.n):
                    for _d in range(d):
                        z[(_m, _n, _d)] = pe.Var(bounds=(0, 1), domain=pe.Binary)
                        model.add_component(
                            f"z_{_m + 1}_{_n + 1}_{_d + 1}", z[(_m, _n, _d)]
                        )

        # CONSTRAINTS
        model.constraints = pe.ConstraintList()

        for _m in range(m):
            cluster_id = f_a.index[_m]
            f_a_values = self.binsA[cluster_id].values
            f_b_values = self.binsB[cluster_id].values

            for _k in range(k):
                _fA, _fB = fA[(_m, _k)], fB[(_m, _k)]

                for _i in range(f_a_values.shape[0]):
                    _yA, _yB = yA[(_m, _k, _i)], yB[(_m, _k, _i)]

                    # Instead of 1 term per inferred CN, use all values in bin
                    model.constraints.add(f_a_values[_i, _k] - _fA <= _yA)
                    model.constraints.add(_fA - f_a_values[_i, _k] <= _yA)

                    # model.constraints.add(myA >= _yA)
                    model.constraints.add(f_b_values[_i, _k] - _fB <= _yB)
                    model.constraints.add(_fB - f_b_values[_i, _k] <= _yB)
                    # model.constraints.add(myB >= _yB)

        if mode_t == "FULL":
            for _m in range(m):
                for _k in range(k):
                    sum_a = 0
                    for _n in range(n):
                        for _b in range(_M):
                            sum_a += vA[(_b, _m, _n, _k)] * math.pow(2, _b)
                            model.constraints.add(
                                vA[(_b, _m, _n, _k)] <= bitcA[(_b, _m, _n)]
                            )
                            model.constraints.add(
                                vA[(_b, _m, _n, _k)] <= self.u[_n][_k]
                            )
                            model.constraints.add(
                                vA[(_b, _m, _n, _k)]
                                >= bitcA[(_b, _m, _n)] + self.u[_n][_k] - 1
                            )

                    model.constraints.add(fA[(_m, _k)] == sum_a)

            for _m in range(m):
                for _k in range(k):
                    sum_b = 0
                    for _n in range(n):
                        for _b in range(_M):
                            sum_b += vB[(_b, _m, _n, _k)] * math.pow(2, _b)
                            model.constraints.add(
                                vB[(_b, _m, _n, _k)] <= bitcB[(_b, _m, _n)]
                            )
                            model.constraints.add(
                                vB[(_b, _m, _n, _k)] <= self.u[_n][_k]
                            )
                            model.constraints.add(
                                vB[(_b, _m, _n, _k)]
                                >= bitcB[(_b, _m, _n)] + self.u[_n][_k] - 1
                            )

                    model.constraints.add(fB[(_m, _k)] == sum_b)

            for _n in range(n):
                for _k in range(k):
                    _sum = 0
                    for _m in range(m):
                        for _b in range(_M):
                            _sum += bitcA[(_b, _m, _n)] + bitcB[(_b, _m, _n)]
                    model.constraints.add(_sum >= self.u[_n][_k])

        if (mode_t == "FULL") or (d > 0 and mode_t == "CARCH"):
            for _m in range(m):
                cluster_id = f_a.index[_m]
                # upper bound for solver
                ub = max(sum(copy_numbers.get(cluster_id, (0, 0))), cn_max)

                for _n in range(n):
                    sum_a = 0
                    sum_b = 0
                    for _b in range(_M):
                        sum_a += bitcA[(_b, _m, _n)] * math.pow(2, _b)
                        sum_b += bitcB[(_b, _m, _n)] * math.pow(2, _b)

                    model.constraints.add(self.cA[_m][_n] == sum_a)
                    model.constraints.add(self.cB[_m][_n] == sum_b)
                    model.constraints.add(self.cA[_m][_n] + self.cB[_m][_n] <= ub)

        if mode_t == "CARCH":
            # TODO: These loops can be collapsed once validation against C++ is complete
            for _m in range(m):
                for _k in range(k):
                    _sumA = 0
                    _sumB = 0
                    for _n in range(n):
                        if self._fixed_u[_n][_k] >= self.mu - self.tol:
                            _sumA += self.cA[_m][_n] * self._fixed_u[_n][_k]
                            _sumB += self.cB[_m][_n] * self._fixed_u[_n][_k]
                    model.constraints.add(fA[(_m, _k)] == _sumA)
                    model.constraints.add(fB[(_m, _k)] == _sumB)

                cluster_id = f_a.index[_m]
                # upper bound for solver
                ub = max(sum(copy_numbers.get(cluster_id, (0, 0))), cn_max)
                for _n in range(n):
                    model.constraints.add(self.cA[_m][_n] + self.cB[_m][_n] <= ub)

        if mode_t in ("FULL", "CARCH"):
            for _m in range(m):
                model.constraints.add(self.cA[_m][0] == 1)
                model.constraints.add(self.cB[_m][0] == 1)

            if ampdel:
                for _m in range(m):
                    cluster_id = f_a.index[_m]
                    if cluster_id not in copy_numbers:
                        for _n in range(1, n):
                            model.constraints.add(
                                self.cA[_m][_n]
                                <= cn_max * adA[_m] + _base - _base * adA[_m]
                            )
                            model.constraints.add(self.cA[_m][_n] >= _base * adA[_m])
                            model.constraints.add(
                                self.cB[_m][_n]
                                <= cn_max * adB[_m] + _base - _base * adB[_m]
                            )
                            model.constraints.add(self.cB[_m][_n] >= _base * adB[_m])

        if mode_t == "UARCH":
            # TODO: These loops can be collapsed once validation against C++ is complete
            for _m in range(m):
                for _k in range(k):
                    _sumA = 0
                    _sumB = 0
                    for _n in range(n):
                        _sumA += int(self._fixed_cA[_m][_n]) * self.u[_n][_k]
                        _sumB += int(self._fixed_cB[_m][_n]) * self.u[_n][_k]
                    model.constraints.add(fA[(_m, _k)] == _sumA)
                    model.constraints.add(fB[(_m, _k)] == _sumB)

        if mode_t in ("FULL", "UARCH"):
            for _k in range(k):
                _sum = 0
                for _n in range(n):
                    _sum += self.u[_n][_k]
                model.constraints.add(_sum == 1)

        if (mode_t in ("FULL", "UARCH")) and self.mu > 0:
            for _k in range(k):
                for _n in range(1, n):
                    model.constraints.add(x[(_n, _k)] >= self.u[_n][_k])
                    model.constraints.add(self.u[_n][_k] >= self.mu * x[(_n, _k)])

        # buildOptionalConstraints
        if (mode_t in ("FULL", "CARCH")) and d > 0:
            for _m in range(self.m):
                for _n in range(1, self.n):
                    _sum = 0
                    for _d in range(self.d):
                        _sum += z[(_m, _n, _d)]
                    model.constraints.add(_sum == 1)

            for _m in range(self.m):
                for _M in range(self.M):
                    for _d in range(d):
                        for _i in range(1, self.n - 1):
                            for _j in range(1, self.n):
                                model.constraints.add(
                                    bitcA[(_M, _m, _i)] - bitcA[(_M, _m, _j)]
                                    <= 2 - z[(_m, _i, _d)] - z[(_m, _j, _d)]
                                )
                                model.constraints.add(
                                    bitcA[(_M, _m, _j)] - bitcA[(_M, _m, _i)]
                                    <= 2 - z[(_m, _i, _d)] - z[(_m, _j, _d)]
                                )
                                model.constraints.add(
                                    bitcB[(_M, _m, _i)] - bitcB[(_M, _m, _j)]
                                    <= 2 - z[(_m, _i, _d)] - z[(_m, _j, _d)]
                                )
                                model.constraints.add(
                                    bitcB[(_M, _m, _j)] - bitcB[(_M, _m, _i)]
                                    <= 2 - z[(_m, _i, _d)] - z[(_m, _j, _d)]
                                )

            for _m in range(self.m):
                for _d in range(d - 1):
                    _sum_l = _sum_l1 = 0
                    for _n in range(1, self.n):
                        _sum_l += z[(_m, _n, _d)] * self.symmCoeff(_n)
                        _sum_l1 += z[(_m, _n, _d + 1)] * self.symmCoeff(_n)
                        model.constraints.add(_sum_l <= _sum_l1)

        if mode_t in ("FULL", "CARCH"):
            self.build_symmetry_breaking(model)
            self.fix_given_cn(model)

        if mode_t == "FULL":
            self.hot_start()

        obj = 0
        for _m in range(m):
            cluster_id = self.cluster_ids[_m]
            n_bins = len(self.binsA[cluster_id])

            for _k in range(k):
                for _i in range(n_bins):
                    obj += (yA[(_m, _k, _i)] + yB[(_m, _k, _i)]) * self.lengths[
                        cluster_id
                    ][_i]

        model.obj = pe.Objective(expr=obj, sense=pe.minimize)
        self.model = model

        if pprint:
            print(str(self))
