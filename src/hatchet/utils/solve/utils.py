import os
import numpy as np
import pandas as pd
import kneed
import matplotlib.pyplot as plt

import hatchet.utils.Supporting as sp

# A list of random states, used as a stack
random_states = []


class Random:
    """
    A context manager that pushes a random seed to the stack for reproducible results,
    and pops it on exit.
    """

    def __init__(self, seed=None):
        self.seed = seed

    def __enter__(self):
        if self.seed is not None:
            # Push current state on stack
            random_states.append(np.random.get_state())
            new_state = np.random.RandomState(self.seed)
            np.random.set_state(new_state.get_state())

    def __exit__(self, *args):
        if self.seed is not None:
            np.random.set_state(random_states.pop())


def store_solve_input(
    out_file: str,
    baf: pd.DataFrame,
    rdr: pd.DataFrame,
    fcn: pd.DataFrame,
    f_a: pd.DataFrame,
    f_b: pd.DataFrame,
    weights: pd.Series,
):
    cluster_ids = f_a.index.tolist()
    sample_ids = f_a.columns.tolist()
    with open(out_file, "w") as fd:
        fd.write("CLUSTER\tSAMPLE\tBAF\tRDR\tFCN\tF_A\tF_B\tweight\n")
        for sample in sample_ids:
            for cid in cluster_ids:
                fd.write(
                    "\t".join(
                        [
                            str(cid),
                            str(sample),
                            str(baf.loc[cid, sample]),
                            str(rdr.loc[cid, sample]),
                            str(fcn.loc[cid, sample]),
                            str(f_a.loc[cid, sample]),
                            str(f_b.loc[cid, sample]),
                            str(weights[cid]),
                        ]
                    )
                    + "\n"
                )
        fd.close()
    return


def store_instance_tofile(
    result: dict,
    f_a: pd.DataFrame,
    f_b: pd.DataFrame,
    baf: pd.DataFrame,
    tempdir: str,
    solve_mode: str,
    n: int,
):
    """
    store temporary solution(s) from optimization.
    TODO add expected BAF and FCN from cn result as well to directly see fitness
    """
    assert tempdir != None
    cluster_ids = f_a.index.tolist()
    sample_ids = f_a.columns.tolist()
    fd2_header = f"CLUSTER\tSAMPLE\tbaf\texp-baf\tfcn\texp-fcn\tcn_normal\tu_normal\t"
    fd2_header += "\t".join(f"cn_clone{i}\tu_clone{i}" for i in range(1, n)) + "\n"
    with open(f"{tempdir}/{solve_mode}_objs.tsv", "w") as fd1:
        fd1.write("sol_id\tobjective\n")
        # TODO also write the sub-objective values
        for i, [obj, cA, cB, u] in result.items():
            fd1.write(f"{i}\t{obj}\n")
            with open(f"{tempdir}/{solve_mode}_sol{i}.tsv", "w") as fd2:
                fd2.write(fd2_header)
                for ci, cid in enumerate(cluster_ids):
                    for si, sample in enumerate(sample_ids):
                        fcn = f_a.loc[cid, sample] + f_b.loc[cid, sample]
                        row = f"{cid}\t{sample}\t"

                        exp_fcn = 0.0
                        exp_bcount = 0.0
                        for oi in range(n):
                            exp_fcn += (cA[ci][oi] + cB[ci][oi]) * u[oi][si]
                            exp_bcount += cB[ci][oi] * u[oi][si]
                        exp_baf = -1
                        if exp_fcn != 0:
                            exp_baf = exp_bcount / exp_fcn
                        row += f"{baf.loc[cid, sample]}\t{exp_baf}\t{fcn}\t{exp_fcn}"
                        for oi in range(n):
                            row += f"\t{cA[ci][oi]}|{cB[ci][oi]}\t{u[oi][si]}"
                        fd2.write(row + "\n")
                fd2.close()
        fd1.close()
    return


def compute_individual_objs(
    pname: str,
    weights: pd.Series,
    fA: pd.DataFrame,
    fB: pd.DataFrame,
    cA: list,
    cB: list,
    u: list,
):
    """
    Compute individual objectives from scalarized solution
    """
    w_ = weights.to_numpy().reshape((len(weights), 1))
    fA_ = fA.to_numpy()
    fB_ = fB.to_numpy()
    cA_ = np.array(cA)
    cB_ = np.array(cB)
    u_ = np.array(u)

    imf_obj = compute_obj_IMF(w_, fA_, fB_, cA_, cB_, u_)
    reg_objs = {
        "MAXCN": compute_obj_MAXCN,
        "DROOT_SUM": compute_obj_DROOT_SUM,
        "DADJ_SUM": compute_obj_DADJ_SUM,
    }
    sub_obj = reg_objs[pname](w_, fA_, fB_, cA_, cB_, u_)
    return [imf_obj, sub_obj]


def compute_obj_IMF(weights, fA, fB, cA, cB, u):
    """
    compute weighted IMF objective
    """
    leftA_w = weights * np.abs(fA - cA @ u)
    leftB_w = weights * np.abs(fB - cB @ u)
    obj = np.sum(leftA_w) + np.sum(leftB_w)
    return obj


def compute_obj_DROOT_SUM(weights, fA, fB, cA, cB, u):
    """
    DROOT: hamming distance between (a,b) and (1,1), for tumor clones, per cluster
    """
    distA = weights * np.abs(cA[:, 1:] - cA[:, :1])
    distB = weights * np.abs(cB[:, 1:] - cB[:, :1])
    obj = np.sum(distA) + np.sum(distB)
    return obj


# TODO
def compute_obj_DADJ_SUM(weights, fA, fB, cA, cB, u):
    """
    DADJ: hamming distance between (a,b) and (a',b'), for all clones, per cluster
    """
    obj = 0
    (m, n) = cA.shape
    for _m in range(m):
        obj_m = 0.0
        for _n1 in range(n - 1):
            for _n2 in range(_n1 + 1, n):
                obj_m += abs(cA[_m, _n1] - cA[_m, _n2])
                obj_m += abs(cB[_m, _n1] - cB[_m, _n2])
        obj += weights[_m, 0] * obj_m
    return obj


def compute_obj_MAXCN(weights, fA, fB, cA, cB, u):
    """
    MAXCN: weighted sum of cn-state per clsuter
    """
    maxA_w = np.dot(np.max(cA[:, 1:], axis=1), weights)[0]
    maxB_w = np.dot(np.max(cB[:, 1:], axis=1), weights)[0]
    return maxA_w + maxB_w


def model_selection_instance(
    f_a: pd.DataFrame,
    f_b: pd.DataFrame,
    weights: pd.Series,
    instances: dict,
    pname: str,
    solve_mode: str,
    outdir: str,
    verbose=False,
):
    """
    use elbow criterion to select best instance from either
    1) ILP or CD+ILP with scalaried solutions, or
    2) CD only solutions

    if solve_mode != cd, float-number error with be estimated.
    """
    assert len(instances) > 0, "ERROR! there is no solution to be selected"

    if pname == "RAW" or len(instances) == 1:
        return instances[0], instances[0][0]

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
    if outdir != None:
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
                msg=f"Failed to locate result in model selection for {solve_mode}, use non-penalized result\n",
                level="WARN",
            )
        else:
            # multiple instance may yield same objective values, pick the one with minimum penalty
            sol_index = sol_indices[0]
            if verbose:
                sp.log(
                    msg=f"Model selection found solution with index={sol_index} for {solve_mode}!\n",
                    level="INFO",
                )
    df.loc[:, "selected"] = ""
    df.loc[sol_index, "selected"] = "*"
    if outdir != None:
        df.to_csv(
            f"{outdir}/model_selections.{solve_mode}.{pname}.tsv",
            sep="\t",
            header=True,
            index=False,
        )

    return instances[df.loc[sol_index, "Lambda"]], df.loc[sol_index, "IMF-objective"]
