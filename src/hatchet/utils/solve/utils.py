import os
import numpy as np
import pandas as pd

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
    samples: list,
    clusters: list,
    baf: pd.DataFrame,
    rdr: pd.DataFrame,
    fcn: pd.DataFrame,
    f_a: pd.DataFrame,
    f_b: pd.DataFrame,
    weights: pd.Series,
):
    with open(out_file, "w") as fd:
        fd.write("CLUSTER\tSAMPLE\tBAF\tRDR\tFCN\tF_A\tF_B\tweight\n")
        for sample in samples:
            for cid in clusters:
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
    cluster_ids: pd.Index,
    sample_ids: pd.Index,
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
    assert solve_mode in ["cd", "ilp", "both"] and tempdir != None
    cluster_ids = cluster_ids.tolist()
    sample_ids = sample_ids.tolist()
    fd2_header = f"CLUSTER\tSAMPLE\tbaf\tfcn\texp-baf\texp-fcn\tcn_normal\tu_normal\t"
    fd2_header += "\t".join(f"cn_clone{i}\tu_clone{i}" for i in range(1, n)) + "\n"
    with open(f"{tempdir}/{solve_mode}_objs.tsv", "w") as fd1:
        fd1.write("sol_id\tobjective\n")
        # TODO also write the sub-objective values
        for i, obj in enumerate(sorted(result.keys())):
            fd1.write(f"{i}\t{obj}\n")
            with open(f"{tempdir}/{solve_mode}_sol{i}.tsv", "w") as fd2:
                fd2.write(fd2_header)
                cA, cB, u = result[obj]
                for ci, cid in enumerate(cluster_ids):
                    for si, sample in enumerate(sample_ids):
                        fcn = f_a.loc[cid, sample] + f_b.loc[cid, sample]
                        row = f"{cid}\t{sample}\t{baf.loc[cid, sample]}\t{fcn}\t"

                        exp_fcn = 0.0
                        exp_bcount = 0.0
                        for oi in range(n):
                            exp_fcn += (cA[ci][oi] + cB[ci][oi]) * u[oi][si]
                            exp_bcount += cB[ci][oi] * u[oi][si]
                        exp_baf = -1
                        if exp_fcn != 0:
                            exp_baf = exp_bcount / exp_fcn
                        row += f"{exp_baf}\t{exp_fcn}"
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
