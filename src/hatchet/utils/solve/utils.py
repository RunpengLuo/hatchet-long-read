import os
import numpy as np
import pandas as pd


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
    sub_obj = 0.0
    if pname == "MAXCN":
        sub_obj = compute_obj_MAXCN(w_, fA_, fB_, cA_, cB_, u_)
    elif pname == "DROOT_SUM":
        sub_obj = compute_obj_DROOT_SUM(w_, fA_, fB_, cA_, cB_, u_)
    elif pname == "DADJ":
        sub_obj = -1
    else:
        pass
    return [imf_obj, sub_obj]


def compute_obj_IMF(weights, fA, fB, cA, cB, u):
    """
    compute weighted IMF objective
    """
    leftA_w = weights * np.abs(fA - cA @ u)
    leftB_w = weights * np.abs(fB - cB @ u)
    obj = np.sum(leftA_w) + np.sum(leftB_w)
    return obj


# TODO
def compute_obj_DROOT_SUM(weights, fA, fB, cA, cB, u):
    """
    compute weighted DROOT objective

    DROOT: hamming distance between (a,b) and (1,1), for tumor clones, per cluster
    DADJ: hamming distance between (a,b) and (a',b'), for all clones, per cluster
    """
    (m, n) = cA.shape
    assert n > 1, "at least one tumor clone is needed"
    distA = weights * np.abs(cA[:, 1:] - cA[:, :1])
    distB = weights * np.abs(cB[:, 1:] - cB[:, :1])
    obj = np.sum(distA) + np.sum(distB)
    return obj


def compute_obj_MAXCN(weights, fA, fB, cA, cB, u):
    """
    compute weighted max cn-state objective
    """
    maxA_w = np.dot(np.max(cA[:, 1:], axis=1), weights)[0]
    maxB_w = np.dot(np.max(cB[:, 1:], axis=1), weights)[0]
    return maxA_w + maxB_w
