from collections import OrderedDict
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


def parse_clonal(clonal):
    copy_numbers = OrderedDict()  # dict from cluster_id => (cn_a, cn_b) 2-tuple
    clonal_parts = clonal.split(",")
    n_clonal_parts = len(clonal_parts)

    for i, c in enumerate(clonal_parts):
        cluster_id, cn_a, cn_b = [int(_c) for _c in c.split(":")]

        # The first two clonal clusters (used for scaling) MUST have different total copy numbers
        if i == 1:
            _first_cn_a, _first_cn_b = list(copy_numbers.values())[0]
            if _first_cn_a + _first_cn_b == cn_a + cn_b:
                raise ValueError(
                    (
                        "When >= 2 clonal copy numbers are given, the first two must be different in the two segmental "
                        "clusters"
                    )
                )

        cn_total = cn_a + cn_b
        if (cn_total == 2) and (n_clonal_parts > 1):
            # warnings.warn('Please specify a single cluster when CN_A+CN_B=2')
            # TODO: The C++ implementation generates a warning corresponding to the above
            # This is suppressed by default, which is what we do here for now.
            pass
        if cluster_id in copy_numbers:
            raise ValueError("Already encountered cluster_id =", str(cluster_id))
        copy_numbers[cluster_id] = cn_a, cn_b

    return copy_numbers


def scale_rdr(rdr, copy_numbers, purity_tol=0.05):
    assert len(copy_numbers) >= 1
    if len(copy_numbers) == 1:
        diploid_cluster_id = list(copy_numbers.keys())[0]
        diploid_rdr = rdr.loc[diploid_cluster_id]
        scale = 2 / diploid_rdr
    else:
        # Use the first two copy-number specifications to determine scaling factors
        # Note the reversed assignment order to conform to C++ behavior - check with Simone!
        cluster_id_2, cluster_id_1 = tuple(copy_numbers.keys())[:2]  # TODO
        rdr_1, rdr_2 = rdr.loc[cluster_id_1], rdr.loc[cluster_id_2]
        cn_sum_1, cn_sum_2 = (
            sum(copy_numbers[cluster_id_1]),
            sum(copy_numbers[cluster_id_2]),
        )
        purity = 2 * (rdr_1 - rdr_2) / ((2 - cn_sum_2) * rdr_1 - (2 - cn_sum_1) * rdr_2)

        purity[(1 <= purity) & (purity <= 1 + purity_tol)] = 1
        purity[(-purity_tol <= purity) & (purity <= 0)] = 0

        scale = (2 - (2 - cn_sum_1) * purity) / rdr_1
        assert np.all((0 <= purity) & (purity <= 1) & (scale >= 0)), "scaling failed"

    return scale


def segmentation(
    cA,
    cB,
    u,
    cluster_ids,
    sample_ids,
    bbc_file,
    bbc_out_file=None,
    seg_out_file=None,
):
    df = pd.read_csv(bbc_file, sep="\t")
    # Chromosomes may or may not have chr notation - force a string dtype anyway
    df["#CHR"] = df["#CHR"].astype(str)
    # TODO: The legacy C++ implementation interprets the 'coverage' column as an int
    df["COV"] = df["COV"].astype(int)

    n_clone = len(cA[0])
    cA = pd.DataFrame(cA, index=cluster_ids, columns=range(n_clone))
    cB = pd.DataFrame(cB, index=cluster_ids, columns=range(n_clone))
    u = pd.DataFrame(u, index=range(n_clone), columns=sample_ids)
    # Make (n_sample, n_clone) in shape; easier to merge later
    u = u.T

    # copy-numbers represented as <CN_A>|<CN_B> strings
    cN = cA.astype(str) + "|" + cB.astype(str)
    cN.columns = ["cn_normal"] + [f"cn_clone{i}" for i in range(1, n_clone)]

    # Merge in copy-number + proportion information to our original Dataframe
    df = df.merge(cN, left_on="CLUSTER", right_index=True)
    u.columns = ["u_normal"] + [f"u_clone{i}" for i in range(1, n_clone)]
    df = df.merge(u, left_on="SAMPLE", right_index=True)

    # Sorting the values by start/end position critical for merging contiguous
    # segments with identical copy numbers later on
    df = df.sort_values(["#CHR", "START", "END", "SAMPLE"])
    df = df.reset_index(drop=True)

    # last 2*n_clone columns names = [cn_normal, u_normal, cn_clone1, u_clone1, cn_clone2, ...]
    extra_columns = [col for sublist in zip(cN.columns, u.columns) for col in sublist]
    all_columns = df.columns.values[: -2 * n_clone].tolist() + extra_columns

    if bbc_out_file is not None:
        # rearrange columns for easy comparison to legacy files
        df = df[all_columns]
        df.to_csv(bbc_out_file, sep="\t", index=False)

    if seg_out_file is not None:
        # create a new column that will use to store the contiguous segment number (1-indexed)
        df["segment"] = 0
        # all column names with cnA|cnB information (normal + clones)
        cN_column_names = cN.columns.tolist()
        # create a new column with all cnA|cnB strings joined as a single column
        df["all_copy_numbers"] = df[cN_column_names].apply(
            lambda x: ",".join(x), axis=1
        )

        _first_sample_name = df["SAMPLE"].iloc[0]

        # Grouping by consecutive identical values
        # See https://towardsdatascience.com/pandas-dataframe-group-by-consecutive-same-values-128913875dba
        group_name_to_indices = df.groupby(
            (
                # Find indices where we see the first sample name AND
                (df["SAMPLE"] == _first_sample_name)
                & (
                    # The chromosome changed values from the previous row OR
                    # any of the copy-numbers changed from the previous row OR
                    # the START changed from the END in the previous row
                    (df["#CHR"] != df["#CHR"].shift())
                    | (df["all_copy_numbers"] != df["all_copy_numbers"].shift())
                    | (df["START"] != df["END"].shift())
                )
            ).cumsum(),
            # cumulative sum increments whenever a True is encountered, thus creating a series of monotonically
            # increasing values we can use as segment numbers
            sort=False,
        ).indices
        # 'indices' of a Pandas GroupBy object gives us a mapping from the group 'name'
        # (numbers starting from 1) -> indices in the Dataframe

        for group_name, indices in group_name_to_indices.items():
            df.loc[indices, "segment"] = group_name

        aggregation_rules = {
            "#CHR": "first",
            "START": "min",
            "END": "max",
            "SAMPLE": "first",
        }
        aggregation_rules.update({c: "first" for c in extra_columns})
        df = df.groupby(["segment", "SAMPLE"]).agg(aggregation_rules)

        df.to_csv(seg_out_file, sep="\t", index=False)

def store_temp_result(result: dict, cluster_ids: pd.Index, sample_ids: pd.Index, 
                      f_a: pd.DataFrame, f_b: pd.DataFrame, baf: pd.DataFrame, 
                      tempdir: str, solve_mode: str, n: int):
    """
    store temporary solution(s) from optimization.
    TODO add expected BAF and FCN from cn result as well to directly see fitness
    """
    assert solve_mode in ["cd", "ilp", "both"] and tempdir != None
    cluster_ids = cluster_ids.tolist() 
    sample_ids = sample_ids.tolist()
    fd2_header = f"CLUSTER\tSAMPLE\tbaf\tfcn\texp-baf\texp-fcn\tcn_normal\tu_normal\t"
    fd2_header += '\t'.join(f"cn_clone{i}\tu_clone{i}" for i in range(1, n)) + '\n'
    with open(f"{tempdir}/{solve_mode}_objs.tsv", 'w') as fd1:
        fd1.write("sol_id\tobjective\n")
        # TODO also write the sub-objective values
        for i, obj in enumerate(sorted(result.keys())):
            fd1.write(f"{i}\t{obj}\n")
            with open(f"{tempdir}/{solve_mode}_sol{i}.tsv", 'w') as fd2:
                fd2.write(fd2_header)
                cA, cB, u = result[obj]
                for ci, cID in enumerate(cluster_ids):
                    for si, sample in enumerate(sample_ids):
                        fcn = f_a.loc[cID, sample] + f_b.loc[cID, sample]
                        row = f"{cID}\t{sample}\t{baf.loc[cID, sample]}\t{fcn}\t"

                        exp_fcn = 0.0
                        exp_bcount = 0.0
                        for oi in range(n):
                            exp_fcn += (cA[ci][oi] + cB[ci][oi]) * u[oi][si]
                            exp_bcount += cB[ci][oi] * u[oi][si]
                        exp_baf = exp_bcount / exp_fcn
                        row += f"{exp_baf}\t{exp_fcn}"
                        for oi in range(n):
                            row += f"\t{cA[ci][oi]}|{cB[ci][oi]}\t{u[oi][si]}"
                        fd2.write(row + '\n')
                fd2.close()
        fd1.close()
    return

def load_pre_config_txt(pre_config_txt: str):
    """
    load pre_config.txt for optimization \\
    example: \\
    MAXCN:30-0.01 <type-steps-step_size> one penalty term \\
    0.8,0.2,0.0;0.6,0.1,0.3  <uprop_i;> one per sample \\
    1:1|1,1|2; <segID:<cA|cB>,<cA|cB>;> one per cluster
    """
    problem_params = None
    purities_fixed = None
    copy_numbers_fixed = None
    with open(pre_config_txt, 'r') as fd:
        lines = fd.readlines()
        assert len(lines) == 3
        pstr = lines[0].strip()
        if len(pstr) != 0:
            pname, pval = pstr.split(":")
            assert pname in ["MAXCN", "DROOT", "DADJ"], "unsupported penalty term"
            steps, step_size = [float(p) for p in pval.split('-')]
            problem_params = [pname, int(steps), step_size]
        
        fixed_ps = lines[1].strip()
        if len(fixed_ps) != 0:
            purities_fixed = []
            for pstr in fixed_ps.split(';'):
                purities_fixed.append([float(p) for p in pstr.split(',')])
        
        fixed_cns = lines[2].strip()
        if len(fixed_cns) != 0:
            copy_numbers_fixed = {}
            for seg in fixed_cns.split(';'):
                segID, cns_str = seg.split(':')
                segID = int(segID)
                copy_numbers_fixed[segID] = []
                for cn_str in cns_str.split(','):
                    a, b = cn_str.split('|')
                    copy_numbers_fixed[segID].append((int(a), int(b)))
        fd.close()
    return problem_params, purities_fixed, copy_numbers_fixed

def compute_individual_objs(pname: str, weights: pd.Series, fA: pd.DataFrame, fB: pd.DataFrame, 
                            cA: list, cB: list, u: list):
    """
    Compute individual objectives from scalarized solution
    """
    w_  = weights.to_numpy().reshape((len(weights), 1))
    fA_ = fA.to_numpy()
    fB_ = fB.to_numpy()
    cA_ = np.array(cA)
    cB_ = np.array(cB)
    u_ = np.array(u)

    imf_obj = compute_obj_IMF(w_, fA_, fB_, cA_, cB_, u_)
    sub_obj = 0.0
    if pname == "MAXCN":
        sub_obj = compute_obj_MAXCN(w_, fA_, fB_, cA_, cB_, u_)
    elif pname == "DROOT":
        sub_obj = compute_obj_DROOT(w_, fA_, fB_, cA_, cB_, u_)
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
def compute_obj_DROOT(weights, fA, fB, cA, cB, u):
    """
    compute weighted DROOT objective

    DROOT: hamming distance between (a,b) and (1,1), for tumor clones, per cluster
    DADJ: hamming distance between (a,b) and (a',b'), for all clones, per cluster
    """
    (m, n) = cA.shape



    return 0

def compute_obj_MAXCN(weights, fA, fB, cA, cB, u):
    """
    compute weighted max cn-state objective
    """
    maxA_w = np.dot(np.max(cA[:, 1:], axis=1), weights)[0]
    maxB_w = np.dot(np.max(cB[:, 1:], axis=1), weights)[0]
    return maxA_w + maxB_w
