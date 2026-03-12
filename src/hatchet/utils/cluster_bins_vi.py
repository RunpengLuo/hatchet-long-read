from collections import Counter
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.special import logsumexp
from scipy.spatial.distance import pdist, squareform
from hmmlearn import hmm

from hatchet.utils.ArgParsing import parse_cluster_bins_args
import hatchet.utils.Supporting as sp

from hatchet.utils.cluster_bins import read_bb, make_transmat

def main(args=None):
    sp.log(msg="# Parsing and checking input arguments\n", level="STEP")
    args = parse_cluster_bins_args(args)
    sp.logArgs(args, 80)

    sp.log(msg="# Reading the combined BB file\n", level="STEP")
    tracks, bb, sample_labels, chr_labels = read_bb(
        args["bbfile"], subset=args["subset"], allow_gaps=args["allow_gaps"]
    )

    outdir = args["outbins"][:str.rindex(args["outbins"], "/")]


def hmm_vi_model_select(
        outdir: str,
        tracks: list, 
        minK: int,
        maxK: int,
        tau: float,
        tmat: str,
        decode_alg: float,
        covar: float,
        state_selection: float,
        restarts=10):
    assert tmat in ["fixed", "diag", "free"]
    assert decode_alg in ["map", "viterbi"]
    assert state_selection in ["silhouette", "bic"]

    scores_record = []
    # format input
    tracks = [a for a in tracks if a.shape[0] > 0 and a.shape[1] > 0]
    if len(tracks) > 1:
        X = np.concatenate(tracks, axis=1).T
        lengths = [a.shape[1] for a in tracks]
    else:
        X = tracks[0].T
        lengths = [tracks[0].shape[1]]

    best_K = 0
    if state_selection == "silhouette":
        best_score = -1.01  # below minimum silhouette score value
    else:
        best_score = np.inf  # BIC is always negative
    best_model = None
    best_labels = None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    C = squareform(pdist(X_scaled)) # default: euclidean

    rs = {}
    for K in range(minK, maxK + 1):
        my_best_ll = -1 * np.inf
        my_best_labels = None
        my_best_model = None
        for s in range(restarts):
            A = make_transmat(1 - tau, K)
            assert np.all(A > 0), (
                "Found 0 or negative elements in transition matrix."
                "This is likely a numerical precision issue -- try increasing tau.",
                A,
            )
            assert np.allclose(np.sum(A, axis=1), 1), (
                "Not all rows in transition matrix sum to 1.",
                A,
            )
