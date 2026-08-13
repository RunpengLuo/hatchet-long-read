# TODO

_Last updated: 2026-07-19_

- **Post-merge step after HMM decoding.** ICL/BIC scoring regularized likelihoods by parameter counts and posterior entropy but under-penalize when sample size is very large (the data log-likelihood scales with N while the penalty scales with log N). Thus, the model selected cluster results lead to over segmentation. A post-HMM merging strategy to merge statistically indistinguishable clusters may solve this problem.
- **Joint CN deconvolution and CNT tree inference.** The prototype `cnt_cd` mode (per-chromosome copy-number-tree MILP C-step alternating with a global U-step LP over enumerated clone-tree topologies) was removed from the tree: it was the throughput bottleneck (enumerating clone-tree topologies x Dirichlet seeds and re-solving a per-chromosome MILP each iteration, scaling poorly with clone number and genome size) and needs tree-space pruning, MILP warm-starting/relaxation, or a cheaper C-step before it is practical. The full implementation (`cnt_model.py`, `cnt_tree.py`, `cnt_distance.py`, segment-mode `build_data`, and its wiring) is recoverable from git history at commit `3f97f73e` for a future redesign.
