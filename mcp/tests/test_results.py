"""Summaries and figure listings parsed from HATCHet stage outputs."""

from hatchet_mcp.results import (
    Previews,
    check_results,
    cluster_bins_results,
    compute_cn_results,
    pdf_page_count,
    selected_k,
)


def _tsv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\t".join(header) + "\n" + "".join("\t".join(map(str, r)) + "\n" for r in rows))


def _pdf(run_dir, rel, pages, preview_root):
    """A stand-in PDF with ``pages`` page objects, plus the previews a stage renders."""
    path = run_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.4\n" + b"<< /Type /Page >>\n" * pages + b"<< /Type /Pages >>\n")
    out = preview_root / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    for i in range(1, pages + 1):
        out.with_name(f"{out.stem}-{i}.png").write_bytes(b"png")
    return path


def test_page_count_and_numeric_page_order(tmp_path):
    run, prev = tmp_path / "run", tmp_path / "previews"
    pdf = _pdf(run, "f.pdf", 11, prev)
    assert pdf_page_count(pdf) == 11
    assert [p.stem for p in Previews(run, prev).pages(pdf)][-2:] == ["f-10", "f-11"]


def test_cluster_bins_summary_and_figures(tmp_path):
    run, prev = tmp_path / "run", tmp_path / "previews"
    bbc = run / "cluster-bins"
    _tsv(
        bbc / "model_scores.tsv",
        ["K", "restart_it", "ll", "bic", "icl"],
        [[3, 0, -10, 50, 51], [3, 1, -9, 49, 52], [4, 0, -8, 60, 61]],
    )
    _tsv(
        bbc / "bulk.seg",
        ["CLUSTER", "SAMPLE", "#BINS", "#SNPS", "LENGTH", "RD", "BAF"],
        [[0, "T1", 5, 20, 5000, 1.0, 0.5], [1, "T1", 3, 9, 3000, 1.4, 0.36]],
    )
    _tsv(
        bbc / "bulk.bbc",
        ["#CHR", "START", "END", "SAMPLE", "CLUSTER"],
        [["chr1", 0, 10, "T1", 0], ["chr1", 0, 10, "T2", 0], ["chr1", 10, 20, "T1", 1]],
    )
    _pdf(run, "cluster-bins/bulk.K3.pdf", 2, prev)
    _pdf(run, "cluster-bins/plots/model_scores.pdf", 1, prev)
    _pdf(run, "cluster-bins/plots/elbo_traces.pdf", 3, prev)
    _pdf(run, "cluster-bins/plots/K3.diagnostic.pdf", 3, prev)

    summary, figures = cluster_bins_results(str(run), str(prev))
    assert summary["selected_k"] == 3 == selected_k(bbc)
    assert summary["model_scores_best_per_k"][0] == {"K": 3, "ll": -9.0, "bic": 49.0, "icl": 51.0}
    assert summary["n_clusters_after_filtering"] == 2 and summary["n_bins_clustered"] == 2

    inline = [f for f in figures if f["inline"]]
    assert [f["description"].split(" (")[0] for f in inline] == [
        "Model selection across K",
        "Model-selected clustering, K=3",
        "QC: EM convergence per K",
    ]
    assert len(inline[1]["preview_png"]) == 2 and len(inline[2]["preview_png"]) == 1
    assert all(p.startswith(str(prev)) for f in inline for p in f["preview_png"])
    other = [f for f in figures if not f["inline"]]
    assert [f["description"] for f in other] == ["K3.diagnostic"]
    assert other[0]["pages"] == 3 and "preview_png" not in other[0]


def test_compute_cn_summary_and_figures(tmp_path):
    run, prev = tmp_path / "run", tmp_path / "previews"
    res = run / "compute-cn"
    (res / "plots").mkdir(parents=True)
    (res / "selected_ploidy.txt").write_text("tetraploid\n")
    _tsv(
        res / "summary.tsv",
        ["instance_id", "ploidy", "n_clones", "selected"],
        [["p0", "diploid", 3, "*"], ["p0", "tetraploid", 3, "tetraploid"]],
    )
    _tsv(
        res / "best.seg.ucn",
        ["#CHR", "START", "END", "SAMPLE", "CLUSTER"]
        + ["cn_normal", "u_normal", "cn_clone1", "u_clone1", "cn_clone2", "u_clone2"],
        [
            ["chr1", 0, 100, "T1", 0, "1|1", 0.2, "2|2", 0.5, "2|1", 0.3],
            ["chr1", 100, 400, "T1", 1, "1|1", 0.2, "2|2", 0.5, "2|2", 0.3],
            ["chr1", 0, 100, "T2", 0, "1|1", 0.4, "2|2", 0.1, "2|1", 0.5],
        ],
    )
    _pdf(run, "compute-cn/plots/model_selection.pdf", 3, prev)
    _pdf(run, "compute-cn/plots/scaling_2d.tetraploid.pdf", 1, prev)
    _pdf(run, "compute-cn/plots/tetraploid_n3/S.tetraploid_n3.1D.pdf", 1, prev)
    for name in ("S.1D.pdf", "S.1D.FCN_AB.pdf", "S.2D.pdf"):
        _pdf(run, f"plot-cn/{name}", 1, prev)

    summary, figures = compute_cn_results(str(run), str(prev))
    assert summary["selected_ploidy"] == "tetraploid" and summary["wgd"] is True
    assert summary["n_tumor_clones"] == 2 and summary["hatchet_n_clones"] == 3
    assert summary["samples"]["T1"]["purity"] == 0.8
    assert summary["samples"]["T2"]["u_clone2"] == 0.5
    assert summary["clone_ploidy_length_weighted"] == {"clone1": 4.0, "clone2": 3.75}

    inline = [f for f in figures if f["inline"]]
    assert inline[0]["description"] == "Model selection (compute-cn)"
    assert inline[0]["preview_png"][0].endswith("model_selection-3.png")
    assert [f["description"].split(" ")[0] for f in inline[1:]] == ["1D", "1D", "2D", "QC:"]
    assert all(f["pdf"].endswith(".pdf") for f in figures)
    other = [f["description"] for f in figures if not f["inline"]]
    assert other == ["S.tetraploid_n3.1D"]


def test_check_results_parsing():
    log = "x\n  [PASS] a  ok\n  [FAIL] b  bad\nhatchet check FAILED (1 failing)\n"
    parsed = check_results(log)
    assert (parsed["n_pass"], parsed["n_fail"], parsed["passed"]) == (1, 1, False)
