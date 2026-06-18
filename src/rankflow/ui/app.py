"""RankFlow Web UI -- main Streamlit application.

Launch with: rankflow ui ./experiments
Or directly: streamlit run src/rankflow/ui/app.py -- ./experiments
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

from rankflow.experiments import ExperimentStore


def _get_store_path() -> Path:
    """Get the experiment store path from CLI args or sidebar input."""
    # Check sys.argv for path (passed after --)
    for i, arg in enumerate(sys.argv):
        if arg == "--" and i + 1 < len(sys.argv):
            return Path(sys.argv[i + 1])
    # Check if last arg looks like a path (not a streamlit flag)
    if len(sys.argv) > 1 and not sys.argv[-1].startswith("-"):
        candidate = Path(sys.argv[-1])
        if candidate.is_dir():
            return candidate
    return Path("./experiments")


def main():
    st.set_page_config(
        page_title="RankFlow",
        page_icon="📊",
        layout="wide",
    )

    store_path = _get_store_path()
    store = ExperimentStore(store_path)

    st.sidebar.title("RankFlow")
    st.sidebar.caption(f"Store: `{store_path}`")

    page = st.sidebar.radio(
        "Navigate",
        ["Experiments", "Compare", "Query Explorer", "Deep Dive"],
    )

    if page == "Experiments":
        _page_experiments(store)
    elif page == "Compare":
        _page_compare(store)
    elif page == "Query Explorer":
        _page_query_explorer(store)
    elif page == "Deep Dive":
        _page_deep_dive(store)


# ------------------------------------------------------------------
# Page: Experiment List
# ------------------------------------------------------------------


def _page_experiments(store: ExperimentStore):
    st.header("Experiments")

    experiments = store.list()
    if not experiments:
        st.info(f"No experiments found in `{store.path}`. Save one with `ExperimentStore.save()`.")
        return

    tag_filter = st.sidebar.text_input("Filter by tag")
    if tag_filter:
        experiments = [e for e in experiments if tag_filter in e.get("tags", [])]

    # Build table data
    rows = []
    for exp in experiments:
        config_summary = ", ".join(f"{k}={v}" for k, v in list(exp["config"].items())[:3])
        rows.append({
            "Name": exp["name"],
            "Queries": exp["n_queries"],
            "Tags": ", ".join(exp.get("tags", [])),
            "Config": config_summary,
            "Timestamp": exp.get("timestamp", ""),
        })

    st.dataframe(rows, use_container_width=True)

    # Quick metrics preview
    selected = st.selectbox("Preview experiment", [e["name"] for e in experiments])
    if selected and st.button("Load metrics"):
        with st.spinner("Computing metrics..."):
            exp = store.load(selected)
            summary = exp.metrics_summary(k=10)
            if summary:
                cols = st.columns(len(summary))
                for col, (metric, value) in zip(cols, summary.items(), strict=True):
                    col.metric(metric.replace("_", " ").title(), f"{value:.3f}")
            else:
                st.warning("No metrics available (are relevant_chunks set?).")


# ------------------------------------------------------------------
# Page: Compare
# ------------------------------------------------------------------


def _page_compare(store: ExperimentStore):
    st.header("Experiment Comparison")

    experiments = store.list()
    names = [e["name"] for e in experiments]

    if len(names) < 2:
        st.info("Need at least 2 experiments to compare.")
        return

    col1, col2 = st.columns(2)
    baseline_name = col1.selectbox("Baseline", names, index=0)
    challenger_name = col2.selectbox("Challenger", names, index=min(1, len(names) - 1))

    k = st.sidebar.slider("K (top-K for metrics)", 1, 50, 10)

    if baseline_name == challenger_name:
        st.warning("Select two different experiments.")
        return

    if st.button("Compare"):
        from rankflow.comparison import compare_experiments

        with st.spinner("Comparing..."):
            baseline = store.load(baseline_name)
            challenger = store.load(challenger_name)
            report = compare_experiments(baseline, challenger, k=k)

        # Config diff
        if report.config_diff:
            st.subheader("Configuration Differences")
            diff_rows = []
            for key, vals in report.config_diff.items():
                diff_rows.append({
                    "Parameter": key,
                    "Baseline": str(vals["baseline"]),
                    "Challenger": str(vals["challenger"]),
                })
            st.table(diff_rows)

        # Metric deltas
        st.subheader("Metric Comparison")
        metric_cols = st.columns(len(report.metric_deltas))
        for col, (metric, data) in zip(metric_cols, report.metric_deltas.items(), strict=True):
            delta = data["delta"]
            p_val = data["p_value"]
            sig = "✓" if p_val < 0.05 else ""
            col.metric(
                metric.replace("_", " ").title(),
                f"{data['challenger_mean']:.3f}",
                delta=f"{delta:+.3f} {sig}",
            )

        # Win/Loss/Tie
        st.subheader("Win / Loss / Tie")
        wl_cols = st.columns(3)
        wl_cols[0].metric("Wins", report.wins)
        wl_cols[1].metric("Losses", report.losses)
        wl_cols[2].metric("Ties", report.ties)

        total = report.wins + report.losses + report.ties
        if total > 0:
            st.progress(report.wins / total, text=f"Win rate: {report.win_rate:.0%}")

        # Per-query table
        st.subheader("Per-Query Details")
        st.dataframe(report.per_query, use_container_width=True)


# ------------------------------------------------------------------
# Page: Query Explorer
# ------------------------------------------------------------------


def _page_query_explorer(store: ExperimentStore):
    st.header("Query Explorer")

    experiments = store.list()
    names = [e["name"] for e in experiments]
    if not names:
        st.info("No experiments found.")
        return

    selected = st.selectbox("Experiment", names)
    k = st.sidebar.slider("K", 1, 50, 10, key="qe_k")

    if not selected:
        return

    exp = store.load(selected)
    if not exp.rankflows:
        st.warning("Experiment has no queries.")
        return

    # Build query table
    query_data = []
    for i, rf in enumerate(exp.rankflows):
        label = getattr(rf, "query_label", None) or f"query_{i}"
        m = rf.metrics(k=k)
        final = m[-1] if m else {}
        query_data.append({
            "Query": label,
            "Steps": rf.ranks.shape[0],
            "Docs": rf.ranks.shape[1],
            "NDCG@K": f"{final.get('ndcg_at_k', 0):.3f}" if final else "N/A",
            "MRR": f"{final.get('mrr', 0):.3f}" if final else "N/A",
            "P@K": f"{final.get('precision_at_k', 0):.3f}" if final else "N/A",
        })

    st.dataframe(query_data, use_container_width=True)

    # Drill-down: select a query and show its rank flow plot
    query_labels = [getattr(rf, "query_label", None) or f"query_{i}" for i, rf in enumerate(exp.rankflows)]
    selected_query = st.selectbox("Select query for detail view", query_labels)

    if selected_query:
        idx = query_labels.index(selected_query)
        rf = exp.rankflows[idx]
        st.subheader(f"Rank Evolution: {selected_query}")

        import matplotlib

        matplotlib.use("Agg")
        fig, _ax = rf.plot()
        st.pyplot(fig)


# ------------------------------------------------------------------
# Page: Deep Dive (single experiment)
# ------------------------------------------------------------------


def _page_deep_dive(store: ExperimentStore):
    st.header("Experiment Deep Dive")

    experiments = store.list()
    names = [e["name"] for e in experiments]
    if not names:
        st.info("No experiments found.")
        return

    selected = st.selectbox("Experiment", names, key="dd_exp")
    k = st.sidebar.slider("K", 1, 50, 10, key="dd_k")

    if not selected:
        return

    exp = store.load(selected)
    batch = exp.as_batch()

    st.subheader("Configuration")
    if exp.config:
        st.json(exp.config)
    else:
        st.caption("No config metadata stored.")

    st.subheader("Metrics Dashboard")
    import matplotlib

    matplotlib.use("Agg")

    try:
        fig, _axes = batch.plot_dashboard(k=k)
        st.pyplot(fig)
    except ValueError as e:
        st.warning(str(e))

    st.subheader("Metric Evolution")
    try:
        fig, _ax = batch.plot_metric_evolution(k=k)
        st.pyplot(fig)
    except ValueError as e:
        st.warning(str(e))

    st.subheader("Win / Loss Analysis")
    wl = batch.win_loss_analysis(k=k)
    if wl:
        st.table(wl)
    else:
        st.caption("No win/loss data available.")

    st.subheader("Failure Cases")
    failures = batch.failure_cases(k=k, threshold=-0.05)
    if failures:
        st.dataframe(failures, use_container_width=True)
    else:
        st.success("No regression queries found (threshold=-0.05).")


if __name__ == "__main__":
    main()
