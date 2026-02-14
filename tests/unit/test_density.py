import numpy as np

from rankflow.core import RankFlow


def test_density_mode_runs():
    """Smoke test: density mode should not crash with many docs."""
    rng = np.random.RandomState(42)
    n_chunks = 100
    n_steps = 3
    ranks = np.zeros((n_steps, n_chunks), dtype=int)
    for s in range(n_steps):
        ranks[s] = rng.permutation(n_chunks)

    rf = RankFlow(
        ranks=ranks,
        step_labels=["BM25", "Semantic", "Cross-encoder"],
        chunk_labels=[f"Doc_{i}" for i in range(n_chunks)],
        relevant_chunks=["Doc_0", "Doc_1", "Doc_2"],
    )
    import matplotlib
    matplotlib.use("Agg")
    fig, _ax = rf.plot(mode="density")
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_density_mode_with_source_labels():
    """Smoke test: density mode with source provenance."""
    rng = np.random.RandomState(42)
    n_chunks = 50
    n_steps = 2
    ranks = np.zeros((n_steps, n_chunks), dtype=int)
    for s in range(n_steps):
        ranks[s] = rng.permutation(n_chunks)

    source = {}
    for i in range(n_chunks):
        if i < 20:
            source[f"Doc_{i}"] = "text"
        elif i < 40:
            source[f"Doc_{i}"] = "vector"
        else:
            source[f"Doc_{i}"] = "both"

    rf = RankFlow(
        ranks=ranks,
        chunk_labels=[f"Doc_{i}" for i in range(n_chunks)],
        source_labels=source,
    )
    import matplotlib
    matplotlib.use("Agg")
    fig, _ax = rf.plot(mode="density")
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)
