import numpy as np
from matplotlib import pyplot as plt
from rankflow.main import RankFlow


def test_main():
    my_step_labels: list[str] = [
        "Hybrid Search",
        "Cross-encoder",
        "Graph-reranker",
        "Booster",
    ]
    my_chunk_labels: list[str] = [
        "Doc 0",
        "Doc 1",
        "Doc 2",
        "Doc 3",
        "Doc 4",
        "Doc 5",
        "Doc 6",
        "Doc 7",
        "Doc 8",
        "Doc 9",
    ]
    my_ranks = np.array(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [3, 0, 2, 4, 1, 6, 7, 9, 5, 8],
            [2, 3, 0, 4, 6, 1, 7, 8, 5, 9],
            [5, 3, 2, 1, 0, 4, 6, 7, 8, 9],
        ]
    )

    rf = RankFlow(
        ranks=my_ranks,
        step_labels=my_step_labels,
        chunk_labels=my_chunk_labels,
        title_font_size=24,
    )
    _ = rf.plot()
    # plt.savefig("../img/rankflow.png")
    plt.show()


def test_df_basic():
    import pandas as pd

    data = {"Doc 1": [2, 1, 3, 2], "Doc 2": [1, 2, 1, 3], "Doc 3": [3, 3, 2, 1]}
    df = pd.DataFrame(data, index=["Step_1", "Step_2", "Step_3", "Step_4"])

    rf = RankFlow(df=df)
    _ = rf.plot()
    # save
    plt.savefig("./rankflow_basic_pandas.png")
    plt.show()


def test_df_1_col__many_ranks():
    import pandas as pd

    data = {"A": [1, 2, 1, 3]}
    df = pd.DataFrame(data, index=["step_1", "step_2", "step_3", "step_4"])

    rf = RankFlow(df=df)
    _ = rf.plot()
    plt.show()


def test_df_1_col__1_rank():
    import pandas as pd

    data = {"A": [1, 1, 1, 1]}
    df = pd.DataFrame(data, index=["step_1", "step_2", "step_3", "step_4"])

    rf = RankFlow(df=df)
    _ = rf.plot()
    plt.show()



def test_df_two_steps():
    import pandas as pd

    data = gen_data(n_docs=3, n_steps=2)
    df = pd.DataFrame(data, index=["Step_1", "Step_2"])

    rf = RankFlow(df=df)
    _ = rf.plot()
    plt.show()


def gen_data(n_docs: int, n_steps):
    """Generate data for n_docs documents and n_steps steps.

    E.g. for 3 documents and 4 steps:
    data = {"Doc 1": [2, 1, 3, 2], "Doc 2": [1, 2, 1, 3], "Doc 3": [3, 3, 2, 1]}

    Note that the values in the rows are unique.
    use seed=42 for reproducibility.
    """
    np.random.seed(42)
    # create empty matrix
    data_matrix = np.zeros((n_steps, n_docs))
    # generate matrix where each row is a permutation of 1, 2, ..., n_docs
    for i in range(n_steps):
        data_matrix[i, :] = np.random.permutation(n_docs)
    # convert to integer
    data_matrix = data_matrix.astype(int)

    # shift ranks by +1
    data_matrix += 1

    # create dictionary with column names as keys and rows as values
    data = {f"Doc {i+1}": data_matrix[:, i] for i in range(n_docs)}
    return data


def test_df_many_rows():
    import pandas as pd

    # generate data for 1000 documents
    n_docs = 50
    n_steps = 3
    data = gen_data(n_docs=n_docs, n_steps=n_steps)
    df = pd.DataFrame(data, index=["Step_1", "Step_2", "Step_3"])

    my_ranks = df.to_numpy()
    my_step_labels = df.index.to_list()
    my_chunk_labels = df.columns.to_list()
    rf = RankFlow(
        ranks=my_ranks,
        step_labels=my_step_labels,
        chunk_labels=my_chunk_labels,
        title_font_size=24,
    )
    _ = rf.plot()
    plt.show()
