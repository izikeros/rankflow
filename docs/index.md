# rankflow

**Visualize and evaluate rank evolution across retrieval steps**

rankflow is a Python library for creating rank flow plots (bump charts), helping visualize changes in ranking of nodes across processing steps. It's particularly useful for RAG (Retrieval Augmented Generation) pipeline evaluation.

![RankFlow Example](https://raw.githubusercontent.com/izikeros/rankflow/main/img/rankflow_crop.png)

## Features

- **Rank Flow Plots**: Visualize how document rankings change across retrieval/re-ranking steps
- **Retrieval Metrics**: Compute P@K, R@K, MRR, NDCG, MAP for each step
- **Batch Evaluation**: Aggregate metrics across multiple queries
- **Multiple Backends**: matplotlib (static) and plotly (interactive)
- **Export Options**: DataFrame, dict, and JSON formats
- **Adapters**: Support for TREC, RAGAS, and ranx formats

## Quick Start

```python
import pandas as pd
from rankflow import RankFlow

# Create ranking data
data = {"Doc 1": [2, 1, 3, 2], "Doc 2": [1, 2, 1, 3], "Doc 3": [3, 3, 2, 1]}
df = pd.DataFrame(data, index=["Step_1", "Step_2", "Step_3", "Step_4"])

# Plot
rf = RankFlow(df=df)
rf.plot()
```

## Installation

```bash
pip install rankflow
```

With optional dependencies:

```bash
pip install rankflow[pandas]       # DataFrame support
pip install rankflow[interactive]  # Plotly backend
pip install rankflow[all]          # All extras
```

## Documentation

- [Getting Started](getting-started.md) - Installation and first steps
- [Tutorials](tutorials/index.md) - Step-by-step guides
- [API Reference](api.md) - Complete API documentation

## Links

- [GitHub Repository](https://github.com/izikeros/rankflow)
- [PyPI Package](https://pypi.org/project/rankflow/)
- [Blog Post: RankFlow plot for retriever visual evaluation](https://safjan.com/rankflow-plot-for-retriever-visual-evaluation/)

## License

MIT License - see [LICENSE](https://github.com/izikeros/rankflow/blob/main/LICENSE)

---
*Author: [Krystian Safjan](https://safjan.com)*
