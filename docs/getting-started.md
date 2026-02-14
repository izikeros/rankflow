# Getting Started

## Installation

### Basic Installation

```bash
pip install rankflow
```

### With Optional Dependencies

```bash
# For pandas DataFrame support
pip install rankflow[pandas]

# For interactive plotly visualizations
pip install rankflow[interactive]

# For all optional dependencies
pip install rankflow[all]
```

### Development Installation

```bash
git clone https://github.com/izikeros/rankflow.git
cd rankflow
uv sync --group dev
```

## Quick Start

### From pandas DataFrame

```python
import pandas as pd
import matplotlib.pyplot as plt
from rankflow import RankFlow

# Create ranking data: columns are documents, rows are steps
data = {
    "Doc 1": [2, 1, 3, 2],
    "Doc 2": [1, 2, 1, 3],
    "Doc 3": [3, 3, 2, 1]
}
df = pd.DataFrame(data, index=["Search", "Rerank", "Filter", "Final"])

# Create and plot
rf = RankFlow(df=df)
rf.plot()
plt.show()
```

### From numpy arrays

```python
import numpy as np
from rankflow import RankFlow

ranks = np.array([
    [0, 1, 2, 3],  # Step 1 rankings
    [2, 0, 3, 1],  # Step 2 rankings
    [1, 2, 0, 3],  # Step 3 rankings
])

rf = RankFlow(
    ranks=ranks,
    step_labels=["Retrieval", "Rerank", "Final"],
    chunk_labels=["Doc A", "Doc B", "Doc C", "Doc D"],
)
rf.plot()
```

## Computing Metrics

```python
# Mark relevant documents for metric computation
rf = RankFlow(
    ranks=ranks,
    chunk_labels=["Doc A", "Doc B", "Doc C", "Doc D"],
    relevant_chunks=["Doc A", "Doc C"],
)

# Get metrics for each step
metrics = rf.metrics(k=3)
for step_idx, m in enumerate(metrics):
    print(f"Step {step_idx}: P@3={m['precision_at_k']:.2f}, MRR={m['mrr']:.2f}")
```

## Next Steps

- Check out the [Tutorials](tutorials/index.md) for detailed examples
- Browse the [API Reference](api.md) for complete documentation
