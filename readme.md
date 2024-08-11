Exploratory data analysis for tabular and 1-D time series data.

# Installation

using GitHub link for the latest code

	python -m pip install git+https://github.com/AtrCheema/tabulight.git

# Usage

```python
from tabulight import EDA
from tabulight import wq_data
data = wq_data()
print(data.shape)
eda = EDA(data)
_ = eda.heatmap()
_ = eda.plot_missing()
_ = eda.plot_histograms()
_ = eda.box_plot(max_features=18, palette="Set3")
```