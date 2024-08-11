Quick Start
***********


.. code-block:: python

    >>> from tabulight import EDA
    >>> from tabulight import wq_data

    >>> data = wq_data()
    >>> data.shape
    
    >>> eda = EDA(data)

    >>> _ = eda.heatmap()

    >>> _ = eda.plot_missing()

    >>> _ = eda.plot_histograms()

    >>> _ = eda.box_plot(max_features=18, palette="Set3")
