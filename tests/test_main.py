
import os
import site
import unittest

import matplotlib.pyplot as plt
import pandas as pd

tabulight_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
site.addsitedir(tabulight_dir)

import numpy as np

from tabulight import EDA
from tabulight.utils import pac_yw, auto_corr, ccovf_np, ccf_np
from tabulight import wq_data


beach_data = wq_data()

pcp = beach_data['pcp_mm']


def call_methods(eda_obj:EDA):

    eda_obj.plot_histograms()
    eda_obj.plot_data()
    eda_obj.plot_ecdf()
    eda_obj.plot_missing()
    eda_obj.plot_index()
    eda_obj.box_plot()
    eda_obj.box_plot(violen=True)
    eda_obj.lag_plot(n_lags=2)
    plt.close('all')
    eda_obj.probability_plots()
    plt.close('all')
    #eda_obj.heatmap()
    eda_obj.data_availability()
    eda_obj.grouped_scatter()

    if isinstance(eda_obj.data, pd.DataFrame) and eda_obj.data.shape[1]>1:
        for method in ['pearson', 'spearman', 'covariance', 'kendall']:
            eda_obj.correlation(method=method)
            plt.close('all')

    eda_obj.autocorrelation(n_lags=3)
    eda_obj.partial_autocorrelation(n_lags=3)

    for method in ["kolmogorov", "shapiro", "anderson"]:
        eda_obj.normality_test(method)

    eda_obj.stats()
    eda_obj.parallel_corrdinates(st=0, en=100)
    plt.close('all')
    eda_obj.show_unique_vals()
    return


class TestEDA(unittest.TestCase):

    show=False
    save=False

    def test_series(self):
        eda = EDA(data=pcp, save=self.save, show=self.show)
        call_methods(eda)
        return

    def test_dataframe(self):
        eda = EDA(data=beach_data.iloc[:-3:], dpi=50, save=self.save,
                  show=self.show)
        call_methods(eda)
        return

    def test_with_input_features(self):
        eda = EDA(
            data=beach_data.iloc[:, 0:3], in_cols=beach_data.columns.to_list()[0:3],
            dpi=50,
            save=self.save,
            show=self.show)
        call_methods(eda)
        return

    def test_with_output_features(self):
        data = wq_data(inputs=['tide_cm', 'wat_temp_c'])
        eda = EDA(
            data=data, out_cols=data.columns.to_list()[-1:],
            dpi=50,
            save=self.save,
            show=self.show)
        call_methods(eda)
        return

    def test_with_input_output_features(self):
        data = wq_data(inputs=['tide_cm', 'wat_temp_c'])
        eda = EDA(
            data=data,
            in_cols=data.columns.to_list()[0:-1],
            out_cols=data.columns.to_list()[-1:],
            dpi=50,
            save=self.save,
            show=self.show)
        call_methods(eda)
        return

    def test_autocorr_against_statsmodels(self):
        try:
            from statsmodels.tsa.stattools import acf, pacf
        except (ModuleNotFoundError, ImportError):
            acf, pacf = None, None

        if acf is not None:
            a = np.sin(np.linspace(0, 20, 100))

            np.testing.assert_allclose(acf(a, nlags=4), auto_corr(a, 4))
            np.testing.assert_allclose(pacf(a, nlags=4), pac_yw(a, 4))

        return

    def test_cross_corr_against_statsmodels(self):
        try:
            from statsmodels.tsa.stattools import ccf, ccovf
        except (ModuleNotFoundError, ImportError):
            ccf, ccovf = None, None

        if ccf is not None:
            a = np.linspace(0, 40, 100)
            b = np.sin(a)
            c = np.cos(b)

            np.allclose(ccf(b, c), ccf_np(b, c))

            np.allclose(ccovf_np(b, c), ccovf(b, c).sum())

        return

    def test_ndarray(self):
        # create a random state and generate random array from it
        rng = np.random.RandomState(0)
        eda = EDA(rng.randn(100, 3), save=self.save, show=self.show)
        call_methods(eda)
        assert isinstance(eda.data, pd.DataFrame)
        return

    def test_pie(self):
        data = pd.DataFrame(np.random.randint(1, 11, (100, 3)), columns=['a', 'b', 'c'])
        eda = EDA(data, save=self.save, show=self.show)
        #call_methods(eda) todo: fix
        eda.show_unique_vals(cols=['a'])
        return


if __name__ == "__main__":

    unittest.main()