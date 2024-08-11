"""
===========================
beach water quality
===========================
"""

import os
import site

if __name__ == '__main__':
    wd_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath('__file__')))))
    wd_dir = os.path.dirname(os.path.dirname(os.path.realpath('__file__')))
    print(wd_dir)
    site.addsitedir(wd_dir)

import pandas as pd

from tabulight import EDA
from tabulight.utils import print_info

# sphinx_gallery_thumbnail_number = 7

print_info()
###########################################################


# %%

def busan_beach(target=None):
    fpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "arg_busan.csv")

    if os.path.exists(fpath):
        df = pd.read_csv(fpath, index_col="index")
    else:
        df = pd.read_csv(
            "https://raw.githubusercontent.com/AtrCheema/AI4Water/ec2a4a426673b11e3589b64cef9d7160b1de28d4/ai4water/datasets/arg_busan.csv",
                         index_col="index")
        df.to_csv(fpath, index=True, index_label="index")
    df.index = pd.to_datetime(df.index)

    default_inputs = ['tide_cm', 'wat_temp_c', 'sal_psu', 'air_temp_c', 'pcp_mm', 'pcp3_mm', 'pcp6_mm',
                      'pcp12_mm', 'wind_dir_deg', 'wind_speed_mps', 'air_p_hpa', 'mslp_hpa', 'rel_hum'
                      ]

    default_targets = [col for col in df.columns if col not in default_inputs]

    if not isinstance(target, list):
        if isinstance(target, str):
            target = [target]
    elif isinstance(target, list):
        pass
    else:
        target = default_targets

    assert isinstance(target, list)

    return df[default_inputs + target]

data = busan_beach(target=['ecoli', 'sul1_coppml', 'aac_coppml',
                           'tetx_coppml', 'blaTEM_coppml'])
print(data.shape)

###########################################################

data.head()

###########################################################

data.isna().sum()

###########################################################

data.isna().sum()

###########################################################

eda = EDA(data, save=False)

###########################################################

eda.heatmap()

###########################################################

# _ = eda.plot_missing()

###########################################################

# _ = eda.plot_data(subplots=True, max_cols_in_plot=20, figsize=(14, 20))
#
# ###########################################################

eda.plot_data(subplots=True, max_cols_in_plot=20, figsize=(14, 20),
              ignore_datetime_index=True)

###########################################################

_ = eda.plot_histograms()

###########################################################

eda.box_plot(max_features=18, palette="Set3")

###########################################################

eda.box_plot(max_features=18, palette="Set3", violen=True)

###########################################################

eda.correlation(figsize=(14, 14))

# ###########################################################
#
#
# eda.grouped_scatter(max_subplots=18)

###########################################################

_ = eda.autocorrelation(n_lags=15)

###########################################################

_ = eda.partial_autocorrelation(n_lags=15)

###########################################################

_ = eda.lag_plot(n_lags=14, s=0.4)

############################################################


_ = eda.plot_ecdf(figsize=(10, 14))

############################################################

eda.normality_test()