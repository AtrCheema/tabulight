
import os
import sys
import json
import datetime
import warnings
from types import FunctionType
from collections import abc as collections_abc
from typing import Any, Dict, Union

from scipy import linalg

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from ._backend import scipy, seaborn, statsmodels

if scipy is not None:
    from scipy.stats import skew, kurtosis, gmean, variation, hmean


def auto_corr(x, nlags, demean=True):
    """
    autocorrelation like statsmodels
    https://stackoverflow.com/a/51168178
    """

    var = np.var(x)

    if demean:
        x -= np.mean(x)

    corr = np.full(nlags+1, np.nan, np.float64)
    corr[0] = 1.

    for lag in range(1, nlags+1):
        corr[lag] = np.sum(x[lag:]*x[:-lag])/len(x)/var

    return corr


def pac_yw(x, nlags):
    """partial autocorrelation according to ywunbiased method"""

    pac = np.full(nlags+1, fill_value=np.nan, dtype=np.float64)
    pac[0] = 1.

    for lag in range(1, nlags+1):
        pac[lag] = ar_yw(x, lag)[-1]

    return pac


def ar_yw(x, order=1, adj_needed=True, demean=True):
    """Performs autoregressor using Yule-Walker method.
    Returns:
        rho : np array
        coefficients of AR
    """
    x = np.array(x, dtype=np.float64)

    if demean:
        x -= x.mean()

    n = len(x)
    r = np.zeros(order+1, np.float64)
    r[0] = (x ** 2).sum() / n
    for k in range(1, order+1):
        r[k] = (x[0:-k] * x[k:]).sum() / (n - k * adj_needed)
    R = linalg.toeplitz(r[:-1])

    rho = np.linalg.solve(R, r[1:])
    return rho


def plot_autocorr(
        x,
        axis=None,
        plot_marker=True,
        show=True,
        legend=None,
        title=None,
        xlabel=None,
        vlines_colors=None,
        hline_color=None,
        marker_color=None,
        legend_fs=None
):

    if not axis:
        _, axis = plt.subplots()

    if plot_marker:
        axis.plot(x, 'o', color=marker_color, label=legend)
        if legend:
            axis.legend(fontsize=legend_fs)
    axis.vlines(range(len(x)), [0], x, colors=vlines_colors)
    axis.axhline(color=hline_color)

    if title:
        axis.set_title(title)
    if xlabel:
        axis.set_xlabel("Lags")

    if show:
        plt.show()

    return axis


def ccovf_np(x, y, unbiased=True, demean=True):
    n = len(x)
    if demean:
        xo = x - x.mean()
        yo = y - y.mean()
    else:
        xo = x
        yo = y
    if unbiased:
        xi = np.ones(n)
        d = np.correlate(xi, xi, 'full')
    else:
        d = n
    return (np.correlate(xo, yo, 'full') / d)[n - 1:]


def ccf_np(x, y, unbiased=True):
    """cross correlation between two time series
    # https://stackoverflow.com/a/24617594
    """
    cvf = ccovf_np(x, y, unbiased=unbiased, demean=True)
    return cvf / (np.std(x) * np.std(y))


def _missing_vals(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Modified after https://github.com/akanz1/klib/blob/main/klib/utils.py#L197
     Gives metrics of missing values in the dataset.
    Parameters
    ----------
    data : pd.DataFrame
        2D dataset that can be coerced into Pandas DataFrame
    Returns
    -------
    Dict[str, float]
        mv_total: float, number of missing values in the entire dataset
        mv_rows: float, number of missing values in each row
        mv_cols: float, number of missing values in each column
        mv_rows_ratio: float, ratio of missing values for each row
        mv_cols_ratio: float, ratio of missing values for each column
    """

    data = pd.DataFrame(data).copy()
    mv_rows = data.isna().sum(axis=1)
    mv_cols = data.isna().sum(axis=0)
    mv_total = data.isna().sum().sum()
    mv_rows_ratio = mv_rows / data.shape[1]
    mv_cols_ratio = mv_cols / data.shape[0]

    return {
        "mv_total": mv_total,
        "mv_rows": mv_rows,
        "mv_cols": mv_cols,
        "mv_rows_ratio": mv_rows_ratio,
        "mv_cols_ratio": mv_cols_ratio,
    }


def find_tot_plots(features, max_subplots):

    tot_plots = np.linspace(0, features, int(features / max_subplots) + 1 if features % max_subplots == 0 else int(
        features / max_subplots) + 2)
    # converting each value to int because linspace can return array containing floats if features is odd
    tot_plots = [int(i) for i in tot_plots]
    return tot_plots



def get_nrows_ncols(n_rows, n_subplots)->"tuple[int, int]":

    if n_rows is None:
        n_rows = int(np.sqrt(n_subplots))
    n_cols = max(int(n_subplots / n_rows), 1)  # ensure n_cols != 0
    n_rows = int(n_subplots / n_cols)

    while not ((n_subplots / n_cols).is_integer() and
               (n_subplots / n_rows).is_integer()):
        n_cols -= 1
        n_rows = int(n_subplots / n_cols)
    return n_rows, n_cols



def dateandtime_now() -> str:
    """
    Returns the datetime in following format as string
    YYYYMMDD_HHMMSS
    """
    jetzt = datetime.datetime.now()
    dt = ''
    for time in ['year', 'month', 'day', 'hour', 'minute', 'second']:
        _time = str(getattr(jetzt, time))
        if len(_time) < 2:
            _time = '0' + _time
        if time == 'hour':
            _time = '_' + _time
        dt += _time
    return dt


def dict_to_file(
        path,
        config=None, errors=None,
        indices=None, others=None, name=''):

    sort_keys = True
    if errors is not None:
        suffix = dateandtime_now()
        fpath = path + "/errors_" + name + suffix + ".json"
        # maybe some errors are not json serializable.
        for er_name, er_val in errors.items():
            if "int" in er_val.__class__.__name__:
                errors[er_name] = int(er_val)
            elif "float" in er_val.__class__.__name__:
                errors[er_name] = float(er_val)

        data = errors
    elif config is not None:
        fpath = path + "/config.json"
        data = config
        sort_keys = False
    elif indices is not None:
        fpath = path + "/indices.json"
        data = indices
    else:
        assert others is not None
        data = others
        fpath = path

    if 'config' in data:
        if data['config'].get('model', None) is not None:
            model = data['config']['model']
            # because ML args which come algorithms may not be of json serializable.
            if 'layers' not in model:

                model = jsonize(model)
                data['config']['model'] = model

    with open(fpath, 'w') as fp:
        json.dump(data, fp, sort_keys=sort_keys, indent=4, cls=JsonEncoder)

    return


def ts_features(
        data: Union[np.ndarray, pd.DataFrame, pd.Series],
        precision: int = 3,
        name: str = '',
        st: int = 0,
        en: int = None,
        features: Union[list, str] = None
        ) -> dict:
    """
    Extracts features from 1d time series data. Features can be
        * point, one integer or float point value for example mean
        * 1D, 1D array for example sin(data)
        * 2D, 2D array for example wavelent transform
    Arguments:
        Gets all the possible stats about an array like object `data`.
        data: array like
        precision: number of significant figures
        name: str, only for erro or warning messages
        st: str/int, starting index of data to be considered.
        en: str/int, end index of data to be considered.
        features: name/names of features to extract from data.
    # information holding degree
    """

    stats = dict()

    point_features = {
        'Mean': np.nanmean,
        'Median': np.nanmedian,
        'Variance': np.nanvar,
        'Std': np.nanstd,
        'Non Zeros': np.count_nonzero,
        'Min': np.nanmin,
        'Max': np.nanmax,
        'Sum': np.nansum,
        'Counts': np.size
    }

    point_features_lambda = {
        'Negative counts': lambda x: int(np.sum(x < 0.0)),
        '90th percentile': lambda x: np.round(np.nanpercentile(x, 90), precision),
        '75th percentile': lambda x: np.round(np.nanpercentile(x, 75), precision),
        '50th percentile': lambda x: np.round(np.nanpercentile(x, 50), precision),
        '25th percentile': lambda x: np.round(np.nanpercentile(x, 25), precision),
        '10th percentile': lambda x: np.round(np.nanpercentile(x, 10), precision),
    }

    if features is None:
        features = list(point_features.keys()) + list(point_features_lambda.keys())
    elif isinstance(features, str):
        features = [features]


    if scipy is not None:
        point_features_lambda.update({
        'Shannon entropy': lambda x: np.round(scipy.stats.entropy(pd.Series(x).value_counts()), precision),            
        })

        point_features.update({
        'Skew': skew,
        'Kurtosis': kurtosis,            
        'Geometric Mean': gmean,
        'Standard error of mean': scipy.stats.sem,
        'Coefficient of Variation': variation,
        })

        if 'Harmonic Mean' in features:
            try:
                stats['Harmonic Mean'] = np.round(hmean(data), precision)
            except ValueError:
                warnings.warn(f"""Unable to calculate Harmonic mean for {name}. Harmonic mean only defined if all
                            elements are greater than or equal to zero""", UserWarning)

    if not isinstance(data, np.ndarray):
        if hasattr(data, '__len__'):
            data = np.array(data)
        else:
            raise TypeError(f"{name} must be array like but it is of type {data.__class__.__name__}")

    if np.array(data).dtype.type is np.str_:
        warnings.warn(f"{name} contains string values")
        return {}

    if 'int' not in data.dtype.name:
        if 'float' not in data.dtype.name:
            warnings.warn(f"changing the dtype of {name} from {data.dtype.name} to float")
            data = data.astype(np.float64)

    assert data.size == len(data), f"""
data must be 1 dimensional array but it has shape {np.shape(data)}
"""
    data = data[st:en]

    for feat in features:
        if feat in point_features:
            stats[feat] = np.round(point_features[feat](data), precision)
        elif feat in point_features_lambda:
            stats[feat] = point_features_lambda[feat](data)

    return jsonize(stats)



class JsonEncoder(json.JSONEncoder):

    def default(self, obj):
        if 'int' in obj.__class__.__name__:
            return int(obj)
        elif 'float' in obj.__class__.__name__:
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif 'bool' in obj.__class__.__name__:
            return bool(obj)
        elif callable(obj) and hasattr(obj, '__module__'):
            if isinstance(obj, FunctionType):
                return obj.__name__
            else:
                return obj.__module__
        else:
            return super(JsonEncoder, self).default(obj)


def jsonize(
        obj,
        type_converters:dict=None
):
    """
    Serializes an object to python's native types so that it can be saved
    in json file format. If the object is a sequence, then each member of th sequence
    is serialized. Same goes for nested sequences like lists of lists
    or list of dictionaries.

    Parameters
    ----------
    obj :
        any python object that needs to be serialized.
    type_converters : dict
        a dictionary definiting how to serialize any particular type
        The keys of the dictionary should be ``type`` the the values
        should be callable to serialize that type.

    Return
    ------
        a serialized python object

    Examples
    --------
    >>> import numpy as np
    >>> from ai4water.utils import jsonize
    >>> a = np.array([2.0])
    >>> b = jsonize(a)
    >>> type(b)  # int
    ... # if a data container consists of mix of native and third party types
    ... # only third party types are converted into native types
    >>> print(jsonize({1: [1, None, True, np.array(3)], 'b': np.array([1, 3])}))
    ... {1: [1, None, True, 3], 'b': [1, 2, 3]}

    The user can define the methods to serialize some types
    e. g., we can serialize tensorflow's tensors using serialize method

    >>> from tensorflow.keras.layers import Lambda, serialize
    >>> tensor = Lambda(lambda _x: _x[Ellipsis, -1, :])
    >>> jsonize({'my_tensor': tensor}, {Lambda: serialize})
    """
    # boolean type
    if isinstance(obj, bool):
        return obj

    if isinstance(obj, np.bool_):
        return bool(obj)

    if 'int' in obj.__class__.__name__:
        return int(obj)

    if 'float' in obj.__class__.__name__:
        return float(obj)

    if isinstance(obj, dict):
        return {jsonize(k, type_converters): jsonize(v, type_converters) for k, v in obj.items()}

    if isinstance(obj, tuple):
        return tuple([jsonize(val, type_converters) for val in obj])

    if obj.__class__.__name__ == 'NoneType':
        return obj

    # if obj is a python 'type' such as jsonize(list)
    if type(obj).__name__ == type.__name__:
        return obj.__name__

    if hasattr(obj, '__len__') and not isinstance(obj, str):

        if hasattr(obj, 'shape') and len(obj.shape) == 0:
            # for cases such as np.array(1)
            return jsonize(obj.item(), type_converters)

        if obj.__class__.__name__ in ['Series', 'DataFrame']:
            # simple list comprehension will iterate over only column names
            # if we simply do jsonize(obj.values()), it will not save column names
            return {jsonize(k, type_converters): jsonize(v, type_converters) for k,v in obj.items()}

        return [jsonize(val, type_converters) for val in obj]

    if callable(obj):
        if isinstance(obj, FunctionType):
            return obj.__name__
        if hasattr(obj, '__package__'):
            return obj.__package__

    if isinstance(obj, collections_abc.Mapping):
        return dict(obj)

    if obj is Ellipsis:
        return {'class_name': '__ellipsis__'}

    if type_converters:
        for _type, converter in type_converters.items():
            if isinstance(obj, _type):
                return converter(obj)

    # last resort, call the __str__ method of object on it
    return str(obj)



class Plot(object):

    def __init__(self, path=None, backend='matplotlib', save=True, dpi=300):
        self.path = path
        self.backend = backend
        self.save = save
        self.dpi = dpi

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, x):

        _backend = x
        assert x in ['plotly', 'matplotlib'], f"unknown backend {x}. Allowed values are `plotly` and `matplotlib`"

        if x == 'plotly':
            raise NotImplementedError("plotly backend is not implemented yet")

        self._backend = _backend

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, x):
        if x is None:
            x = os.getcwd()
        self._path = x

    def save_or_show(
            self,
             save: bool = None,
             fname=None,
             where='', dpi=None,
             bbox_inches='tight',
             close=False,
             show=False
    ):

        if save is None:
            save = self.save

        if dpi is None:
            dpi = self.dpi

        return save_or_show(
            self.path,
            save=save,
            fname=fname,
            where=where,
            dpi=dpi,
            bbox_inches=bbox_inches,
            close=close,
            show=show)


def save_or_show(path, save: bool = True, fname=None,
                 where='',
                 dpi=300, bbox_inches='tight', close=False,
                 show=False):

    if save:
        assert isinstance(fname, str)
        if "/" in fname:
            fname = fname.replace("/", "__")
        if ":" in fname:
            fname = fname.replace(":", "__")

        save_dir = os.path.join(path, where)

        if not os.path.exists(save_dir):
            assert os.path.dirname(where) in ['',
                                              'activations',
                                              'weights',
                                              'plots', 'data', 'results'], f"unknown directory: {where}"
            save_dir = os.path.join(path, where)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

        fname = os.path.join(save_dir, fname + ".png")

        plt.savefig(fname, dpi=dpi, bbox_inches=bbox_inches)
    if show:
        plt.show()

    elif close:
        plt.close('all')
    return

# Function to normalize columns
def min_max_normalize(df:pd.DataFrame)->pd.DataFrame:
    for column in df.columns:
        min_col = df[column].min()
        max_col = df[column].max()
        df[column] = (df[column] - min_col) / (max_col - min_col)
    return df



def get_version_info()->dict:

    from .__init__ import __version__

    versions = {
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'tabulight': __version__,
        'python': sys.version,
        'os': os.name
    }

    versions['matplotlib'] = matplotlib.__version__

    try:
        import scipy
        versions['scipy'] = scipy.__version__
    except (ImportError, ModuleNotFoundError):
        pass

    if seaborn is not None:
        versions['seaborn'] = seaborn.__version__

    if statsmodels is not None:
        versions['statsmodels'] = statsmodels.__version__

    return versions


def hardware_info()->dict:
    mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')  # e.g. 4015976448
    mem_gib = mem_bytes / (1024. ** 3)  # e.g. 3.74
    return dict(
        tot_cpus= os.cpu_count(),
        avail_cpus = os.cpu_count() if os.name=="nt" else len(os.sched_getaffinity(0)),
        mem_gib=mem_gib,
    )


def print_info(
        include_run_time:bool=True,
        include_hardware_info:bool=True
        ):
    info = get_version_info()

    if include_run_time:

        jetzt = datetime.datetime.now()
        zeitpunkt = jetzt.strftime("%d %B %Y %H:%M:%S")
        info['Script Executed on: '] = zeitpunkt

    if include_hardware_info:
        info.update(hardware_info())

    for k, v in info.items():
        print(k, v)
    return


def wq_data(inputs=None, target: Union[list, str] = 'tetx_coppml'):
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

    if inputs is None:
        inputs = default_inputs

    if not isinstance(target, list):
        if isinstance(target, str):
            target = [target]
    elif isinstance(target, list):
        pass
    else:
        target = default_targets

    assert isinstance(target, list)

    return df[inputs + target]
