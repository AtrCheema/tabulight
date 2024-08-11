
try:
    import seaborn
except (ModuleNotFoundError, ImportError):
    seaborn = None


try:
    import statsmodels
except (ModuleNotFoundError, ImportError):
    statsmodels = None


try:
    import scipy
except (ModuleNotFoundError, ImportError):
    scipy = None