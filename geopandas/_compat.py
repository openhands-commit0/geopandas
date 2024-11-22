import importlib
from packaging.version import Version
import pandas as pd
import shapely
import shapely.geos
PANDAS_GE_14 = Version(pd.__version__) >= Version('1.4.0rc0')
PANDAS_GE_15 = Version(pd.__version__) >= Version('1.5.0')
PANDAS_GE_20 = Version(pd.__version__) >= Version('2.0.0')
PANDAS_GE_202 = Version(pd.__version__) >= Version('2.0.2')
PANDAS_GE_21 = Version(pd.__version__) >= Version('2.1.0')
PANDAS_GE_22 = Version(pd.__version__) >= Version('2.2.0')
PANDAS_GE_30 = Version(pd.__version__) >= Version('3.0.0.dev0')
SHAPELY_GE_204 = Version(shapely.__version__) >= Version('2.0.4')
GEOS_GE_390 = shapely.geos.geos_version >= (3, 9, 0)
GEOS_GE_310 = shapely.geos.geos_version >= (3, 10, 0)

def import_optional_dependency(name: str, extra: str=''):
    """
    Import an optional dependency.

    Adapted from pandas.compat._optional::import_optional_dependency

    Raises a formatted ImportError if the module is not present.

    Parameters
    ----------
    name : str
        The module name.
    extra : str
        Additional text to include in the ImportError message.
    Returns
    -------
    module
    """
    try:
        module = importlib.import_module(name)
        return module
    except ImportError as e:
        msg = f"Missing optional dependency '{name}'. {extra} "
        msg += f"Use pip or conda to install {name}."
        raise ImportError(msg) from e
import functools

def requires_pyproj(func):
    """Decorator to check if pyproj is installed."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not HAS_PYPROJ:
            raise ImportError(
                "The pyproj package is required for this functionality. "
                f"Import error: {pyproj_import_error}"
            )
        return func(*args, **kwargs)
    return wrapper

try:
    import pyproj
    HAS_PYPROJ = True
except ImportError as err:
    HAS_PYPROJ = False
    pyproj_import_error = str(err)