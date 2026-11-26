__version__ = "1.0.0"
import pandas as pd

if int(pd.__version__.split(".", maxsplit=1)[0]) < 3:
    pd.options.mode.copy_on_write = True
