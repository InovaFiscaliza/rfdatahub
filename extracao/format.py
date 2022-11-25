# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/format.ipynb.

# %% auto 0
__all__ = ['parse_bw', 'optimize_floats', 'optimize_ints', 'optimize_objects', 'df_optimize']

# %% ../nbs/format.ipynb 2
import re
from typing import Iterable, Union, Tuple
from pathlib import Path
from decimal import Decimal

import pandas as pd
from fastcore.utils import listify

from .constants import BW, BW_pattern

# %% ../nbs/format.ipynb 4
def parse_bw(
    bw: str,  # Designação de Emissão (Largura + Classe) codificada como string
) -> Tuple[float, str]:  # Largura e Classe de Emissão
    """Parse the bandwidth string"""
    if match := re.match(BW_pattern, bw):
        multiplier = BW[match.group(2)]
        if mantissa := match.group(3):
            number = float(f"{match.group(1)}.{mantissa}")
        else:
            number = float(match.group(1))
        classe = match.group(4)
        return multiplier * number, classe
    return -1, -1

# %% ../nbs/format.ipynb 7
def optimize_floats(
    df: pd.DataFrame,  # DataFrame a ser otimizado
    exclude: Iterable[str] = None,  # Colunas a serem excluidas da otimização
) -> pd.DataFrame:  # DataFrame com as colunas do tipo `float` otimizadas
    """Otimiza os floats do dataframe para reduzir o uso de memória"""
    floats = df.select_dtypes(include=["float64"]).columns.tolist()
    floats = [c for c in floats if c not in listify(exclude)]
    df[floats] = df[floats].apply(pd.to_numeric, downcast="float")
    return df

# %% ../nbs/format.ipynb 8
def optimize_ints(
    df: pd.DataFrame,  # Dataframe a ser otimizado
    exclude: Iterable[str] = None,  # Colunas a serem excluidas da otimização
) -> pd.DataFrame:  # DataFrame com as colunas do tipo `int` otimizadas
    """Otimiza os ints do dataframe para reduzir o uso de memória"""
    ints = df.select_dtypes(include=["int64"]).columns.tolist()
    ints = [c for c in ints if c not in listify(exclude)]
    df[ints] = df[ints].apply(pd.to_numeric, downcast="integer")
    return df

# %% ../nbs/format.ipynb 9
def optimize_objects(
    df: pd.DataFrame,  # DataFrame a ser otimizado
    datetime_features: Iterable[
        str
    ] = None,  # Colunas que serão convertidas para datetime
    exclude: Iterable[str] = None,  # Colunas que não serão convertidas
) -> pd.DataFrame:  # DataFrame com as colunas do tipo `object` otimizadas
    """Otimiza as colunas do tipo `object` no DataFrame para `category` ou `string` para reduzir a memória e tamanho de arquivo"""
    exclude = listify(exclude)
    datetime_features = listify(datetime_features)
    for col in df.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist():
        if col not in datetime_features:
            if col in exclude:
                continue
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            if float(num_unique_values) / num_total_values < 0.5:
                dtype = "category"
            else:
                dtype = "string"
            df[col] = df[col].astype(dtype)
        else:
            df[col] = pd.to_datetime(df[col]).dt.date
    return df

# %% ../nbs/format.ipynb 10
def df_optimize(
    df: pd.DataFrame,  # DataFrame a ser otimizado
    datetime_features: Iterable[
        str
    ] = None,  # Colunas que serão convertidas para datetime
    exclude: Iterable[str] = None,  # Colunas que não serão convertidas
) -> pd.DataFrame:  # DataFrame com as colunas com tipos de dados otimizados
    """Função que encapsula as anteriores para otimizar os tipos de dados e reduzir o tamanho do arquivo e uso de memória"""
    if datetime_features is None:
        datetime_features = []
    return optimize_floats(
        optimize_ints(optimize_objects(df, datetime_features, exclude), exclude),
        exclude,
    )
