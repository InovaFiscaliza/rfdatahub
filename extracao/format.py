# AUTOGENERATED! DO NOT EDIT! File to edit: ..\nbs\format.ipynb.

# %% auto 0
__all__ = ['RE_BW', 'MAX_DIST', 'LIMIT_FREQ', 'parse_bw', 'get_km_distance', 'merge_on_frequency']

# %% ..\nbs\format.ipynb 2
import re
from typing import Iterable, Tuple, Union, List

import pandas as pd
import numpy as np
from fastcore.utils import listify
from fastcore.xtras import Path
from geopy.distance import geodesic
from rich.progress import Progress
from pyarrow import ArrowInvalid
from dotenv import load_dotenv, find_dotenv


from .constants import BW, BW_pattern, APP_ANALISE_PT, APP_ANALISE_EN

RE_BW = re.compile(BW_pattern)
MAX_DIST = 10  # Km
LIMIT_FREQ = 84812.50
load_dotenv(find_dotenv())

# %% ..\nbs\format.ipynb 3
def _read_df(folder: Union[str, Path], stem: str) -> pd.DataFrame:
    """Lê o dataframe formado por folder / stem.[parquet.gzip | fth | xslx]"""
    file = Path(f"{folder}/{stem}.parquet.gzip")
    try:
        df = pd.read_parquet(file)
    except (ArrowInvalid, FileNotFoundError) as e:
        raise e(f"Error when reading {file}") from e
    return df


# %% ..\nbs\format.ipynb 5
def parse_bw(
    bw: str,  # Designação de Emissão (Largura + Classe) codificada como string
) -> Tuple[str, str]:    # Largura e Classe de Emissão
    """Parse the bandwidth string"""
    if match := re.match(RE_BW, bw):
        multiplier = BW[match[2]]
        if mantissa := match[3]:
            number = float(f"{match[1]}.{mantissa}")
        else:
            number = float(match[1])
        classe = match[4]
        return str(multiplier * number), str(classe)
    return "-1", "-1"


# %% ..\nbs\format.ipynb 6
def _filter_matlab(
    df: pd.DataFrame,  # Arquivo de Dados Base de Entrada
) -> pd.DataFrame:  # Arquivo de Dados formatado para leitura no Matlab
    """Recebe a base de dados da Anatel e formata as colunas para leitura de acordo com os requisitos do Matlab"""
    df["#Estação"] = df["Número_Estação"]
    df.loc[df.Multiplicidade != "1", "#Estação"] = (
        df.loc[df.Multiplicidade != "1", "Número_Estação"]
        + "+"
        + df.loc[df.Multiplicidade != "1", "Multiplicidade"]
    )
    cols_desc = [
        "Fonte",
        "Status",
        "Classe",
        "Entidade",
        "Fistel",
        "#Estação",
        "Município_IBGE",
        "UF",
    ]
    df.loc[:, cols_desc].fillna("NI", inplace=True)

    df["Descrição"] = (
        "["
        + df.Fonte
        + "] "
        + df.Status
        + ", "
        + df.Classe
        + ", "
        + df.Entidade.str.title()
        + " ("
        + df.Fistel
        + ", "
        + df["#Estação"]
        + "), "
        + df.Município_IBGE
        + "/"
        + df.UF
    )

    bad_coords = df.Coords_Valida_IBGE == "0"

    df.loc[bad_coords, "Descrição"] = df.loc[bad_coords, "Descrição"] + "*"

    df.loc[bad_coords, ["Latitude", "Longitude"]] = df.loc[
        bad_coords, ["Latitude_IBGE", "Longitude_IBGE"]
    ].values

    df = df.loc[:, APP_ANALISE_PT]
    df.columns = APP_ANALISE_EN
    return df

# %% ..\nbs\format.ipynb 7
def _format_matlab(
    df: pd.DataFrame,  # Arquivo de Dados Base de Entrada
) -> pd.DataFrame:  # Arquivo de Dados formatado para leitura no Matlab
    """Formata o arquivo final de dados para o formato esperado pela aplicação em Matlab"""
    df = df.astype("string")
    df.loc[len(df), :] = [
        "-1",
        "-15.7801",
        "-47.9292",
        "[TEMP] L, FX, Estação do SMP licenciada (cadastro temporário)",
        "10",
        "999999999",
        "NI",
        "-1",
    ]  # Paliativo...
    for c in ["Latitude", "Longitude"]:
        df[c] = df[c].fillna(-1).astype("float32")
    df["Frequency"] = df["Frequency"].astype("float64")
    df.loc[df.Service.isin(["", "-1"]), "Service"] = pd.NA
    df["Service"] = df.Service.fillna("-1").astype("int16")
    df.loc[df.Station.isin(["", "-1"]), "Station"] = pd.NA
    df["Station"] = df.Station.fillna("-1").astype("int32")
    df.loc[df.BW.isin(["", "-1"]), "BW"] = pd.NA
    df["BW"] = df["BW"].astype("float32").fillna(-1)
    df.loc[df["Class"].isin(["", "-1"]), "Class"] = pd.NA
    df["Class"] = df.Class.fillna("NI").astype("category")
    df = df[df.Frequency <= LIMIT_FREQ]
    df.sort_values(
        by=["Frequency", "Latitude", "Longitude", "Description"], inplace=True
    )
    unique_columns = df.columns.tolist().remove("Description")
    df = df.drop_duplicates(subset=unique_columns, keep="last").reset_index(drop=True)
    df["Id"] = [f"#{i+1}" for i in df.index]
    df["Id"] = df.Id.astype("string")
    df.loc[df.Description == "", "Description"] = pd.NA
    df["Description"] = df["Description"].astype("string").fillna("NI")
    return df[["Id"] + list(APP_ANALISE_EN)]

# %% ..\nbs\format.ipynb 9
def get_km_distance(row):
    return geodesic((row[0], row[1]), (row[2], row[3])).km


def merge_on_frequency(
    df_left: pd.DataFrame,  # DataFrame da esquerda a ser mesclado
    df_right: pd.DataFrame,  # DataFrame da direira a ser mesclado
    on: str = "Frequency",  # Coluna usada como chave de mesclagem
    coords: Tuple[str] = ("Latitude", "Longitude"),
    description: str = "Description",
    suffixes: Tuple[str] = ("_x", "_y"),  # Sufixo para as colunas que foram criadas
) -> pd.DataFrame:  # DataFrame resultante da mesclagem
    df: pd.DataFrame = pd.merge(
        df_left.astype("string"),
        df_right.astype("string"),
        on=on,
        how="outer",
        suffixes=suffixes,
        indicator=True,
        copy=False,
    )

    x, y = suffixes
    lat, long = coords

    left_cols: List[str] = [c for c in df.columns if y not in c]

    right_cols: List[str] = [c for c in df.columns if x not in c]

    left = df._merge == "left_only"
    right = df._merge == "right_only"
    both = df._merge == "both"

    only_left = df.loc[left, left_cols].drop_duplicates().reset_index(drop=True)
    only_left.columns = [c.removesuffix(x) for c in left_cols]

    only_right = df.loc[right, right_cols].drop_duplicates().reset_index(drop=True)
    only_right.columns = [c.removesuffix(y) for c in right_cols]

    both_columns = [f"{lat}{x}", f"{long}{x}", f"{lat}{y}", f"{long}{y}"]

    df.loc[both, "Distance"] = df.loc[both, both_columns].apply(get_km_distance, axis=1)

    close = df.loc[both, "Distance"] <= MAX_DIST
    df_close = df.loc[(both & close)].drop_duplicates().reset_index(drop=True)
    df_close[f"{description}{x}"] = (
        df_close[f"{description}{x}"] + " | " + df_close[f"{description}{y}"]
    )
    df_close = df_close[left_cols]
    df_close.columns = only_left.columns

    df_far_left = (
        df.loc[(both & ~close), left_cols].drop_duplicates().reset_index(drop=True)
    )
    df_far_left.columns = only_left.columns

    df_far_right = (
        df.loc[(both & ~close), right_cols].drop_duplicates().reset_index(drop=True)
    )
    df_far_right.columns = only_right.columns

    merged_df = pd.concat(
        [df_left, df_right, df_close, df_far_right, df_far_left], ignore_index=True
    )
    merged_df.drop(columns=["_merge"], inplace=True)
    
    return merged_df.astype('string')