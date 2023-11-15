# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/00b_format.ipynb.

# %% auto 0
__all__ = ['MAX_DIST', 'LIMIT_FREQ', 'parse_bw', 'get_km_distance', 'merge_on_frequency']

# %% ../nbs/00b_format.ipynb 3
import re
from typing import List, Tuple, Union

import pandas as pd
from dotenv import find_dotenv, load_dotenv
from fastcore.utils import listify
from fastcore.xtras import Path
from geopy.distance import geodesic
from pyarrow import ArrowInvalid

from .constants import APP_ANALISE_EN, APP_ANALISE_PT, BW, RE_BW

MAX_DIST = 10  # Km
LIMIT_FREQ = 84812.50
load_dotenv(find_dotenv(), override=True)

# %% ../nbs/00b_format.ipynb 6
def parse_bw(
    bw: str,  # Designação de Emissão (Largura + Classe) codificada como string
) -> Tuple[str, str]:  # Largura e Classe de Emissão
    """Parse the bandwidth string"""
    if match := re.match(RE_BW, bw):
        multiplier = BW[match[2]]
        if mantissa := match[3]:
            number = float(f"{match[1]}.{mantissa}")
        else:
            number = float(match[1])
        classe = match[4]
        return str(multiplier * number), str(classe)
    return pd.NA, pd.NA

# %% ../nbs/00b_format.ipynb 7
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
    df.loc[:, cols_desc] = df.loc[:, cols_desc].astype("string").fillna("NI")

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

# %% ../nbs/00b_format.ipynb 8
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
    for c in df.columns:
        if c not in ["Description", "Class"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["Latitude", "Longitude"]:
        df[c] = df[c].fillna(-1).astype("float32")
    df["Frequency"] = df["Frequency"].astype("float64")
    df["Service"] = df.Service.fillna(-1).astype("int16")
    df["Station"] = df.Station.fillna(-1).astype("int32")
    df["BW"] = df["BW"].fillna(-1).astype("float32")
    df.loc[df["Class"].isin(["", "-1"]), "Class"] = pd.NA
    df["Class"] = df.Class.fillna("NI").astype("category")
    df = df[df.Frequency <= LIMIT_FREQ]
    df.sort_values(
        by=["Frequency", "Latitude", "Longitude", "Description"], inplace=True
    )
    unique_columns = df.columns.tolist()
    unique_columns.remove("Description")
    df = df.drop_duplicates(subset=unique_columns, keep="last").reset_index(drop=True)
    df["Id"] = [f"#{i+1}" for i in df.index]
    df["Id"] = df.Id.astype("string")
    df.loc[df.Description == "", "Description"] = pd.NA
    df["Description"] = df["Description"].astype("string").fillna("NI")
    return df[["Id"] + list(APP_ANALISE_EN)]

# %% ../nbs/00b_format.ipynb 10
def get_km_distance(row):
    return geodesic((row.iloc[0], row.iloc[1]), (row.iloc[2], row.iloc[3])).km


def merge_on_frequency(
    df_left: pd.DataFrame,  # DataFrame da esquerda a ser mesclado
    df_right: pd.DataFrame,  # DataFrame da direira a ser mesclado
    on: str = "Frequência",  # Coluna usada como chave de mesclagem
    cols2merge: str = ("Entidade", "Fonte"),  # Colunas a serem mescladas
) -> pd.DataFrame:  # DataFrame resultante da mesclagem
    """Mescla os dataframes baseados na frequência
    É assumido que as colunas de ambos uma é subconjunto ou idêntica à outra, caso contrário os filtros não irão funcionar como esperado
    """
    df_left = df_left.astype("string[pyarrow]").drop_duplicates(ignore_index=True)
    df_right = df_right.astype("string[pyarrow]").drop_duplicates(ignore_index=True)
    df: pd.DataFrame = pd.merge(
        df_left,
        df_right,
        on=on,
        how="outer",
        indicator=True,
        copy=False,
    )

    x, y = "_x", "_y"
    lat, long = "Latitude", "Longitude"

    left = df._merge == "left_only"
    right = df._merge == "right_only"
    both = df._merge == "both"
    df = df.drop(columns=["_merge"])

    # Disjuntos
    if df[both].empty:
        return pd.concat([df_left, df_right], ignore_index=True)

    left_cols: List[str] = [c for c in df.columns if y not in c]
    right_cols: List[str] = [c for c in df.columns if x not in c]

    only_left = df.loc[left, left_cols].drop_duplicates(
        subset=left_cols, ignore_index=True
    )
    only_left.columns = [c.replace(x, "") for c in left_cols]

    only_right = df.loc[right, right_cols].drop_duplicates(
        subset=right_cols, ignore_index=True
    )
    only_right.columns = [c.replace(y, "") for c in right_cols]

    intersection_left = len(df_left) - len(only_left)
    intersection_right = len(df_right) - len(only_right)

    # Disjuntos
    # if not intersection_left or not intersection_right:
    # 	return pd.concat([df_left, df_right], ignore_index=True)

    both_columns = [f"{lat}{x}", f"{long}{x}", f"{lat}{y}", f"{long}{y}"]
    df.loc[both, "Distance"] = df.loc[both, both_columns].apply(get_km_distance, axis=1)

    df_both = df[both].sort_values("Distance", ignore_index=True)

    filter_left_cols = df_both.columns[: len(df_left.columns)].to_list()
    filter_right_cols = (
        listify(on) + df_both.columns[len(df_left.columns) : -1].to_list()
    )  # the -1 is to eliminate the distance

    # df_both_left = df_both.groupby(filter_left_cols, as_index=False).first()
    # df_both_right = df_both.groupby(filter_right_cols, as_index=False).first()

    df_both_left = df_both.drop_duplicates(
        filter_left_cols, keep="first", ignore_index=True
    )
    df_both_right = df_both.drop_duplicates(
        filter_right_cols, keep="first", ignore_index=True
    )

    assert (
        len(df_both_left) == intersection_left
    ), f"O Agrupamento por colunas únicas não tem o comprimento esperado: {len(df_both_left)}!= {intersection_left}"

    assert (
        len(df_both_right) == intersection_right
    ), f"Error: {len(df_both_right)}!= {intersection_right}"

    df_both_far_left = df_both_left[df_both_left.Distance > MAX_DIST]
    df_both_far_right = df_both_right[df_both_right.Distance > MAX_DIST]

    df_both_left = df_both_left[df_both_left.Distance <= MAX_DIST]
    df_both_right = df_both_right[df_both_right.Distance <= MAX_DIST]

    merge_cols = df_both.columns.to_list()
    merge_cols.remove("Distance")
    df_close_merge = (
        pd.merge(df_both_left, df_both_right, how="inner", on=merge_cols, copy=False)
        .drop("Distance_y", axis=1)
        .rename(columns={"Distance_x": "Distance"})
    )

    df_both_left = _left_filter(df_both_left, df_close_merge, merge_cols)
    df_both_right = _left_filter(df_both_right, df_close_merge, merge_cols)

    assert pd.merge(
        df_both_left, df_both_right, how="inner", on=merge_cols
    ).empty, "Verifique os passos de mesclagem, df_both_left e df_both_right deveria ser disjuntos"

    df_final_merge = pd.concat(
        [df_close_merge, df_both_left, df_both_right], ignore_index=True
    )

    assert len(df_both_far_left) + len(df_close_merge) + len(df_both_left) == (
        len(df_left) - len(only_left)
    ), "Verifique os passos de mesclagem, validação falhou!"

    assert len(df_both_far_right) + len(df_close_merge) + len(df_both_right) == (
        len(df_right) - len(only_right)
    ), "Verifique os passos de mesclagem, validação falhou!"

    original_cols = df_both.columns.to_list()
    df_both = (
        pd.merge(
            df_both, df_final_merge, how="left", on=filter_left_cols, indicator=True
        )
        .loc[lambda x: x["_merge"] == "left_only"]
        .iloc[:, range(len(original_cols))]
    )
    df_both.columns = original_cols

    df_both = (
        pd.merge(
            df_both, df_final_merge, how="left", on=filter_right_cols, indicator=True
        )
        .loc[lambda x: x["_merge"] == "left_only"]
        .iloc[:, range(len(original_cols))]
    )
    df_both.columns = original_cols

    assert (
        df_both.Distance > MAX_DIST
    ).all(), "Verifique os passos de mesclagem, validação falhou!"

    assert (
        pd.merge(
            df_both,
            df_both_far_left,
            how="left",
            on=filter_left_cols,
            indicator=True,
            copy=False,
        )
        .loc[lambda x: x["_merge"] == "left_only"]
        .iloc[:, range(len(original_cols))]
        .empty
    ), "Verifique os passos de mesclagem, validação falhou!"

    assert (
        pd.merge(
            df_both,
            df_both_far_right,
            how="left",
            on=filter_right_cols,
            indicator=True,
            copy=False,
        )
        .loc[lambda x: x["_merge"] == "left_only"]
        .iloc[:, range(len(original_cols))]
        .empty
    ), "Verifique os passos de mesclagem, validação falhou!"

    for col in cols2merge:
        df_final_merge[f"{col}{x}"] = (
            df_final_merge[f"{col}{x}"] + " | " + df_final_merge[f"{col}{y}"]
        )

    df_final_merge = df_final_merge[left_cols]
    df_final_merge.columns = only_left.columns

    df_both_far_left = df_both_far_left[left_cols]
    df_both_far_left.columns = only_left.columns

    df_both_far_right = df_both_far_right[right_cols]
    df_both_far_right.columns = only_right.columns

    merged_df = pd.concat(
        [only_left, df_both_far_left, df_final_merge, only_right, df_both_far_right],
        ignore_index=True,
    )
    return merged_df.astype("string").drop_duplicates(ignore_index=True)


def _left_filter(df, df_close_merge, merge_cols):
    df = pd.merge(
        df, df_close_merge, how="left", on=merge_cols, indicator=True, copy=False
    )
    df = df.loc[df["_merge"] == "left_only"]
    return df.drop(["_merge", "Distance_y"], axis=1).rename(
        columns={"Distance_x": "Distance"}
    )
