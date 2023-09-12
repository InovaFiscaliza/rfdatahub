# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/04_reading.ipynb.

# %% auto 0
__all__ = ['read_srd', 'read_telecom', 'read_radcom', 'read_stel', 'read_icao', 'read_aisw', 'read_aisg', 'read_redemet',
           'read_aero', 'read_base']

# %% ../nbs/04_reading.ipynb 2
from pathlib import Path
from typing import Union

import pandas as pd
import pyodbc
from dotenv import find_dotenv, load_dotenv
from pymongo import MongoClient

from extracao.updates import (
    update_aero,
    update_base,
    update_radcom,
    update_srd,
    update_stel,
    update_telecom,
)

from .aero.aisgeo import get_aisg
from .aero.aisweb import get_aisw
from .aero.icao import get_icao
from .aero.redemet import get_redemet
from .format import _read_df

load_dotenv(find_dotenv())

# %% ../nbs/04_reading.ipynb 4
def read_srd(
    folder: Union[str, Path],  # Pasta onde ler/salvar os dados
    conn: MongoClient = None,  # Objeto de Conexão com o banco MongoDB, atualiza os dados caso válido
) -> pd.DataFrame:  # Dataframe com os dados do mosaico
    """Lê o banco de dados salvo localmente do MOSAICO e opcionalmente o atualiza."""
    return update_srd(conn, folder) if conn else _read_df(folder, "srd")

# %% ../nbs/04_reading.ipynb 10
def read_telecom(
    folder: Union[str, Path],  # Pasta onde ler/salvar os dados
    conn: MongoClient = None,  # Objeto de Conexão com o banco MongoDB, atualiza os dados caso válido
) -> pd.DataFrame:  # Dataframe com os dados do mosaico
    """Lê o banco de dados salvo localmente do LICENCIAMENTO e opcionalmente o atualiza."""
    return update_telecom(conn, folder) if conn else _read_df(folder, "telecom")

# %% ../nbs/04_reading.ipynb 14
def read_radcom(
    folder: Union[str, Path],  # Pasta onde ler/salvar os dados
    conn: pyodbc.Connection = None,  # Objeto de conexão de banco, atualiza os dados caso válido
) -> pd.DataFrame:  # Dataframe com os dados de RADCOM
    """Lê o banco de dados salvo localmente de RADCOM. Opcionalmente o atualiza pelo Banco de Dados ANATELBDRO05 caso `update = True` ou não exista o arquivo local"""
    return update_radcom(conn, folder) if conn else _read_df(folder, "radcom")

# %% ../nbs/04_reading.ipynb 18
def read_stel(
    folder: Union[str, Path],  # Pasta onde ler/salvar os dados
    conn: pyodbc.Connection = None,  # Objeto de conexão de banco. Atualiza os dados caso válido
) -> pd.DataFrame:  # Dataframe com os dados do STEL
    """Lê o banco de dados salvo localmente do STEL.
     Opcionalmente o atualiza pelo Banco de Dados ANATELBDRO05
    caso `update = True` ou não exista o arquivo local"""
    return update_stel(conn, folder) if conn else _read_df(folder, "stel")

# %% ../nbs/04_reading.ipynb 22
def read_icao(
    folder: Union[str, Path],  # Pasta onde ler/salvar os dados
    update: bool = False,  # Atualiza os dados caso `True`
) -> pd.DataFrame:  # Dataframe com os dados do ICAO
    """Lê a base de dados do Frequency Finder e Canalização VOR/ILS/DME"""
    return get_icao if update else _read_df(folder, "icao")

# %% ../nbs/04_reading.ipynb 23
def read_aisw(
    folder: Union[str, Path],  # Pasta onde ler/salvar os dados
    update: bool = False,  # Atualiza os dados caso `True`
) -> pd.DataFrame:  # Dataframe com os dados do AISWEB
    """Fontes da informação: AISWEB, REDEMET, Ofício nº 2/SSARP/14410 e Canalização VOR/ILS/DME."""
    return get_aisw() if update else _read_df(folder, "aisw")

# %% ../nbs/04_reading.ipynb 24
def read_aisg(
    folder: Union[str, Path],  # Pasta onde ler/salvar os dados
    update: bool = False,  # Atualiza os dados caso `True`
) -> pd.DataFrame:  # Dataframe com os dados do GEOAISWEB
    """Fontes da informação: GEOAISWEB, REDEMET, Ofício nº 2/SSARP/14410 e Canalização VOR/ILS/DME."""
    return get_aisg() if update else _read_df(folder, "aisg")

# %% ../nbs/04_reading.ipynb 25
def read_redemet(
    folder: Union[str, Path],  # Pasta onde ler/salvar os dados
    update: bool = False,  # Atualiza os dados caso `True`
) -> pd.DataFrame:  # Dataframe com os dados do AISWEB
    """Fontes da informação: AISWEB, REDEMET, Ofício nº 2/SSARP/14410 e Canalização VOR/ILS/DME."""
    return get_redemet() if update else _read_df(folder, "redemet")

# %% ../nbs/04_reading.ipynb 26
def read_aero(
    folder: Union[str, Path],  # Pasta onde ler/salvar os dados
    update: bool = False,  # Atualiza os dados caso `True`
) -> (
    pd.DataFrame
):  # Dataframe com os dados mesclados das 3 bases da Aeronáutica anteriores
    """Lê os arquivos de dados da aeronáutica e retorna os registros comuns e únicos"""
    return update_aero(folder) if update else _read_df(folder, "aero")

# %% ../nbs/04_reading.ipynb 30
def read_base(
    folder: Union[str, Path],
    conn: pyodbc.Connection = None,  # Objeto de conexão do banco SQL Server
    clientMongoDB: MongoClient = None,  # Objeto de conexão do banco MongoDB
) -> pd.DataFrame:
    """Lê a base de dados e opcionalmente a atualiza antes da leitura casos as conexões de banco sejam válidas"""
    return (
        update_base(conn, clientMongoDB, folder)
        if all([conn, clientMongoDB])
        else _read_df(folder, "base")
    )
