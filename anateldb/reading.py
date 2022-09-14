# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/reading.ipynb.

# %% auto 0
__all__ = ['read_mosaico', 'read_radcom', 'read_stel', 'read_icao', 'read_aisw', 'read_aisg', 'read_aero', 'read_base']

# %% ../nbs/reading.ipynb 2
from typing import Union, Tuple
from pathlib import Path

import pandas as pd
from pyarrow import ArrowInvalid
import pyodbc

from anateldb.updates import (
update_mosaico, update_stel, update_radcom, update_base
)

# %% ../nbs/reading.ipynb 3
def _read_df(folder: Union[str, Path], stem: str) -> pd.DataFrame:
    """Lê o dataframe formado por folder / stem.[parquet.gzip | fth | xslx]"""
    file = Path(f"{folder}/{stem}.parquet.gzip")
    try:
        df = pd.read_parquet(file)
    except (ArrowInvalid, FileNotFoundError):
        file = Path(f"{folder}/{stem}.fth")
        try:
            df = pd.read_feather(file)
        except (ArrowInvalid, FileNotFoundError):
            file = Path(f"{folder}/{stem}.xlsx")
            try:
                df = pd.read_excel(file, engine="openpyxl", sheet_name="DataBase")
            except Exception as e:
                raise ValueError(f"Error when reading {file}") from e
    return df

# %% ../nbs/reading.ipynb 5
def read_mosaico(folder: Union[str, Path], # Pasta onde ler/salvar os dados
                 update: bool = False, # Atualiza os dados caso `True`
) -> pd.DataFrame: # Dataframe com os dados do mosaico
    """Lê o banco de dados salvo localmente do MOSAICO e opcionalmente o atualiza."""
    return update_mosaico(folder) if update else _read_df(folder, "mosaico")

# %% ../nbs/reading.ipynb 9
def read_radcom(
    conn: pyodbc.Connection, # Objeto de conexão de banco
    folder: Union[str, Path], # Pasta onde ler/salvar os dados
    update: bool = False # Atualiza os dados caso `True`
                
) -> pd.DataFrame: # Dataframe com os dados de RADCOM
    """Lê o banco de dados salvo localmente de RADCOM. Opcionalmente o atualiza pelo Banco de Dados ANATELBDRO05 caso `update = True` ou não exista o arquivo local"""
    return update_radcom(folder) if update else _read_df(folder, "radcom")

# %% ../nbs/reading.ipynb 12
def read_stel(
  conn: pyodbc.Connection, # Objeto de conexão de banco
  folder: Union[str, Path], # Pasta onde ler/salvar os dados
  update: bool = False # Atualiza os dados caso `True`
              
) -> pd.DataFrame: # Dataframe com os dados do STEL
    """Lê o banco de dados salvo localmente do STEL. 
       Opcionalmente o atualiza pelo Banco de Dados ANATELBDRO05
      caso `update = True` ou não exista o arquivo local"""
    return update_stel(folder) if update else _read_df(folder, "stel")

# %% ../nbs/reading.ipynb 15
def read_icao(folder: Union[str, Path], # Pasta onde ler/salvar os dados
              update: bool = False, # Atualiza os dados caso `True`
) -> pd.DataFrame: # Dataframe com os dados do ICAO
    """Lê a base de dados do Frequency Finder e Canalização VOR/ILS/DME"""
    if update:
        # TODO: atualizar a base de dados do Frequency Finder e Canalização VOR/ILS/DME
        # update_icao(folder)
        raise NotImplementedError(
            "Atualizar da base de dados do Frequency Finder e Canalização VOR/ILS/DME não implementado"
        )
    return _read_df(folder, "icao")


# %% ../nbs/reading.ipynb 17
def read_aisw(folder: Union[str, Path], # Pasta onde ler/salvar os dados
              update: bool = False, # Atualiza os dados caso `True`
) -> pd.DataFrame: # Dataframe com os dados do AISWEB
    """Fontes da informação: AISWEB, REDEMET, Ofício nº 2/SSARP/14410 e Canalização VOR/ILS/DME."""
    if update:
        # TODO: Atualizar a base de dados do AISWEB, REDEMET, Ofício nº 2/SSARP/14410 e Canalização VOR/ILS/DME
        # update_pmec(folder)
        raise NotImplementedError(
            "Atualizar da base de dados do Frequency Finder e Canalização VOR/ILS/DME não implementado"
        )
    return _read_df(folder, "aisw")



# %% ../nbs/reading.ipynb 19
def read_aisg(folder: Union[str, Path], # Pasta onde ler/salvar os dados
update: bool = False, # Atualiza os dados caso `True`
) -> pd.DataFrame: # Dataframe com os dados do GEOAISWEB
    """Fontes da informação: GEOAISWEB, REDEMET, Ofício nº 2/SSARP/14410 e Canalização VOR/ILS/DME."""
    if update:
        # TODO: Atualizar a base de dados do GEOAISWEB
        # update_geo(folder)
        raise NotImplementedError(
            "Atualizar da base de dados do Frequency Finder e Canalização VOR/ILS/DME não implementado"
        )
    return _read_df(folder, "aisg")

# %% ../nbs/reading.ipynb 21
def read_aero(
    folder: Union[str, Path], # Pasta onde ler/salvar os dados
update: bool = False, # Atualiza os dados caso `True`
) -> pd.DataFrame: # Dataframe com os dados mesclados das 3 bases da Aeronáutica anteriores
    """Lê os arquivos de dados da aeronáutica e retorna os registros comuns e únicos"""
    if update:
        # TODO: Atualizar a base de dados do GEOAISWEB
        # update_geo(folder)
        raise NotImplementedError(
            "Atualização programática das bases de Dados da Aeronáutica não implementada"
        )
    
    return _read_df(folder, "aero")

# %% ../nbs/reading.ipynb 24
def read_base(
    conn: pyodbc.Connection, # Objeto de conexão de banco
    folder: Union[str, Path], 
    update: bool = False 
) -> pd.DataFrame:
    """Lê a base de dados e opcionalmente a atualiza antes da leitura"""
    return update_base(conn, folder) if update else _read_df(folder, "base")
