# AUTOGENERATED! DO NOT EDIT! File to edit: ..\nbs\main.ipynb.

# %% auto 0
__all__ = ['get_db']

# %% ..\nbs\main.ipynb 3
import os
from pathlib import Path
import json
from typing import Union
from datetime import datetime

import pandas as pd
from fastcore.test import *
from rich import print
import pyodbc
from pymongo import MongoClient
from dotenv import load_dotenv, find_dotenv

from .reading import read_base, read_aero
from .format import merge_on_frequency, _filter_matlab

load_dotenv(find_dotenv())

# %% ..\nbs\main.ipynb 4
def get_db(
    path: Union[str, Path],  # Pasta onde salvar os arquivos",
    connSQL: pyodbc.Connection = None,  # Objeto de conexão do banco SQL Server
    clientMongoDB: MongoClient = None,  # Objeto de conexão do banco MongoDB
) -> pd.DataFrame:  # Retorna o DataFrame com as bases da Anatel e da Aeronáutica
    """Lê e opcionalmente atualiza as bases da Anatel, mescla as bases da Aeronáutica, salva e retorna o arquivo
    A atualização junto às bases de dados da Anatel é efetuada caso ambos objetos de banco `connSQL` e `clientMongoDB` forem válidos`
    """
    dest = Path(path)
    dest.mkdir(parents=True, exist_ok=True)
    print(":scroll:[green]Lendo as bases de dados da Anatel...")
    df = read_base(path, connSQL, clientMongoDB)
    df = _filter_matlab(df)
    mod_times = {"ANATEL": datetime.now().strftime("%d/%m/%Y %H:%M:%S")}
    print(":airplane:[blue]Requisitando os dados da Aeronáutica.")
    update = all([connSQL, clientMongoDB])
    aero = read_aero(path, update=update)
    mod_times["AERONAUTICA"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(":spoon:[yellow]Mesclando os dados da Aeronáutica.")
    df = merge_on_frequency(df, aero)
    print(":card_file_box:[green]Salvando os arquivos...")
    df.to_parquet(f"{dest}/AnatelDB.parquet.gzip", compression="gzip", index=False)
    versiondb = json.loads((dest / "VersionFile.json").read_text())
    mod_times["ReleaseDate"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    versiondb["anateldb"].update(mod_times)
    json.dump(versiondb, (dest / "VersionFile.json").open("w"))
    print("Sucesso :zap:")
    return df
