# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/query.ipynb (unless otherwise specified).

__all__ = ['connect_db', 'update_radcom', 'update_stel', 'update_mosaico', 'update_base']

# Cell
from decimal import Decimal, getcontext
from typing import Union
from pathlib import Path
from urllib.request import urlopen, urlretrieve


import pandas as pd
import pyodbc
from fastcore.test import *
from rich.console import Console
from pyarrow import ArrowInvalid
import pandas_read_xml as pdx
from unidecode import unidecode
from fastcore.foundation import L
from fastcore.utils import listify


from .constants import *
from .format import df_optimize, parse_bw, dict2cols
from .merge import clean_mosaico

getcontext().prec = 5

# Cell
def connect_db():
    """Conecta ao Banco ANATELBDRO01 e retorna o 'cursor' (iterador) do Banco pronto para fazer iterações"""
    return pyodbc.connect(
        "Driver={ODBC Driver 17 for SQL Server};"
        "Server=ANATELBDRO01;"
        "Database=SITARWEB;"
        "Trusted_Connection=yes;"
        "MultipleActiveResultSets=True;",
        timeout=TIMEOUT,
    )

# Internal Cell
def _read_estações(path: Union[str, Path])->pd.DataFrame:
    """Read the zipped xml file `Estações.zip` from MOSAICO and returns a dataframe"""
    def extrair_ato(row):
        if not isinstance(row, str):
            row = listify(row)[::-1]
            for d in row:
                if not isinstance(d, dict):
                    continue
                if (d.get("@TipoDocumento") == "Ato") and (
                    d.get("@Razao") == "Autoriza o Uso de Radiofrequência"
                ):
                    return d["@NumDocumento"], d["@DataDOU"][:10]
            return "", ""
        return "", ""

    es = pdx.read_xml(path, ["estacao_rd"])
    dfs = []
    for i in range(es.shape[0]):
        df = pd.DataFrame(es["row"][i]).replace("", pd.NA)
        df = dict2cols(df)
        df.columns = [unidecode(c).lower().replace("@", "") for c in df.columns]
        dfs.append(df)
    df = pd.concat(dfs)
    df = df[df.state.str.contains("-C1$|-C2$|-C3$|-C4$|-C7|-C98$")].reset_index(
        drop=True
    )
    docs = L(df.historico_documentos.apply(extrair_ato).tolist())
    df = df.loc[:, COL_ESTACOES]
    df["Num_Ato"] = docs.itemgot(0).map(str)
    df["Data_Ato"] = docs.itemgot(1).map(str)
    df.columns = NEW_ESTACOES
    df["Validade_RF"] = df.Validade_RF.astype("string").str.slice(0, 10)
    df["Data_Ato"] = df.Data_Ato.str.slice(0, 10)
    for c in df.columns:
        df.loc[df[c] == "", c] = pd.NA
    return df


def _read_plano_basico(path: Union[str, Path])->pd.DataFrame:
    """Read the zipped xml file `Plano_Básico.zip` from MOSAICO and returns a dataframe"""
    pb = pdx.read_xml(path, ["plano_basico"])
    dfs = []
    for i in range(pb.shape[0]):
        df = pd.DataFrame(pb["row"][i]).replace("", pd.NA)
        df = dict2cols(df)
        df.columns = [unidecode(c).lower().replace("@", "") for c in df.columns]
        dfs.append(df)
    df = pd.concat(dfs)
    df = df.loc[df.pais == "BRA", COL_PB].reset_index(drop=True)
    for c in df.columns:
        df.loc[df[c] == "", c] = pd.NA
    df.columns = NEW_PB
    df.sort_values(["Id", "Canal"], inplace=True)
    ENTIDADES.update(
        {r.Fistel: r.Entidade for r in df.itertuples() if str(r.Entidade) == "<NA>"}
    )
    df = df.groupby("Id", as_index=False).first()  # remove duplicated with NaNs
    df.dropna(subset=["Status"], inplace=True)
    df = df[df.Status.str.contains("-C1$|-C2$|-C3$|-C4$|-C7|-C98$")].reset_index(
        drop=True
    )
    return df

# Cell
def update_radcom(pasta: Union[str, Path])->pd.DataFrame:
    """Atualiza a tabela local retornada pela query `RADCOM`"""
    console = Console()
    with console.status(
        "[cyan]Lendo o Banco de Dados de Radcom...", spinner="earth"
    ) as status:
        try:
            conn = connect_db()
            df = pd.read_sql_query(RADCOM, conn)
            df["Unidade"] = "MHz"
            df = df_optimize(df, exclude=["Frequência"])
            try:
                df.to_feather(f"{pasta}/radcom.fth")
            except ArrowInvalid:
                Path(f"{pasta}/radcom.fth").unlink()
                df.to_excel(f"{pasta}/radcom.xlsx", engine="openpyxl", index=False)
        except pyodbc.OperationalError:
            status.console.log(
                "Não foi possível abrir uma conexão com o SQL Server. Esta conexão somente funciona da rede cabeada!"
            )
    return df


def update_stel(pasta: Union[str, Path])->pd.DataFrame:
    """Atualiza a tabela local retornada pela query `STEL`"""
    console = Console()
    with console.status(
        "[red]Lendo o Banco de Dados do STEL. Processo Lento, aguarde...",
        spinner="moon",
    ) as status:
        try:
            conn = connect_db()
            df = pd.read_sql_query(STEL, conn)
            df["Validade_RF"] = df.Validade_RF.astype("str").str.slice(0, 10)
            df["Num_Serviço"] = df.Num_Serviço.astype("category")
            df.loc[df.Unidade == "kHz", "Frequência"] = df.loc[
                df.Unidade == "kHz", "Frequência"
            ].apply(lambda x: Decimal(x) / Decimal(1000))
            df.loc[df.Unidade == "GHz", "Frequência"] = df.loc[
                df.Unidade == "GHz", "Frequência"
            ].apply(lambda x: Decimal(x) * Decimal(1000))
            df["Frequência"] = df.Frequência.astype("float")
            df.loc[df.Unidade == "kHz", "Unidade"] = "MHz"
            df = df_optimize(df, exclude=["Frequência"])
            try:
                df.to_feather(f"{pasta}/stel.fth")
            except ArrowInvalid:
                Path(f"{pasta}/stel.fth").unlink()
                df.to_excel(f"{pasta}/stel.xlsx", engine="openpyxl", index=False)
        except pyodbc.OperationalError:
            status.console.log(
                "Não foi possível abrir uma conexão com o SQL Server. Esta conexão somente funciona da rede cabeada!"
            )
    return df


def update_mosaico(pasta: Union[str, Path])->pd.DataFrame:
    """Atualiza a tabela local do Mosaico. É baixado e processado arquivos xml zipados da página pública do Spectrum E"""
    console = Console()
    with console.status(
        "[blue]Baixando as Estações do Mosaico...", spinner="shark"
    ) as status:
        stations, _ = urlretrieve(ESTACOES, f"{pasta}/estações.zip")
    with console.status(
        "[blue]Baixando o Plano Básico das Estações...", spinner="weather"
    ) as status:
        pb, _ = urlretrieve(PLANO_BASICO, f"{pasta}/canais.zip")
    console.print(":package: [blue]Consolidando as bases de dados...")
    estações = _read_estações(stations)
    plano_basico = _read_plano_basico(pb)
    df = estações.merge(plano_basico, on="Id", how="left")
    df["Número_da_Estação"] = df["Número_da_Estação"].fillna('-1')
    df["Número_da_Estação"] = df["Número_da_Estação"].astype("string")
    df = clean_mosaico(pasta, df)
    try:
        df.reset_index(drop=True).to_feather(f"{pasta}/mosaico.fth")
    except ArrowInvalid:
        Path(f"{pasta}/mosaico.fth").unlink()
        with pd.ExcelWriter(f"{pasta}/mosaico.xlsx") as workbook:
            df.reset_index(drop=True).to_excel(
                workbook, sheet_name="Sheet1", engine="openpyxl", index=False
            )
    Path(stations).unlink()
    Path(pb).unlink()
    return df

def update_base(pasta: Union[str, Path])->pd.DataFrame:
    """Wrapper que atualiza opcionalmente lê e atualiza as três bases indicadas anteriormente, as combina e salva o arquivo consolidado na pasta `pasta`"""
    console = Console()
    stel = update_stel(pasta).loc[:, TELECOM]
    radcom = update_radcom(pasta)
    mosaico = update_mosaico(pasta)
    radcom["Num_Serviço"] = "231"
    radcom["Status"] = "RADCOM"
    radcom["Classe_Emissão"] = ""
    radcom["Largura_Emissão"] = BW_MAP['231']
    filtro = radcom.Fase.notna() & radcom.Situação.notna()
    radcom.loc[filtro, "Classe"] = radcom.loc[filtro, "Fase"].astype("string") + '-' + radcom.loc[filtro, "Situação"].astype("string")
    radcom["Entidade"] = radcom.Entidade.str.rstrip().str.lstrip()
    radcom["Num_Ato"] = "-1"
    radcom["Data_Ato"] = ""
    radcom["Validade_RF"] = ""
    radcom["Fonte"] = "SRD"
    radcom = df_optimize(radcom, exclude=["Frequência"])
    stel["Status"] = "L"
    stel["Num_Ato"] = "-1"
    stel["Data_Ato"] = ""
    stel["Entidade"] = stel.Entidade.str.rstrip().str.lstrip()
    stel["Fonte"] = "STEL"
    stel = df_optimize(stel, exclude=["Frequência"])
    mosaico["Fonte"] = "MOS"
    mosaico["Classe_Emissão"] = ""
    mosaico["Largura_Emissão"] = mosaico.Num_Serviço.map(BW_MAP)
    mosaico = mosaico.loc[:, RADIODIFUSAO]
    mosaico = df_optimize(mosaico, exclude=["Frequência"])
    rd = (
        pd.concat([mosaico, radcom, stel])
        .sort_values("Frequência")
        .reset_index(drop=True)
    )
    rd["Num_Serviço"] = rd.Num_Serviço.astype("int")
    rd = df_optimize(rd, exclude=["Frequência"])
    rd = rd.drop_duplicates(keep="first").reset_index(drop=True)
    rd['BW(kHz)'] = rd.Largura_Emissão.apply(parse_bw)
    console.print(":trophy: [green]Base Consolidada. Salvando os arquivos...")
    try:
        rd.to_feather(f"{pasta}/base.fth")
    except ArrowInvalid:
        Path(f"{pasta}/base.fth").unlink()
        with pd.ExcelWriter(f"{pasta}/base.xlsx") as workbook:
            rd.to_excel(workbook, sheet_name="Sheet1", engine="openpyxl", index=False)
    return rd