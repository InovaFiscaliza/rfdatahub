# AUTOGENERATED! DO NOT EDIT! File to edit: queries.ipynb (unless otherwise specified).

__all__ = ['TIMEOUT', 'RELATORIO', 'ESTACOES', 'ESTACAO', 'PLANO_BASICO', 'HISTORICO', 'REJECT_ESTACOES',
           'COL_ESTACOES', 'NEW_ESTACOES', 'COL_PB', 'NEW_PB', 'TELECOM', 'RADIODIFUSAO', 'APP_ANALISE', 'ENTIDADES',
           'RADCOM', 'STEL', 'optimize_floats', 'optimize_ints', 'optimize_objects', 'df_optimize', 'connect_db',
           'row2dict', 'dict2cols', 'parse_plano_basico', 'scrape_dataframe', 'read_stel', 'read_radcom',
           'read_estações', 'read_plano_basico', 'read_historico', 'read_mosaico', 'clean_merge', 'update_radcom',
           'update_stel', 'update_mosaico']

# Cell
import requests
from decimal import *
from typing import *
from gazpacho import Soup
from rich.progress import track
from pathlib import Path
from unidecode import unidecode
import pandas as pd
import pandas_read_xml as pdx
import pyodbc
import re
import xml.etree.ElementTree as et
from zipfile import ZipFile
import collections
from fastcore.utils import listify
from fastcore.foundation import L
from .constants import console
getcontext().prec = 5

# Cell
TIMEOUT = 5
RELATORIO = "http://sistemas.anatel.gov.br/se/eApp/reports/b/srd/resumo_sistema.php?id={id}&state={state}"
ESTACOES = "http://sistemas.anatel.gov.br/se/public/file/b/srd/estacao_rd.zip"
ESTACAO = "http://sistemas.anatel.gov.br/se/public/view/b/srd.php?wfid=estacoes&id={}"
PLANO_BASICO = "http://sistemas.anatel.gov.br/se/public/file/b/srd/Canais.zip"
HISTORICO = (
    "http://sistemas.anatel.gov.br/se/public/file/b/srd/documento_historicos.zip"
)
REJECT_ESTACOES = [
    "atenuacao",
    "historico_documentos",
    "estacao_auxiliar",
    "rds",
    "aprovacao_locais",
    "item",
]
COL_ESTACOES = (
    "siglaservico",
    "num_servico",
    "state",
    "entidade",
    "fistel",
    "municipio",
    "uf",
    "id",
    "numero_estacao",
    "latitude",
    "longitude",
    "cnpj",
    "habilitacao_datavalfreq",
)
NEW_ESTACOES = [
    "Serviço",
    "Num_Serviço",
    "Status",
    "Entidade",
    "Fistel",
    "Município",
    "UF",
    "Id",
    "Número_da_Estação",
    "Latitude_Transmissor",
    "Longitude_Transmissor",
    "CNPJ",
    "Validade_RF",
    "Num_Ato",
    "Data_Ato",
]
COL_PB = (
    "id",
    "municipio",
    "frequencia",
    "classe",
    "servico",
    "entidade",
    "latitude",
    "longitude",
    "uf",
    "status",
    "cnpj",
    "fistel",
    'canal'
)
NEW_PB = (
    "Id",
    "Município",
    "Frequência",
    "Classe",
    "Serviço",
    "Entidade",
    "Latitude_Estação",
    "Longitude_Estação",
    "UF",
    "Status",
    "CNPJ",
    "Fistel",
    "Canal"
)
TELECOM = (
    "Frequência",
    "Serviço",
    "Entidade",
    "Fistel",
    "Número da Estação",
    "Município",
    "UF",
    "Latitude",
    "Longitude",
)
RADIODIFUSAO = (
    "Frequência",
    "Num_Serviço",
    "Status",
    "Classe",
    "Entidade",
    "Fistel",
    "Número_da_Estação",
    "Município",
    "UF",
    "Latitude",
    "Longitude",
    "Validade_RF",
    "Num_Ato",
    "Data_Ato",
)

APP_ANALISE = (
    "Frequency",
    "Latitude",
    "Longitude",
    "Description",
    "Service",
    "Station",
    "ActNumber",
    "ActDate",
    "ValRF",
)

ENTIDADES = {}

# Cell
RADCOM = """
       select f.MedFrequenciaInicial as 'Frequência',
       Sitarweb.dbo.FN_SRD_RetornaIndFase(PB.NumServico, Pr.idtPedidoRadcom) as 'Fase',
       Sitarweb.dbo.FN_SRD_RetornaSiglaSituacao(h.IdtHabilitacao, Es.IdtEstacao) as 'Situação',
       uf.SiglaUnidadeFrequencia as 'Unidade',
       e.NomeEntidade as 'Entidade',
       h.NumFistel as 'Fistel',
       es.NumEstacao as 'Número da Estação',
       m.NomeMunicipio as 'Município',
       pb.SiglaUF as 'UF',
       es.MedLatitudeDecimal as 'Latitude',
       es.MedLongitudeDecimal as 'Longitude',
       e.NumCnpjCpf as 'CNPJ'
from ENTIDADE e
inner join HABILITACAO h on h.IdtEntidade = e.IdtEntidade
inner join SRD_PEDIDORADCOM pr on pr.IdtHabilitacao = h.IdtHabilitacao
inner join SRD_PLANOBASICO pb on pb.IdtPlanoBasico = pr.IdtPlanoBasico
inner join estacao es on es.IdtHabilitacao = h.IdtHabilitacao
inner join FREQUENCIA f on f.IdtEstacao = es.IdtEstacao
inner join UnidadeFrequencia uf on uf.IdtUnidadeFrequencia = f.IdtUnidadeFrequencia
inner join Municipio m on m.CodMunicipio = pb.CodMunicipio
where h.NumServico = '231'
"""

# Cell
STEL = """IF OBJECT_ID('tempDB..##faixas','U') is not null
drop table ##faixas
create table ##faixas (id int not null, faixa varchar(20), inic float, fim float,);
insert into ##faixas values(0,'De 20 MHz - 6 GHz','20000', '6000000');

select distinct f.MedTransmissaoInicial as 'Frequência',
uf.SiglaUnidadeFrequencia as 'Unidade',
e.NumServico as 'Serviço',
ent.NomeEntidade as 'Entidade',
h.NumFistel as 'Fistel',
e.NumEstacao as 'Número da Estação',
mu.NomeMunicipio as 'Município',
e.SiglaUf as 'UF',
e.MedLatitudeDecimal as 'Latitude',
e.MedLongitudeDecimal as 'Longitude',
ent.NumCnpjCpf as 'CNPJ',
c.DataValidadeRadiofrequencia as 'Validade_RF'
from contrato c
inner join estacao e on e.IdtContrato = c.Idtcontrato
inner join frequencia f on f.IdtEstacao = e.IdtEstacao
inner join HABILITACAO h on h.IdtHabilitacao = c.IdtHabilitacao
inner join entidade ent on ent.IdtEntidade = h.IdtEntidade
inner join endereco en on en.IdtEstacao = e.IdtEstacao
inner join Municipio mu on mu.CodMunicipio = en.CodMunicipio
inner join Servico s on s.NumServico = h.NumServico and s.IdtServicoAreaAtendimento = 4
left join UnidadeFrequencia uf on uf.IdtUnidadeFrequencia = f.IdtUnidadeTransmissao
left outer join ##faixas fx on (
(fx.inic <= f.MedRecepcaoInicialKHz and fx.fim >= f.MedRecepcaoInicialKHz)
or (fx.inic <= f.MedTransmissaoInicialKHz and fx.fim >= f.medtransmissaoinicialkhz)
or (fx.inic <= f.MedFrequenciaInicialKHz and fx.fim >= f.MedFrequenciaInicialKHz)
or (fx.inic <= f.MedFrequenciaFinalKHz and fx.fim >= f.MedFrequenciaFinalKHz)
)
where e.DataExclusao is null and
fx.faixa is not null and
f.MedTransmissaoInicial is not null
and h.NumServico <> '010'
"""

# Cell
def optimize_floats(df: pd.DataFrame, exclude = None) -> pd.DataFrame:
    floats = df.select_dtypes(include=["float64"]).columns.tolist()
    floats = [c for c in floats if c not in listify(exclude)]
    df[floats] = df[floats].apply(pd.to_numeric, downcast="float")
    return df


def optimize_ints(df: pd.DataFrame) -> pd.DataFrame:
    ints = df.select_dtypes(include=["int64"]).columns.tolist()
    df[ints] = df[ints].apply(pd.to_numeric, downcast="integer")
    return df


def optimize_objects(df: pd.DataFrame, datetime_features: List[str]) -> pd.DataFrame:
    for col in df.select_dtypes(include=["object"]):
        if col not in datetime_features:
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            if float(num_unique_values) / num_total_values < 0.5:
                df[col] = df[col].astype("category")
        else:
            df[col] = pd.to_datetime(df[col]).dt.date
    return df


def df_optimize(df: pd.DataFrame, datetime_features: List[str] = [], exclude = None):
    return optimize_floats(optimize_ints(optimize_objects(df, datetime_features)), exclude)

# Cell
def connect_db():
    """Conecta ao Banco ANATELBDRO01 e retorna o 'cursor' (iterador) do Banco pronto para fazer iterações"""
    conn = pyodbc.connect(
        "Driver={ODBC Driver 17 for SQL Server};"
        "Server=ANATELBDRO01;"
        "Database=SITARWEB;"
        "Trusted_Connection=yes;"
        "MultipleActiveResultSets=True;",
        timeout=TIMEOUT,
    )
    return conn

# Cell
def row2dict(row):
    """Receives a json row and return the dictionary from it"""
    return {k: v for k, v in row.items()}


def dict2cols(df, reject=()):
    """Recebe um dataframe com dicionários nas células e extrai os dicionários como colunas
    Opcionalmente ignora e exclue as colunas em reject
    """
    for column in df.columns:
        if column in reject:
            df.drop(column, axis=1, inplace=True)
            continue
        if type(df[column].iloc[0]) == collections.OrderedDict:
            try:
                new_df = pd.DataFrame(df[column].apply(row2dict).tolist())
                df = pd.concat([df, new_df], axis=1)
                df.drop(column, axis=1, inplace=True)
            except AttributeError:
                continue
    return df


def parse_plano_basico(row, cols=COL_PB):
    """Receives a json row and filter the column in `cols`"""
    return {k: row[k] for k in cols}


def scrape_dataframe(id_list):
    df = pd.DataFrame()
    for id_ in track(id_list, description="Baixando informações complementares da Web"):
        html = requests.get(ESTACAO.format(id_))
        df = df.append(pd.read_html(Soup(html.text).find("table").html)[0])

    df.rename(columns={'NumFistel': 'Fistel',
                       'Num Serviço': 'Num_Serviço'}, inplace=True)
    return df[["Status", "Entidade", "Fistel", "Frequência", "Classe", 'Num_Serviço', 'Município', 'UF']]

# Cell
def read_stel(pasta, update=False):
    """Lê o banco de dados salvo localmente do STEL. Opcionalmente o atualiza pelo Banco de Dados ANATELBDRO01."""
    if update:
        update_stel(pasta)
    if (file := Path(f"{pasta}/stel.fth")).exists():
        stel = pd.read_feather(file)
    elif (file := Path(f"{pasta}/stel.csv")).exists():
        stel = pd.read_csv(file)
    elif (file := Path(f"{pasta}/Base_de_Dados.xlsx")).exists():
        stel = pd.read_excel(file, sheet_name="Stel", engine="openpyxl")
    else:
        update_stel(pasta)
        try:
            stel = pd.read_feather(Path(f"{pasta}/stel.fth"))
        except FileNotFoundError as e:
            raise ConnectionError(
                "Base de Dados do Stel inexistente e não foi possível atualizá-la"
            ) from e
    return df_optimize(stel)


def read_radcom(pasta, update=False):
    """Lê o banco de dados salvo localmente de RADCOM. Opcionalmente o atualiza pelo Banco de Dados ANATELBDRO01."""
    if update:
        update_radcom(pasta)
    if (file := Path(f"{pasta}/radcom.fth")).exists():
        radcom = pd.read_feather(file)
    elif (file := Path(f"{pasta}/radcom.csv")).exists():
        radcom = pd.read_csv(file)
    elif (file := Path(f"{pasta}/Base_de_Dados.xlsx")).exists():
        radcom = pd.read_excel(file, sheet_name="Radcom", engine="openpyxl")
    else:
        update_radcom(pasta)
        try:
            radcom = pd.read_feather(Path(f"{pasta}/radcom.fth"))
        except FileNotFoundError as e:
            raise ConnectionError(
                "Base de Dados do Stel inexistente e não foi possível atualizá-la"
            ) from e
    return df_optimize(radcom)


def read_estações(path):
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
            else:
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
    df = df[df.state.str.contains("-C1$|-C2$|-C3$|-C4$|-C7|-C98$")].reset_index(drop=True)
    docs = L(df.historico_documentos.apply(extrair_ato).tolist())
    df = df.loc[:, COL_ESTACOES]
    df["Num_Ato"] = docs.itemgot(0).map(str)
    df["Data_Ato"] = docs.itemgot(1).map(str)
    df.columns = NEW_ESTACOES
    df['Entidade'] = df.Entidade.fillna('')
    ENTIDADES.update({r.Fistel : r.Entidade for r in df.itertuples() if r.Entidade != ''})
    return df


def read_plano_basico(path):
    pb = pdx.read_xml(path, ["plano_basico"])
    # df = pd.DataFrame(df["row"].apply(row2dict).tolist()).replace("", pd.NA)
    dfs = []
    for i in range(pb.shape[0]):
        df = pd.DataFrame(pb["row"][i]).replace("", pd.NA)
        df = dict2cols(df)
        df.columns = [unidecode(c).lower().replace("@", "") for c in df.columns]
        dfs.append(df)
    df = pd.concat(dfs)
    df = df.loc[df.pais == "BRA", COL_PB].reset_index(drop=True)
    df.columns = NEW_PB
    df.sort_values(["Id", "Canal"], inplace=True)
    df['Entidade'] = df.Entidade.fillna('')
    ENTIDADES.update({r.Fistel : r.Entidade for r in df.itertuples() if r.Entidade != ''})
    df = df.groupby("Id", as_index=False).first()  # remove duplicated with NaNs
    df.dropna(subset=['Status'], inplace=True)
    df = df[df.Status.str.contains("-C1$|-C2$|-C3$|-C4$|-C7|-C98$")].reset_index(drop=True)
    return df



def read_historico(path):
    regex = r"\s([a-zA-Z]+)=\'{1}([\w\-\ :\.]*)\'{1}"
    with ZipFile(path) as xmlzip:
        with xmlzip.open("documento_historicos.xml", "r") as xml:
            xml_list = xml.read().decode().split("\n")[2:-1]
    dict_list = []
    for item in xml_list:
        matches = re.finditer(regex, item, re.MULTILINE)
        dict_list.append(dict(match.groups() for match in matches))
    df = pd.DataFrame(dict_list)
    df = df[
        (df.tipoDocumento == "Ato") & (df.razao == "Autoriza o Uso de Radiofrequência")
    ].reset_index()
    df = df.loc[:, ["id", "numeroDocumento", "orgao", "dataDocumento"]]
    df.columns = ["Id", "Num_Ato", "Órgao", "Data_Ato"]
    df["Data_Ato"] = pd.to_datetime(df.Data_Ato)
    return df.sort_values("Data_Ato").groupby("Id").last().reset_index()


def read_mosaico(pasta, update=False):
    if update:
        update_mosaico(pasta)
    if not (file := Path(f"{pasta}/mosaico.xlsx")).exists():
        return read_mosaico(pasta, update=True)
    return pd.read_excel(f"{pasta}/mosaico.xlsx")

# Cell
def clean_merge(pasta, df):
    df = df.copy()
    COLS = [c for c in df.columns if "_x" in c]
    for col in COLS:
        col_x = col
        col_y = col.split("_")[0] + "_y"
        if df[col_x].count() > df[col_y].count():
            a, b = col_x, col_y
        else:
            a, b = col_y, col_x

        df.loc[df[a].isna(), a] = df.loc[df[a].isna(), b]
        df.drop(b, axis=1, inplace=True)
        df.rename({a: a[:-2]}, axis=1, inplace=True)

    df.loc[df.Latitude_Transmissor == "", "Latitude_Transmissor"] = df.loc[
        df.Latitude_Transmissor == "", "Latitude_Estação"
    ]
    df.loc[df.Longitude_Transmissor == "", "Longitude_Transmissor"] = df.loc[
        df.Longitude_Transmissor == "", "Longitude_Estação"
    ]
    df.loc[df.Latitude_Transmissor.isna(), "Latitude_Transmissor"] = df.loc[
        df.Latitude_Transmissor.isna(), "Latitude_Estação"
    ]
    df.loc[df.Longitude_Transmissor.isna(), "Longitude_Transmissor"] = df.loc[
        df.Longitude_Transmissor.isna(), "Longitude_Estação"
    ]
    df.drop(["Latitude_Estação", "Longitude_Estação"], axis=1, inplace=True)
    df.rename(
        columns={
            "Latitude_Transmissor": "Latitude",
            "Longitude_Transmissor": "Longitude",
        },
        inplace=True,
    )
    m = pd.read_excel(f"{pasta}/municípios.xlsx", engine='openpyxl')
    m.loc[
        m.Município == "Sant'Ana do Livramento", "Município"
    ] = "Santana do Livramento"
    m["Município"] = (
        m.Município.apply(unidecode).str.lower().str.replace("'", " ")
    )  # (lambda x: "".join(e for e in x if e.isalnum()))
    m["UF"] = m.UF.str.lower()
    df["Coordenadas_do_Município"] = False
    df["Latitude"] = df.Latitude.str.replace(",", ".")
    df["Longitude"] = df.Longitude.str.replace(",", ".")
    df["Frequência"] = df.Frequência.str.replace(",", ".")
    df.loc[df["Município"] == "Poxoréo", "Município"] = "Poxoréu"
    df.loc[df["Município"] == "Couto de Magalhães", "Município"] = "Couto Magalhães"
    for row in df[(df.Latitude == "") | (df.Latitude.isna())].itertuples():
        try:
            left = unidecode(row.Município).lower()
            m_coord = (
                m.loc[
                    (m.Município == left) & (m.UF == row.UF.lower()),
                    ["Latitude", "Longitude"],
                ]
                .values.flatten()
                .tolist()
            )
            df.loc[row.Index, "Latitude"] = m_coord[0]
            df.loc[row.Index, "Longitude"] = m_coord[1]
            df.loc[row.Index, "Coordenadas_do_Município"] = True
        except ValueError:
            print(left, row.UF, m_coord)
            continue

    freq_nans = df[df.Frequência.isna()].Id.tolist()
    complement_df = scrape_dataframe(freq_nans)
    df.loc[
        df.Frequência.isna(), ["Status", "Entidade", "Fistel", "Frequência", "Classe",
                               'Num_Serviço', 'Município', 'UF']
        ] = complement_df.values

    for r in df[(df.Entidade.isna()) | (df.Entidade == '')].itertuples():
        df.loc[r.Index, 'Entidade'] = ENTIDADES.get(r.Fistel, '')



    df.loc[df["Número_da_Estação"] == "", "Número_da_Estação"] = -1
#     df["Número_da_Estação"] = df["Número_da_Estação"].astype("int")
#     df["Canal"] = df["Canal"].astype("str")
#     df.loc[(df.Classe == '') | (df.Classe.isna()), 'Classe'] = 'Z'
#     df = df_optimize(df, ["Validade_RF", "Data_Ato"])
#     df['Num_Serviço'] = df.Num_Serviço.astype('category')
#     df['Fistel'] = df.Fistel.astype('category')
#     df.loc[df['Validade_RF'].notna(), 'Validade_RF'] = df.loc[df['Validade_RF'].notna(), 'Validade_RF'].astype(str).str.slice(0,10)
#     df.loc[df['Data_Ato'].notna(), 'Data_Ato']  = df.loc[df['Data_Ato'].notna(), 'Data_Ato'].astype(str).str.slice(0,10)
    df["Latitude"] = df["Latitude"].astype("float")
    df["Longitude"] = df["Longitude"].astype("float")
    df["Frequência"] = df.Frequência.astype("float")
    df.loc[df.Serviço == 'OM', 'Frequência'] = df.loc[df.Serviço == 'OM', 'Frequência'].apply(lambda x: Decimal(x) / Decimal(1000))
    df["Frequência"] = df.Frequência.astype("float")
    return df_optimize(df, exclude=['Latitude', 'Longitude', 'Frequência'])

# Cell
def update_radcom(folder):
    """Update the Radcom File querying the Database"""
    with console.status(
        "[cyan]Lendo o Banco de Dados de Radcom...", spinner="earth"
    ) as status:
        try:
            conn = connect_db()
            df = pd.read_sql_query(RADCOM, conn)
            df = df_optimize(df)
            df.to_feather(f"{folder}/radcom.fth")
        except pyodbc.OperationalError:
            status.console.log(
                "Não foi possível abrir uma conexão com o SQL Server. Esta conexão somente funciona da rede cabeada!"
            )


def update_stel(folder):
    """Update the Stel File querying the Database"""
    with console.status(
        "[magenta]Lendo o Banco de Dados do STEL. Processo Lento, aguarde...",
        spinner="moon",
    ) as status:
        try:
            conn = connect_db()
            df = pd.read_sql_query(STEL, conn)
            df = df_optimize(df)
            df.to_feather(f"{folder}/stel.fth")
        except pyodbc.OperationalError:
            status.console.log(
                "Não foi possível abrir uma conexão com o SQL Server. Esta conexão somente funciona da rede cabeada!"
            )


def update_mosaico(pasta):
    """Update the Mosaico File by downloading the zipped xml file from the Spectrum E Web page"""
    with console.status(
        "[blue]Baixando as Estações do Mosaico...", spinner="shark"
    ) as status:
        file = requests.get(ESTACOES)
        with open(f"{pasta}/estações.zip", "wb") as estações:
            estações.write(file.content)
    with console.status(
        "[blue]Baixando o Plano Básico das Estações...", spinner="weather"
    ) as status:
        file = requests.get(PLANO_BASICO)
        with open(f"{pasta}/Canais.zip", "wb") as plano_basico:
            plano_basico.write(file.content)
    console.print("[blue]Consolidando as bases de dados...")
    estações = read_estações(f"{pasta}/estações.zip")
    plano_basico = read_plano_basico(f"{pasta}/Canais.zip")
    df = estações.merge(plano_basico, on="Id", how="left")
    df = clean_merge(pasta, df)
#    df.reset_index(drop=True).to_feather(f"{pasta}/mosaico.fth")
    with pd.ExcelWriter(f"{pasta}/mosaico.xlsx") as workbook:
        df.reset_index(drop=True).to_excel(workbook, sheet_name='Sheet1', engine="openpyxl", index=False)
    console.print("Kbô :vampire:")
    return df