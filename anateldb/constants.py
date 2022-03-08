# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/constants.ipynb (unless otherwise specified).

__all__ = ['console', 'ESTADOS', 'SIGLAS', 'REGEX_ESTADOS', 'TIMEOUT', 'RELATORIO', 'ESTACOES', 'ESTACAO',
           'PLANO_BASICO', 'HISTORICO', 'REJECT_ESTACOES', 'COL_ESTACOES', 'NEW_ESTACOES', 'COL_PB', 'NEW_PB',
           'TELECOM', 'RADIODIFUSAO', 'APP_ANALISE', 'ENTIDADES', 'RADCOM', 'STEL']

# Cell
from rich.console import Console

console = Console()

# Cell
ESTADOS = [
    "AC",
    "AL",
    "AP",
    "AM",
    "BA",
    "CE",
    "ES",
    "GO",
    "MA",
    "MT",
    "MS",
    "MG",
    "PA",
    "PB",
    "PR",
    "PE",
    "PI",
    "RJ",
    "RN",
    "RS",
    "RO",
    "RR",
    "SC",
    "SP",
    "SE",
    "TO",
    "DF",
]

SIGLAS = {
    "Acre": "AC",
    "Alagoas": "AL",
    "Amapá": "AP",
    "Amazonas": "AM",
    "Bahia": "BA",
    "Ceará": "CE",
    "Espírito Santo": "ES",
    "Goiás": "GO",
    "Maranhão": "MA",
    "Mato Grosso": "MT",
    "Mato Grosso do Sul": "MS",
    "Minas Gerais": "MG",
    "Pará": "PA",
    "Paraíba": "PB",
    "Paraná": "PR",
    "Pernambuco": "PE",
    "Piauí": "PI",
    "Rio de Janeiro": "RJ",
    "Rio Grande do Norte": "RN",
    "Rio Grande do Sul": "RS",
    "Rondônia": "RO",
    "Roraima": "RR",
    "Santa Catarina": "SC",
    "São Paulo": "SP",
    "Sergipe": "SE",
    "Tocantins": "TO",
    "Distrito Federal": "DF",
}

REGEX_ESTADOS = f'({"|".join(ESTADOS)})'

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
    "Num_Serviço",
    "Classe",
    "Classe_Emissão",
    "Largura_Emissão",
    "Entidade",
    "Fistel",
    "Número_da_Estação",
    "Município",
    "UF",
    "Latitude",
    "Longitude",
    "Validade_RF"
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
    "Class",
    "BW",
    "ActNumber",
    "ActDate",
    "ValRF",
)

ENTIDADES = {}

# Cell
RADCOM = """SELECT F.MedFrequenciaInicial as 'Frequência',
SRD.IndFase as 'Fase',
ID.SiglaSituacao as 'Situação',
Ent.NomeEntidade as 'Entidade',
H.NumFistel as 'Fistel',
E.NumEstacao as 'Número da Estação',
M.NomeMunicipio as 'Município',
PB.SiglaUF as 'UF',
SRD.MedLatitudeDecimal as 'Latitude',
SRD.MedLongitudeDecimal as 'Longitude',
Ent.NumCnpjCpf as 'CNPJ'

  FROM SRD_PEDIDORADCOM SRD

  inner join ESTACAO E on E.IdtHabilitacao =  SRD.IdtHabilitacao
  inner join FREQUENCIA F on F.IdtEstacao = E.IdtEstacao
  inner join HABILITACAO H on H.IdtEntidade = SRD.IdtEntidade
  inner join ENTIDADE Ent on Ent.IdtEntidade = SRD.IdtEntidade
  inner join SRD_PLANOBASICO PB on PB.IdtPlanoBasico = SRD.IdtPlanoBasico
  inner join Municipio M on M.CodMunicipio = PB.CodMunicipio
  left join SRD_INDICESESTACAO ID on ID.IdtHabilitacao = SRD.IdtHabilitacao
  where SRD.IdtPlanoBasico is not Null and SRD.IndFase is not Null

  order by UF, Município, Frequencia"""

# Cell
STEL = """select distinct f.MedTransmissaoInicial as 'Frequência',
uf.SiglaUnidadeFrequencia as 'Unidade',
d.CodClasseEmissao as 'Classe_Emissão',
d.SiglaLarguraEmissao as 'Largura_Emissão',
ce.CodTipoClasseEstacao as 'Classe',
e.NumServico as 'Num_Serviço',
ent.NomeEntidade as 'Entidade',
h.NumFistel as 'Fistel',
e.NumEstacao as 'Número_da_Estação',
mu.NomeMunicipio as 'Município',
e.SiglaUf as 'UF',
e.MedLatitudeDecimal as 'Latitude',
e.MedLongitudeDecimal as 'Longitude',
ent.NumCnpjCpf as 'CNPJ',
c.DataValidadeRadiofrequencia as 'Validade_RF'
from contrato c
inner join estacao e on e.IdtContrato = c.Idtcontrato
inner join frequencia f on f.IdtEstacao = e.IdtEstacao
inner join CLASSEESTACAO ce on ce.IdtFrequencia = f.IdtFrequencia
inner join DESIGNACAOEMISSAO d  on d.IdtClasseEstacao = ce.IdtClasseEstacao
inner join HABILITACAO h on h.IdtHabilitacao = c.IdtHabilitacao
inner join entidade ent on ent.IdtEntidade = h.IdtEntidade
inner join endereco en on en.IdtEstacao = e.IdtEstacao
inner join Municipio mu on mu.CodMunicipio = en.CodMunicipio
inner join Servico s on s.NumServico = h.NumServico and s.IdtServicoAreaAtendimento = 4
left join UnidadeFrequencia uf on uf.IdtUnidadeFrequencia = f.IdtUnidadeTransmissao
where h.NumServico <> '010'
and e.DataExclusao is null
and e.IndStatusEstacao = 'L'
and f.MedTransmissaoInicial is not null
and f.CodStatusRegistro = 'L'
and c.DataValidadeRadiofrequencia is not null
order by 'Frequência'"""