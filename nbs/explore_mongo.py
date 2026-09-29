import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    from dotenv import load_dotenv, find_dotenv
    from pymongo import MongoClient
    import pandas as pd

    return MongoClient, find_dotenv, load_dotenv, os, pd


@app.cell
def _(find_dotenv, load_dotenv, os):
    load_dotenv(find_dotenv())
    URI = os.environ.get("MONGO_URI")
    PROJECTION_SRD = {
        "_id": 1.0,
        "frequency": 1.0,
        "licensee": 1.0,
        "NumFistel": 1.0,
        "NumEstacao": "$estacao.NumEstacao",
        "NomeMunicipio": "$srd_planobasico.NomeMunicipio",
        "CodMunicipio": "$srd_planobasico.CodMunicipio",
        "SiglaUF": "$srd_planobasico.SiglaUF",
        "locpb": "$locpb.coordinates",
        "loctx": "$loctx.coordinates",
        "stnClass": 1.0,
        "NumServico": 1.0,
        "DataValFreq": "$habilitacao.DataValFreq",
        "state": "$Status.state",
        "MedCotaBaseTorre": "$estacao.MedCotaBaseTorre",
        "hpat": 1,
        "MedPotenciaOperacao": "$equipamento.transmissor.MedPotenciaOperacao",
        "MedGMaxdBd": "$antena.principal.MedGMaxdBd",
        "MedBeamTilt": "$antena.principal.MedBeamTilt",
        "MedOrientNV": "$antena.principal.MedOrientNV",
        "IndPolariz": "$antena.principal.IndPolariz",
        "MedHCI": "$antena.principal.MedHCI",
        "MedAtenLinhaTransmissaodB100m": "$linhatransmissao.principal.MedAtenLinhaTransmissaodB100m",
        "MedComprimento": "$linhatransmissao.principal.MedComprimento",
        "PerdasAcessorias_db": "$linhatransmissao.principal.PerdasAcessorias_db",
    }
    return (URI,)


@app.cell
def _(MongoClient, URI):
    client = MongoClient(URI)
    list(client.list_databases())
    return (client,)


@app.cell
def _(client):
    sms = client["sms"]
    list(sms.list_collections())
    return (sms,)


@app.cell
def _(sms):
    from pprint import pprint
    list(sms.list_collection_names())
    return


@app.cell
def _(sms):
    sms["licenciamento"].estimated_document_count()
    return


@app.cell
def _(sms):
    sms["licenciamento"].find_one()
    return


@app.cell
def _(sms):
    sms["srd"].find_one()
    return


@app.cell
def _():
    MONGO_SRD = {
        "$and": [
            {"frequency": {"$nin": [None, "", 0], "$type": 1.0}},
            {"srd_planobasico.CodMunicipio": {"$nin": [None, ""]}},
            {"NumFistel": {"$nin": ["", None]}},
        ]
    }
    MONGO_SMP = {
        "$and": [
            {"DataExclusao": {"$in": [None, ""]}},
            {"DataMotivoExclusao": {"$in": [None, ""]}},
            {"DataValidade": {"$nin": ["", None]}},
            {"Status.state": "LIC-LIC-01"},
            {"NumServico": "010"},
            {"FreqTxMHz": {"$nin": [None, "", 0], "$type": 1.0}},
            {"CodMunicipio": {"$nin": [None, ""]}},
            {"NumFistel": {"$nin": [None, ""]}},
            {"CodTipoClasseEstacao": {"$nin": [None, ""]}},
            {"DesignacaoEmissao": {"$nin": [None, ""]}},
            {"Tecnologia": {"$nin": [None, ""]}},
        ]
    }

    MONGO_TELECOM = {
        "$and": [
            {"DataExclusao": {"$in": [None, ""]}},
            {"DataMotivoExclusao": {"$in": [None, ""]}},
            {"DataValidade": {"$nin": ["", None]}},
            {"Status.state": "LIC-LIC-01"},
            {"NumServico": {"$nin": ["010", "045", "171", "450", "750", "", None]}},
            {"FreqTxMHz": {"$nin": [None, "", 0], "$type": 1.0}},
            {"CodMunicipio": {"$nin": [None, ""]}},
            {"NumFistel": {"$nin": [None, ""]}},
            {"CodTipoClasseEstacao": {"$nin": [None, ""]}},
            {"DesignacaoEmissao": {"$nin": [None, ""]}},
            {"Latitude": {"$in": [None, ""]}},
        ]
    }


    DICT_LICENCIAMENTO = {
        "NumAto": "Num_Ato",
        "NumFistel": "Fistel",
        "NumServico": "Serviço",
        "NomeEntidade": "Entidade",
        "SiglaUf": "UF",
        "NumEstacao": "Estação",
        "CodTipoClasseEstacao": "Classe",
        "NomeMunicipio": "Município",
        "CodMunicipio": "Código_Município",
        "DataValidade": "Validade_RF",
        "FreqTxMHz": "Frequência",
        "formId": "Tipo_Estação",
        "Tecnologia": "Tecnologia",
        "Latitude": "Latitude",
        "Longitude": "Longitude",
        "DesignacaoEmissao": "Designação_Emissão",
        # 'PotenciaTransmissorWatts': 'Potência_Transmissor(W)',
        # 'CodTipoAntena': 'Cod_Tipo_Antena',
        # 'Polarizacao': 'Polarização_Antena',
        # 'GanhoAntena': 'Ganho_Antena',
        # 'FrenteCostaAntena': 'FC_Antena',
        # 'AnguloMeiaPotenciaAntena': 'Ang_MP_Antena',
        # 'AnguloElevacao': 'Ângulo_Elevação_Antena',
        # 'Azimute_Antena': 'Azimute_Antena',
        # 'AlturaAntena': 'Altura_Antena',
        # 'PerdasAcessorias': 'Perdas_Acessorias',
    }

    PROJECTION_LICENCIAMENTO = {
      "NumAto": 1.0,
      "NumFistel": 1.0,
      "NumServico": 1.0,
      "NomeEntidade": 1.0,
      "SiglaUf": 1.0,
      "NumEstacao": 1.0,
      "CodTipoClasseEstacao": 1.0,
      "NomeMunicipio": 1.0,
      "CodMunicipio": 1.0,
      "DataValidade": 1.0,
      "FreqTxMHz": 1.0,
      "formId": 1.0,
      "Tecnologia": 1.0,
      "Latitude": 1.0,
      "Longitude": 1.0,
      "DesignacaoEmissao": 1.0,
      "DataExclusao": 1.0,
      "DataMotivoExclusao": 1.0,
      "loctx": 1.0
    }
    return MONGO_TELECOM, PROJECTION_LICENCIAMENTO


@app.cell
def _(MONGO_TELECOM, PROJECTION_LICENCIAMENTO, pd, sms):
    pipeline = [{"$match": MONGO_TELECOM}, {"$project": PROJECTION_LICENCIAMENTO}] #, {"$limit": 100000}]
    df = pd.DataFrame(list(sms["licenciamento"].aggregate(pipeline)))
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
