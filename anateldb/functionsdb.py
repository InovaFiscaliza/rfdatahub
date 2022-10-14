# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/functionsdb.ipynb.

# %% auto 0
__all__ = ['ConsultaSRD']

# %% ../nbs/functionsdb.ipynb 1
from pymongo import MongoClient
import pandas as pd
from decimal import Decimal, getcontext

# %% ../nbs/functionsdb.ipynb 3
def ConsultaSRD(
        mongo_client: MongoClient, # Objeto com o cliente para acesso aos dados do MongoDB                
) -> pd.DataFrame: # DataFrame com os dados atualizados
    """Atualiza a tabela local retornada pela query `RADCOM`"""

    # Colunas retornadas no mosaico
    MOSAICO_COLUMNS = (
        "Num_Serviço",
        "Id",
        "Número_Estação",
        "Latitude",
        "Longitude",
        "Validade_RF",
        "Município",
        "Frequência",
        "Classe",
        "Serviço",
        "Entidade",
        "UF",
        "Status",
        "CNPJ",
        "Fistel"
    )
        
    database = mongo_client["sms"]
    # Database com as informações de Radio e difusão
    collection = database["srd"]

    query = {}
    projection = {}

    projection["SiglaServico"] = 1.0
    projection["_id"] = 1.0
    projection["Status.state"] = 1.0
    projection["licensee"] = 1.0
    projection["NumFistel"] = 1.0
    projection["cnpj"] = 1.0
    projection["frequency"] = 1.0
    projection["stnClass"] = 1.0
    projection["srd_planobasico.NomeMunicipio"] = 1.0
    projection["srd_planobasico.SiglaUF"] = 1.0
    projection["NumServico"] = 1.0
    projection["estacao.NumEstacao"] = 1.0
    projection["estacao.MedLatitudeDecimal"] = 1.0
    projection["estacao.MedLongitudeDecimal"] = 1.0
    projection["habilitacao.DataValFreq"] = 1.0

    list_data = list(collection.find(query, projection = projection))
    mosaico_df = pd.json_normalize(list_data)
    mosaico_df = mosaico_df.drop(columns=['estacao'])
    mosaico_df = mosaico_df[["NumServico"
                            ,"_id"
                            ,"estacao.NumEstacao"
                            ,"estacao.MedLatitudeDecimal"
                            ,"estacao.MedLongitudeDecimal"
                            ,"habilitacao.DataValFreq"
                            ,"srd_planobasico.NomeMunicipio"
                            ,"frequency"
                            ,"stnClass"
                            ,"SiglaServico"
                            ,"licensee"
                            ,"srd_planobasico.SiglaUF"
                            ,"Status.state"
                            ,"cnpj"
                            ,"NumFistel"
    ]]

    mosaico_df.columns = MOSAICO_COLUMNS
    
    mosaico_df = mosaico_df[mosaico_df.Status.str.contains("-C1$|-C2$|-C3$|-C4$|-C7|-C98$", na=False)].reset_index(drop=True)
    
    for c in mosaico_df.columns:
        mosaico_df.loc[mosaico_df[c] == "", c] = pd.NA    
    mosaico_df = mosaico_df.dropna(subset=['UF'])
    mosaico_df = mosaico_df[mosaico_df.Frequência.notna()].reset_index(drop=True)
   
    # mosaico_df = input_coordenates(mosaico_df, "../dados")
    # mosaico_df.loc["Frequência"] = mosaico_df.Frequência.str.replace(",", ".") 
    # mosaico_df.loc[:, "Frequência"] = mosaico_df.Frequência.astype("float")
    # mosaico_df.loc[mosaico_df.Serviço == "OM", "Frequência"] = mosaico_df.loc[
    #     mosaico_df.Serviço == "OM", "Frequência"
    # ].apply(lambda x: Decimal(x) / Decimal(1000))
    # mosaico_df.loc[:, "Validade_RF"] = mosaico_df.Validade_RF.astype("string").str.slice(0, 10)
    return mosaico_df    
