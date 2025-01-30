"""
GEOAISWEB
Este módulo concentra as constantes, funções de carga, processamento, mesclagem e salvamento de dados aeronáuticos provenientes da API do GeoAisWeb
"""
import json
import os
from pathlib import Path
from typing import List
from urllib.request import urlopen

import pandas as pd
from rfdatahub.constants import VOR_ILS_DME

LINK_VOR = "https://geoaisweb.decea.mil.br/geoserver/ICA/ows?service=WFS&version=1.0.0&request=GetFeature&typeName=ICA:vor&outputFormat=application%2Fjson"
LINK_DME = "https://geoaisweb.decea.mil.br/geoserver/ICA/ows?service=WFS&version=1.0.0&request=GetFeature&typeName=ICA:dme&outputFormat=application%2Fjson"
LINK_NDB = "https://geoaisweb.decea.mil.br/geoserver/ICA/ows?service=WFS&version=1.0.0&request=GetFeature&typeName=ICA:ndb&outputFormat=application%2Fjson"
COLS_VOR = (
    "properties.frequency",
    "properties.frequnits",
    "properties.latitude",
    "properties.longitude",
    "properties.tipo",
    "properties.txtname",
    "properties.txtrmk",
)
COLS_NDB = (
    "properties.valfreq",
    "properties.uomfreq",
    "properties.geolat",
    "properties.geolong",
    "properties.tipo",
    "properties.txtname",
    "properties.txtrmk",
)

COLS_DME = (
    "properties.valchannel",
    "properties.codechanne",
    "properties.geolat",
    "properties.geolong",
    "properties.tipo",
    "properties.txtname",
    "Channel",
)

UNIQUE_COLS = ["Frequência", "Latitude", "Longitude"]

def convert_frequency(freq: float, unit: str) -> float:
    """Converts frequency values to MHz based on the input unit.

    Args:
        freq (float): The frequency value to convert
        unit (str): Unit of the frequency value ('Hz', 'kHz', 'MHz', 'GHz')

    Returns:
        float: The frequency value converted to MHz
    """
    conversion = {
        'GHZ': lambda x: x * 1000,
        'MHZ': lambda x: x,
        'KHZ': lambda x: x / 1000,
        'HZ': lambda x: x / 1e6
    }
    unit = unit.upper()
    return conversion.get(unit, lambda _: -1.0)(freq)

def _process_frequency(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Processes frequency data from the input DataFrame.
    
    For DME data, looks up ground frequencies from channel numbers.
    For other data, converts frequencies to MHz.

    Args:
        df (pd.DataFrame): Input DataFrame containing frequency information
        cols (List[str]): List of column names to process

    Returns:
        pd.DataFrame: DataFrame with processed frequency values
    """
    if cols == COLS_DME:
        df_channels = pd.read_csv(VOR_ILS_DME, dtype={'Channel': str, 'DMEground': float})
        df = df.dropna(subset=[cols[0]]).copy()
        
        # Vectorized channel creation
        df['Channel'] = df[cols[0]].astype("int").astype("string") + df[cols[1]].str.upper()
        
        
        # Merge instead of iterative lookup
        df = df.merge(
            df_channels[['Channel', 'DMEground']],
            on='Channel',
            how='left'
        ).rename(columns={'DMEground': 'Frequência'})
        
    else:
        # Vectorized frequency conversion
        df['Frequência'] = (
            df[[cols[0], cols[1]]]
            .apply(lambda x: convert_frequency(x.iloc[0], x.iloc[1]), axis=1)
            .astype("float")
        )
    
    return df

def _filter_df(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Filters and formats DataFrame columns.

    Creates standardized entity descriptions and adds source information.

    Args:
        df (pd.DataFrame): Input DataFrame to filter
        cols (List[str]): List of column names to include

    Returns:
        pd.DataFrame: Filtered DataFrame with standardized columns
    """
    # Use vectorized string operations
    df = df.assign(
        Entidade=(
            df[cols[4]].fillna('') + ' - ' +
            df[cols[5]].fillna('') + ' ' +
            df[cols[6]].fillna('')
        ).astype("string").str.strip(),
        Fonte=pd.Categorical(len(df) * ['AISGEO'], categories=['AISGEO'])
    )
    
    return (
        df[['Frequência', cols[2], cols[3], 'Entidade', 'Fonte']]
        .rename(columns={cols[2]: 'Latitude', cols[3]: 'Longitude'})
        .astype({'Frequência': float, 'Latitude': float, 'Longitude': float})
    )

def get_geodf(link: str, cols: List[str]) -> pd.DataFrame:
    """Retrieves and processes geospatial data from API endpoint.

    Args:
        link (str): API endpoint URL
        cols (List[str]): List of columns to extract from response

    Returns:
        pd.DataFrame: Processed DataFrame containing geospatial data

    Raises:
        ValueError: If API request fails or returns invalid response
    """
    with urlopen(link) as response:
        if response.status != 200 or not response.headers.get('content-type', '').startswith('application/json'):
            raise ValueError(f"Resposta a requisição não foi bem sucedida: {response.status=}")
        
        data = json.load(response)
    
    return (
        pd.json_normalize(data['features'])
        .filter(cols)
        .pipe(_process_frequency, cols=cols)
        .pipe(_filter_df, cols=cols)
    )

def get_aisg() -> pd.DataFrame:
    """Retrieves and combines all GEOAISWEB data sources.

    Fetches NDB, VOR and DME data from GEOAISWEB API endpoints and combines
    them into a single DataFrame.

    Returns:
        pd.DataFrame: Combined DataFrame containing all GEOAISWEB data
    """
    return pd.concat(
        get_geodf(link, cols)
        for link, cols in zip(
            [LINK_NDB, LINK_VOR, LINK_DME], [COLS_NDB, COLS_VOR, COLS_DME]
        )
    )