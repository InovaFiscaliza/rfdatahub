# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/02d_redemet.ipynb.

# %% auto 0
__all__ = ['URL', 'UNIQUE_COLS', 'get_redemet']

# %% ../../nbs/02d_redemet.ipynb 2
import json
import os
from datetime import datetime
from urllib.request import urlopen

import pandas as pd
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(), override=True)

# %% ../../nbs/02d_redemet.ipynb 5
URL = 'https://api-redemet.decea.mil.br/produtos/radar/maxcappi?api_key={}&{}'
UNIQUE_COLS = ['Frequência', 'Latitude', 'Longitude']


# %% ../../nbs/02d_redemet.ipynb 6
def get_redemet() -> (
	pd.DataFrame
):  # DataFrame com frequências, coordenadas e descrição das estações VOR
	# sourcery skip: use-fstring-for-concatenation
	"""Makes an API call, process the json
	Returns a DataFrame with the frequencies, coordinates and description of the VOR stations"""
	date = datetime.now().strftime('%Y%m%d')
	link = URL.format(os.environ['RMETKEY'], date)
	response = urlopen(link)
	if response.status != 200 or 'application/json' not in response.headers['content-type']:
		raise ValueError(f'Resposta a requisição não foi bem sucedida: {response.status=}')
	data_json = json.loads(response.read())
	df = pd.json_normalize(
		data_json['data']['radar'][0],
	)
	df['Frequência'] = '2800'
	df['Entidade'] = df.nome.astype('string', copy=False)
	df = df[['Frequência', 'lat_center', 'lon_center', 'Entidade']].astype('string', copy=False)
	df['Fonte'] = 'REDEMET'
	df = df.rename(columns={'lat_center': 'Latitude', 'lon_center': 'Longitude'})
	return df.drop_duplicates(UNIQUE_COLS, ignore_index=True)
