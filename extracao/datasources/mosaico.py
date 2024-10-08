# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/01d_mosaico.ipynb.

# %% auto 0
__all__ = ['MONGO_URI', 'Mosaico']

# %% ../../nbs/01d_mosaico.ipynb 3
import os
import gc

import pandas as pd
from dotenv import find_dotenv, load_dotenv
from fastcore.foundation import GetAttr

from .base import Base
from .connectors import MongoDB

# %% ../../nbs/01d_mosaico.ipynb 4
load_dotenv(find_dotenv(), override=True)

# %% ../../nbs/01d_mosaico.ipynb 6
MONGO_URI = os.environ.get('MONGO_URI', '')
pd.options.mode.copy_on_write = True


# %% ../../nbs/01d_mosaico.ipynb 7
class Mosaico(Base, GetAttr):
	"""Base Class with the common API from the MOSAICO MongoDB Source"""

	def __init__(self, mongo_uri: str = MONGO_URI, read_cache: bool = False):
		self.read_cache = read_cache
		self.database = 'sms'
		self.default = MongoDB(mongo_uri)

	@property
	def collection(self):
		raise NotImplementedError("Subclasses should implement the property 'collection'")

	@property
	def query(self):
		raise NotImplementedError("Subclasses should implement the property 'query'")

	@property
	def projection(self):
		raise NotImplementedError("Subclasses should implement the property 'projection'")

	def _extract(self, collection: str, pipeline: list):
		if self.read_cache:
			return self._read(f'{self.stem}_raw')
		client = self.connect()
		database = client[self.database]
		db_collection = database[collection]
		df = pd.DataFrame(list(db_collection.aggregate(pipeline)), copy=False, dtype='string')
		# Substitui strings vazias, espaços e listas vazias por nulo
		df = df.replace(r'^\s*$|^\[\]$', pd.NA, regex=True)
		# Create the Log Column
		df['Log'] = '[]'
		return df

	@staticmethod
	def split_designacao(
		df: pd.DataFrame,  # DataFrame com coluna original DesignacaoEmissao
	) -> pd.DataFrame:  # DataFrame com novas colunas Largura_Emissão(kHz) e Classe_Emissão
		"""Parse a bandwidth string
		It returns the numerical component and a character class
		"""

		# split then explode
		df['Temp'] = (
			df['Designação_Emissão']
			.str.strip()
			.str.replace(',', ' ')
			.str.strip()
			.str.upper()
			.str.split()
		)
		# Log
		# processing = 'Largura e Classe de Emissão individuais extraídas'
		# Base.register_log(df, processing, 'Designação_Emissão')

		df = df.explode('Temp', ignore_index=True)

		# Removes empty rows
		df = df[df['Temp'] != '/']
		df['Temp'] = df['Temp'].astype('string', copy=False).fillna('')

		# Apply the parse_bw function
		parsed_data = zip(*df['Temp'].apply(Base.parse_bw))

		df['Largura_Emissão(kHz)'], df['Classe_Emissão'] = parsed_data
		df['Largura_Emissão(kHz)'] = df['Largura_Emissão(kHz)'].astype('string', copy=False)
		df['Classe_Emissão'] = df['Classe_Emissão'].astype('string', copy=False)
		return df.drop(['Designação_Emissão', 'Temp'], axis=1)

	def exclude_duplicated(
		self,
		df: pd.DataFrame,  # DataFrame com os dados de Estações
		agg_cols: list,  # Lista de colunas a serem agrupadas
	) -> pd.DataFrame:  # DataFrame com os dados duplicados excluídos
		f"""Exclude and log the duplicated rows
        Columns considered are defined by agg_cols 
        """
		df['Estação'] = (
			df['Estação'].astype('string', copy=False).fillna('-1').astype('int', copy=False)
		)
		df = df.sort_values('Estação', ignore_index=True)
		df['Largura_Emissão(kHz)'] = pd.to_numeric(df['Largura_Emissão(kHz)'], errors='coerce')
		duplicated = df.duplicated(subset=agg_cols, keep='first')

		# Log discarded
		# df_temp = df[duplicated]
		# processing = f'Registro agrupado no arquivo final. Colunas Consideradas: {agg_cols}'
		# Mosaico.register_log(df_temp, processing)
		# self.append2discarded(df_temp)
		# del df_temp
		# gc.collect()

		# I didn't find a better way to do this, the LLMs suggestions were wrong!
		df_temp = df[~duplicated]
		df_sub = df_temp.dropna(subset=agg_cols).reset_index(drop=True)
		# df_temp = df_temp.loc[~df_temp.index.isin(df_sub.index)]

		# # Discard and Log dropped rows
		# processing = f'Valor nulo presente nas colunas utilizadas para agrupamento: {agg_cols}'
		# Mosaico.register_log(df_temp, processing)
		# self.append2discarded(df_temp)
		del df_temp
		gc.collect()

		grouped_stations = df.groupby(agg_cols, dropna=True, sort=False, observed=True)
		del df
		gc.collect()

		df_sub['Multiplicidade'] = grouped_stations.size().values

		df_sub['#Estação'] = grouped_stations['Estação'].apply(lambda x: list(x)).values
		df_sub['#Estação'] = df_sub['#Estação'].astype('string', copy=False)

		row_filter = df_sub['Multiplicidade'] > 1
		processing = f'Registro agrupado.'  # Colunas consideradas: {agg_cols}'
		# TODO: Adicionar nova chave "Colunas Consideradas"
		Mosaico.register_log(df_sub, processing, column='#Estação', row_filter=row_filter)

		return df_sub
