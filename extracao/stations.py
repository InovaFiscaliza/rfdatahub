# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/04_estacoes.ipynb.

# %% auto 0
__all__ = ['Estacoes']

# %% ../nbs/04_estacoes.ipynb 3
import urllib.request
from typing import List, Union, Tuple
from zipfile import ZipFile

import geopandas as gpd
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from fastcore.foundation import L
from fastcore.parallel import parallel
from fastcore.xtras import Path
from pyarrow import ArrowInvalid, ArrowTypeError


from extracao.constants import (
	COLS_SRD,
	IBGE_MUNICIPIOS,
	IBGE_POLIGONO,
	MALHA_IBGE,
	FLOAT_COLUMNS,
	INT_COLUMNS,
	STR_COLUMNS,
	CAT_COLUMNS,
)
from extracao.location import Geography

from .datasources.aeronautica import Aero
from .datasources.base import Base
from .datasources.mosaico import MONGO_URI
from .datasources.sitarweb import SQLSERVER_PARAMS, Radcom, Stel
from .datasources.smp import Smp
from .datasources.srd import SRD
from .datasources.telecom import Telecom
from .format import merge_on_frequency, LIMIT_FREQ


# %% ../nbs/04_estacoes.ipynb 4
load_dotenv(find_dotenv(), override=True)
pd.options.mode.copy_on_write = True


# %% ../nbs/04_estacoes.ipynb 6
class Estacoes(Base):
	"""Helper Class to aggregate and process the data from different sources"""

	def __init__(
		self,
		sql_params: dict = SQLSERVER_PARAMS,
		mongo_uri: str = MONGO_URI,
		limit: int = 0,
		parallel: bool = True,
		read_cache: bool = False,
	):
		self.sql_params = sql_params
		self.mongo_uri = mongo_uri
		self.limit = limit
		self.parallel = parallel
		self.read_cache = read_cache
		self.init_data_sources()

	@property
	def columns(self):
		return COLS_SRD

	@property
	def stem(self):
		return 'estacoes'

	@staticmethod
	def _update_source(class_instance):
		"""Helper functions to update and save the individual data sources"""
		try:
			class_instance.update()
			class_instance.save()
		except Exception as e:
			print(f'Erro ao atualizar a classe {class_instance.__class__.__name__}: {e}')
		return class_instance

	def init_data_sources(self):
		"""Initializes the individual classes and saves them in a property list"""
		self.sources = L(
			[
				Telecom(self.mongo_uri, self.limit, self.read_cache),
				Smp(self.mongo_uri, self.limit, self.read_cache),
				SRD(self.mongo_uri, self.limit, self.read_cache),
				Stel(self.sql_params, self.read_cache),
				Radcom(self.sql_params, self.read_cache),
				Aero(self.read_cache),
			]
		)

	def extraction(self) -> L:
		if self.parallel:
			self.sources = parallel(
				Estacoes._update_source,
				self.sources,
				n_workers=len(self.sources),
				progress=True,
			)
		else:
			self.sources = self.sources.map(Estacoes._update_source)
		return self.sources.attrgot('df')

	def update(self):
		df = self.extraction()
		self.df = self._format(df)

	@staticmethod
	def _simplify_sources(df):
		df['Fonte'] = df['Fonte'].str.replace(
			'ICAO-CANALIZACAO-VOR/ILS/DME | AISWEB-CANALIZACAO-VOR/ILS/DME',
			'CANALIZACAO-VOR/ILS/DME',
		)
		df['Fonte'] = df['Fonte'].str.replace(
			r'(ICAO-)?(AISWEB-)?CANALIZACAO-VOR/ILS/DME',
			'CANALIZACAO-VOR/ILS/DME',
			regex=True,
		)

		return df

	@staticmethod
	def _remove_invalid_frequencies(df):
		df['Frequência'] = df['Frequência'].astype('float')
		df.sort_values(['Frequência', 'Latitude', 'Longitude'], ignore_index=True, inplace=True)
		return df[df['Frequência'] <= LIMIT_FREQ].reset_index(drop=True)
		# TODO: save to discarded and log
		# log = f"""[("Colunas", "Frequência"),
		# 		   ("Processamento", "Frequência Inválida: Maior que {LIMIT_FREQ}")
		# 		  """
		# self.register_log(df, log, check_coords)

	def _format(
		self,
		dfs: L,  # List with the individual API sources
	) -> pd.DataFrame:  # Processed DataFrame
		aero = dfs.pop()
		anatel = pd.concat(dfs, ignore_index=True, copy=False).astype('string', copy=False)
		df = merge_on_frequency(anatel, aero)
		df = Geography(df).validate()
		df = Estacoes._simplify_sources(df)
		df = Estacoes._remove_invalid_frequencies(df)
		df = df.astype('string', copy=False).replace('-1.0', '-1').astype('category', copy=False)
		return df.loc[:, self.columns]
