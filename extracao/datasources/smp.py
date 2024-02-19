# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/01g_smp.ipynb.

# %% auto 0
__all__ = ['MONGO_URI', 'SMP']

# %% ../../nbs/01g_smp.ipynb 4
import os

import pandas as pd
import numpy as np
from dotenv import find_dotenv, load_dotenv

from extracao.constants import (
	AGG_SMP,
	CHANNELS,
	COLUNAS,
	DICT_LICENCIAMENTO,
	IBGE_MUNICIPIOS,
	MONGO_SMP,
	PROJECTION_LICENCIAMENTO,
)
from .mosaico import Mosaico
from extracao.location import Geography

# %% ../../nbs/01g_smp.ipynb 5
load_dotenv(find_dotenv())

# %% ../../nbs/01g_smp.ipynb 7
MONGO_URI = os.environ.get('MONGO_URI')


# %% ../../nbs/01g_smp.ipynb 8
class SMP(Mosaico):
	"""Classe para encapsular a lógica de extração do SMP"""

	def __init__(self, mongo_uri: str = MONGO_URI, limit: int = 0) -> None:
		super().__init__(mongo_uri)
		self.limit = limit

	@property
	def stem(self):
		return 'smp'

	@property
	def collection(self):
		return 'licenciamento'

	@property
	def query(self):
		return MONGO_SMP

	@property
	def projection(self):
		return PROJECTION_LICENCIAMENTO

	@property
	def columns(self):
		return COLUNAS

	@property
	def cols_mapping(self):
		return DICT_LICENCIAMENTO

	def extraction(self) -> pd.DataFrame:
		"""This method returns a DataFrame with the results of the mongo query"""
		pipeline = [{'$match': self.query}, {'$project': self.projection}]
		if self.limit > 0:
			pipeline.append({'$limit': self.limit})
		df = self._extract(self.collection, pipeline)
		df['Log'] = ''
		return df

	def exclude_duplicated(
		self,
		df: pd.DataFrame,  # DataFrame com os dados de Estações
	) -> pd.DataFrame:  # DataFrame com os dados duplicados excluídos
		f"""Exclude the duplicated rows
        Columns considered are defined by the AGG_SMP constant
        """
		df['Estação'] = (
			df['Estação'].astype('string', copy=False).fillna('-1').astype('int', copy=False)
		)
		df = df.sort_values('Estação', ignore_index=True)
		df['Largura_Emissão(kHz)'] = pd.to_numeric(df['Largura_Emissão(kHz)'], errors='coerce')
		# df['Largura_Emissão(kHz)'] = df['Largura_Emissão(kHz)'].fillna(0)
		# df['Classe_Emissão'] = df['Classe_Emissão'].fillna('NI')
		# df['Tecnologia'] = df['Tecnologia'].fillna('NI')
		duplicated = df.duplicated(subset=AGG_SMP, keep='first')
		df_sub = df[~duplicated].copy().reset_index(drop=True)
		# discarded = df[duplicated].copy().reset_index(drop=True)
		# log = f"""[("Colunas", {AGG_SMP}),
		#         ("Processamento", "Registro agrupado e descartado do arquivo final")]"""
		# self.append2discarded(self.register_log(discarded, log))
		# for col in AGG_SMP:
		#     discarded_with_na = df_sub[df_sub[col].isna()]
		#     log = f"""[("Colunas", {col}),
		#             ("Processamento", "Registro com valor nulo presente")]"""
		#     self.append2discarded(self.register_log(discarded_with_na, log))
		df_sub.dropna(subset=AGG_SMP, inplace=True)
		df_sub['Multiplicidade'] = (
			df.groupby(AGG_SMP, dropna=True, sort=False, observed=True).size().values
		)
		log = f'[("Colunas", {AGG_SMP}), ("Processamento", "Agrupamento")]'
		return self.register_log(df_sub, log, df_sub['Multiplicidade'] > 1)

	@staticmethod
	def read_channels():
		"""Reads and formats the SMP channels files"""
		channels = pd.read_csv(CHANNELS, dtype='string')
		cols = ['Downlink_Inicial', 'Downlink_Final', 'Uplink_Inicial', 'Uplink_Final']
		channels[cols] = channels[cols].astype('float')
		channels = channels.sort_values(['Downlink_Inicial'], ignore_index=True)
		channels['N_Bloco'] = channels['N_Bloco'].str.strip()
		channels['Faixa'] = channels['Faixa'].str.strip()
		return channels

	def exclude_invalid_channels(
		self,
		df: pd.DataFrame,  # DataFrame de Origem
	) -> pd.DataFrame:  # DataFrame com os canais inválidos excluídos
		"""Helper function to keep only the valid downlink channels"""
		df_sub = df[df.Canalização == 'Downlink'].reset_index(drop=True)
		# for flag in ["Uplink", "Inválida"]:
		#     discarded = df[df.Canalização == flag]
		#     if not discarded.empty:
		#         log = f"""[("Colunas", ("Frequência", "Largura_Emissão(kHz)")),
		#                  ("Processamento", "Canalização {flag}")]"""
		#         self.append2discarded(self.register_log(discarded, log))
		return df_sub

	def validate_channels(
		self,
		df: pd.DataFrame,  # DataFrame with the original channels info
	) -> pd.DataFrame:  # DataFrame with the channels validated and added info
		"""Read the SMP channels file, validate and merge the channels present in df"""
		bw = df['Largura_Emissão(kHz)'].astype('float') / 2000  # Unidade em kHz
		df['Início_Canal_Down'] = df.Frequência.astype(float) - bw
		df['Fim_Canal_Down'] = df.Frequência.astype(float) + bw
		channels = self.read_channels()
		grouped_channels = df.groupby(
			['Início_Canal_Down', 'Fim_Canal_Down'], as_index=False
		).size()
		grouped_channels.sort_values('size', ascending=False, inplace=True, ignore_index=True)
		grouped_channels['Canalização'] = 'Inválida'
		grouped_channels['Offset'] = np.nan
		grouped_channels['Blocos_Downlink'] = pd.NA
		grouped_channels['Faixas'] = pd.NA
		grouped_channels.loc[['Blocos_Downlink', 'Faixas', 'Canalização']] = grouped_channels[
			['Blocos_Downlink', 'Faixas', 'Canalização']
		].astype('string', copy=False)
		grouped_channels['Offset'] = grouped_channels['Offset'].astype('float', copy=False)

		for row in grouped_channels.itertuples():
			interval = channels[
				(row.Início_Canal_Down < channels['Downlink_Final'])
				& (row.Fim_Canal_Down > channels['Downlink_Inicial'])
			]
			faixa = 'Downlink'
			if interval.empty:
				interval = channels[
					(row.Início_Canal_Down < channels['Uplink_Final'])
					& (row.Fim_Canal_Down > channels['Uplink_Inicial'])
				]
				if interval.empty:
					continue
				faixa = 'Uplink'

			down = ' | '.join(
				interval[['Downlink_Inicial', 'Downlink_Final']].apply(
					lambda x: f'{x.iloc[0]}-{x.iloc[1]}', axis=1
				)
			)
			faixas = ' | '.join(interval.Faixa.unique())
			if len(offset := interval.Offset.unique()) != 1:
				continue
			grouped_channels.loc[
				row.Index, ['Blocos_Downlink', 'Faixas', 'Canalização', 'Offset']
			] = (down, faixas, faixa, float(offset[0]))
		grouped_channels = grouped_channels[
			[
				'Início_Canal_Down',
				'Fim_Canal_Down',
				'Blocos_Downlink',
				'Faixas',
				'Canalização',
				'Offset',
			]
		]
		df = pd.merge(df, grouped_channels, how='left', on=['Início_Canal_Down', 'Fim_Canal_Down'])
		return self.exclude_invalid_channels(df)

	def generate_uplink(
		self,
		df: pd.DataFrame,  # Source dataFrame with downlink frequencies and offset
	) -> pd.DataFrame:  # DataFrame with the uplink frequencies added
		"""Generate the respective Uplink channels based on the Downlink frequencies and Offset"""
		df['Offset'] = pd.to_numeric(df['Offset'], errors='coerce').astype('float')
		df['Largura_Emissão(kHz)'] = pd.to_numeric(
			df['Largura_Emissão(kHz)'], errors='coerce'
		).astype('float')
		valid = (
			(df.Offset.notna())
			& (~np.isclose(df.Offset, 0))
			& (df['Largura_Emissão(kHz)'].notna())
			& (~np.isclose(df['Largura_Emissão(kHz)'], 0))
		)
		df[['Frequência', 'Offset']] = df[['Frequência', 'Offset']].astype('float')
		df.loc[valid, 'Frequência_Recepção'] = df.loc[valid, 'Frequência'] - df.loc[valid, 'Offset']
		return df

	def substitute_coordinates(
		self,
		df: pd.DataFrame,  # Source dataframe
	) -> pd.DataFrame:  # Source dataframe with coordinates replace for the city one
		"""Substitute the coordinates for the central coordinates of the municipality
		Only does it for the grouped rows (Multiplicity > 1) since for these rows the
		coordinate values are no longer valid.

		"""
		geo = Geography(df)
		df = geo.merge_df_with_ibge(df)
		rows = df.Multiplicidade > 1
		df.loc[rows, 'Latitude'] = df.loc[rows, 'Latitude_IBGE'].copy()
		df.loc[rows, 'Longitude'] = df.loc[rows, 'Longitude_IBGE'].copy()
		log = """[("Colunas", ("Latitude", "Longitude")), 
        ("Processamento", "Substituição por Coordenadas do Município (Agrupamento)")]"""
		return self.register_log(df, log, df.Multiplicidade > 1)

	def input_fixed_columns(
		self,
		df: pd.DataFrame,  # Source dataframe
	) -> pd.DataFrame:  # Cleaned dataframe with some additional columns added
		"""Formats and adds some helper columns to the dataframe"""
		df['Status'] = 'L'
		df['Serviço'] = '010'
		down = df.drop('Frequência_Recepção', axis=1)
		down['Fonte'] = 'MOSAICO-LICENCIAMENTO'
		down['Classe'] = 'FB'
		up = df.drop('Frequência', axis=1)
		up = up.rename(columns={'Frequência_Recepção': 'Frequência'})
		up.dropna(subset='Frequência', inplace=True)
		up['Fonte'] = 'CANALIZACAO-SMP'
		up['Classe'] = 'ML'
		return pd.concat([down, up], ignore_index=True)

	def _format(
		self,
		df: pd.DataFrame,  # Source dataframe
	) -> pd.DataFrame:  # Final processed dataframe
		"""Formats, cleans, groups, adds and standardizes the queried data from the database"""
		df = df.rename(columns=self.cols_mapping)
		df = self.split_designacao(df)
		df = self.exclude_duplicated(df)
		df = self.validate_channels(df)
		df = self.generate_uplink(df)
		df = self.substitute_coordenates(df)
		df = self.input_fixed_columns(df)
		return df.loc[:, self.columns]
