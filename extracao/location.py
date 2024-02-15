from typing import Dict

from functools import cached_property
import urllib.request

from zipfile import ZipFile

from dotenv import load_dotenv, find_dotenv

from fastcore.xtras import Path
from fastcore.foundation import L
import pandas as pd
import geopandas as gpd


from extracao.constants import IBGE_MUNICIPIOS, IBGE_POLIGONO, MALHA_IBGE


# Load environment variables from .env file
load_dotenv(find_dotenv(), override=True)


class Geography:
	def __init__(self, df: pd.DataFrame):
		self.ibge = Path(IBGE_MUNICIPIOS)
		self.shapefile = Path(IBGE_POLIGONO)
		self.check_files()
		self.df = df

	def check_files(self):
		"""Check if the `municipios.csv` file from IBGE exists
		It also calls the `verify_shapefile_folder` method
		"""
		assert self.ibge.is_file(), f'File not found: {IBGE_MUNICIPIOS}'
		self.shapefile.parent.mkdir(exist_ok=True, parents=True)
		self.verify_shapefile_folder()

	def verify_shapefile_folder(self):
		"""It checks the existence and integrity of the all shapefiles from IBGE
		If any of the checks fails, it downloads, extracts and replaces the local files
		"""

		parent_folder = self.shapefile.parent
		zip_file_path = parent_folder.with_suffix('.zip')

		# Check if all required files exist
		required_files = L('.cpg', '.dbf', '.prj', '.shx').map(self.shapefile.with_suffix)
		if not all(required_files.map(Path.is_file)):
			# shutil.rmtree(str(shapefile_path.parent), ignore_errors=True)
			parent_folder.ls().map(Path.unlink)
			# Download and unzip the zipped folder
			urllib.request.urlretrieve(MALHA_IBGE, zip_file_path)
			with ZipFile(zip_file_path, 'r') as zip_ref:
				zip_ref.extractall(parent_folder)
			zip_file_path.unlink()

	def merge_df_with_ibge(
		self,
		df: pd.DataFrame,  # Input dataframe
	):  # DataFrame merged with the IBGE file
		"""It merges the instance df with the IBGE dfs based on `Código_Município`
		The additional columns are: `Latitude_IBGE`, `Longitude_IBGE`, `Município_IBGE`, `UF_IBGE`
		"""

		municipios = pd.read_csv(
			self.ibge,
			usecols=['Código_Município', 'Município', 'UF', 'Latitude', 'Longitude'],
			dtype='string',
			dtype_backend='numpy_nullable',
		)

		df['Código_Município'] = df['Código_Município'].astype('string')

		df = pd.merge(
			df,
			municipios,
			on='Código_Município',
			how='left',
			copy=False,
		)

		df.rename(
			columns={
				'Latitude_x': 'Latitude',
				'Longitude_x': 'Longitude',
				'Município_x': 'Município',
				'UF_x': 'UF',
				'Latitude_y': 'Latitude_IBGE',
				'Longitude_y': 'Longitude_IBGE',
				'Município_y': 'Município_IBGE',
				'UF_y': 'UF_IBGE',
			},
			inplace=True,
		)

		return df

	@cached_property
	def missing(self) -> Dict[str, pd.Series]:
		"""Check the coordinates and city code availability"""
		empty_coords = self.df.Latitude.isna() | self.df.Longitude.isna()
		empty_code = self.df.Código_Município.isna()
		both = empty_coords & empty_code
		left = empty_coords & (~empty_code)
		right = (~empty_coords) & empty_code
		return {'empty_coords': left, 'empty_code': right, 'both': both}

	def fill_missing_coords(self):
		"""Fill the missing coordinates with the central coordinates of the city from IBGE"""
		rows = self.missing['empty_coords']
		self.df.loc[rows, ['Latitude', 'Longitude']] = self.df.loc[
			rows, ['Latitude_IBGE', 'Longitude_IBGE']
		]

	def drop_rows_without_location_info(self):
		rows = self.missing['both']
		self.df = self.df[~rows].reset_index(drop=True)

	def intersect_coordinates_on_poligon(self):
		for column in ['Latitude', 'Longitude']:
			self.df[column] = pd.to_numeric(self.df[column], errors='coerce').astype('float')

		regions = gpd.read_file(self.shapefile)

		# Convert pandas dataframe to geopandas df with geometry point given coordinates
		gdf_points = gpd.GeoDataFrame(
			self.df, geometry=gpd.points_from_xy(self.df.Longitude, self.df.Latitude)
		)

		# Set the same coordinate reference system (CRS) as the regions shapefile
		gdf_points.crs = regions.crs

		# Spatial join points to the regions
		gdf = gpd.sjoin(gdf_points, regions, how='inner', predicate='within')

		gdf['CD_MUN'] = gdf.CD_MUN.astype('string')

		gdf.drop(
			[
				'geometry',
				'AREA_KM2',
				'index_right',
			],
			axis=1,
			inplace=True,
		)

		return gdf

	def fill_missing_code(self):
		rows = self.missing['empty_code']
		self.df.loc[rows, 'Código_Município'] = self.df.loc[rows, 'CD_MUN']

	def validate(self):
		self.df = self.merge_df_with_ibge(self.df)
		self.fill_missing_coords()
		self.drop_rows_without_location_info()
		self.df = self.intersect_coordinates_on_poligon()
		self.fill_missing_code()
		return self.df
