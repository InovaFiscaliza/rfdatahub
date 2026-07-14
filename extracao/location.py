from typing import Dict

from functools import cached_property
import urllib.request
import ssl

from zipfile import ZipFile

from dotenv import load_dotenv, find_dotenv

from fastcore.xtras import Path
from fastcore.foundation import L
import pandas as pd
import geopandas as gpd
from tqdm.auto import tqdm
from rich import print as pp

from extracao.constants import IBGE_MUNICIPIOS, IBGE_POLIGONO, MALHA_IBGE
from extracao.datasources.base import Base


tqdm.pandas()

ssl._create_default_https_context = ssl._create_unverified_context




# Load environment variables from .env file
load_dotenv(find_dotenv(), override=True)
pd.options.mode.copy_on_write = True


class Geography:
    def __init__(self, df: pd.DataFrame):
        self.ibge: Path = Path(IBGE_MUNICIPIOS)
        self.shapefile: Path = Path(IBGE_POLIGONO)
        self.check_files()
        self.df: pd.DataFrame = df
        self._initialize()

    def _initialize(self):
        self.log_empty_coords()
        self.log_empty_code()
        self.drop_rows_without_location_info()
        self.validate_coordinates_as_number()
        self.validate_codigo_municipio_as_number()

    def check_files(self):
        """Check if the `municipios.csv` file from IBGE exists
        It also calls the `verify_shapefile_folder` method
        """
        assert self.ibge.is_file(), f"File not found: {IBGE_MUNICIPIOS}"
        self.shapefile.parent.mkdir(exist_ok=True, parents=True)
        self.verify_shapefile_folder()

    def verify_shapefile_folder(self):
        """It checks the existence and integrity of the all shapefiles from IBGE
        If any of the checks fails, it downloads, extracts and replaces the local files
        """

        parent_folder = self.shapefile.parent
        zip_file_path = parent_folder.with_suffix(".zip")

        # Check if all required files exist
        required_files = L(".cpg", ".dbf", ".prj", ".shx").map(
            self.shapefile.with_suffix
        )
        if not all(required_files.map(Path.is_file)):
            # shutil.rmtree(str(shapefile_path.parent), ignore_errors=True)
            parent_folder.ls().map(Path.unlink)

            # Download and unzip the zipped folder
            urllib.request.urlretrieve(MALHA_IBGE, zip_file_path)
            with ZipFile(zip_file_path, "r") as zip_ref:
                zip_ref.extractall(parent_folder)
            zip_file_path.unlink()

    @cached_property
    def log(self) -> Dict[str, pd.Series]:
        """Check the coordinates and city code availability"""
        empty_coords = self.df.Latitude.isna() | self.df.Longitude.isna()
        empty_code = self.df.Código_Município.isna()
        both = empty_coords & empty_code
        left = empty_coords & (~empty_code)
        right = (~empty_coords) & empty_code
        return {"empty_coords": left, "empty_code": right, "both": both}

    def log_empty_coords(self):
        """Log the rows with empty coordinates"""
        self.log["empty_coords"]
        # processing = 'Coordenadas nulas.'
        # Base.register_log(self.df, processing, row_filter=rows)

    def log_empty_code(self):
        """Log the rows with empty city code"""
        self.log["empty_code"]
        # processing = 'Código do Município nulo.'
        # Base.register_log(self.df, processing, row_filter=rows)

    def drop_rows_without_location_info(self) -> None:
        rows = self.log["both"]
        # processing = 'Coordenadas e Código do Município nulos. Registro descartado'
        # # TODO: append to discarded
        # Base.register_log(self.df, processing, row_filter=rows)
        self.df = self.df[~rows]

    def validate_coordinates_as_number(self) -> None:
        for column in ["Latitude", "Longitude"]:
            self.df[column] = self.df[column].astype("string", copy=False)
            self.df[column] = self.df[column].str.replace(",", ".")
            self.df[column] = self.df[column].str.strip()
            self.df[f"#{column}"] = self.df[column]
            self.df[column] = pd.to_numeric(self.df[column], errors="coerce").astype(
                "float", copy=False
            )  # type: ignore
            bad_coords = ~self.log["empty_coords"]
            bad_coords &= self.df[column].isna()
            self.log.update({"coords_not_numeric": bad_coords})
            Base.register_log(self.df, "Valor Inválido.", f"#{column}", bad_coords)
            self.df.drop(columns=[f"#{column}"], inplace=True)

    def validate_codigo_municipio_as_number(self) -> None:
        self.df["Código_Município"] = self.df["Código_Município"].astype(
            "string", copy=False
        )
        self.df["Código_Município"] = (
            self.df["Código_Município"].str.strip().str.replace(".", "")
        )
        self.df["#Código_Município"] = self.df["Código_Município"]
        self.df["Código_Município"] = pd.to_numeric(
            self.df["Código_Município"], errors="coerce"
        ).astype("string", copy=False)
        bad_codes = ~self.log["empty_code"]
        bad_codes &= self.df["Código_Município"].isna()
        self.log.update({"code_not_numeric": bad_codes})
        Base.register_log(self.df, "Valor Inválido.", "#Código_Município", bad_codes)
        self.df.drop(columns=["#Código_Município"], inplace=True)

    def _replace_columns(self, columns, originals, log, row_filter, float_cols=False):
        for column, original in zip(columns, originals):
            if not float_cols:
                row_filter = row_filter & (
                    self.df[original].astype("string").fillna("")
                    != self.df[column].astype("string").fillna("")
                )
            if log:
                self.df[f"#{original}"] = self.df[original].fillna("")
                Base.register_log(self.df, log, f"#{original}", row_filter)
                self.df.drop(columns=[f"#{original}"], inplace=True)
            # Use of row_filter instead of rows to avoid float comparison
            self.df.loc[row_filter, original] = self.df.loc[row_filter, column]

    def replace_bad_city_codes(self):
        """Tries to fix the wrong city codes by looking at the normalized city names"""
        import re
        import unicodedata

        def remove_punctuation(word):
            # Unicode normalize transforma um caracter em seu equivalente em latin.
            nfkd = unicodedata.normalize("NFKD", word.lower())
            decoded_word = "".join(c for c in nfkd if not unicodedata.combining(c))

            return re.sub(r"[^a-zA-Z0-9 ]", "", decoded_word)

        municipios = pd.read_csv(
            self.ibge,
            usecols=["Código_Município", "Município", "UF"],
            dtype="string",
            dtype_backend="numpy_nullable",
        )

        municipios["Município_ASCII"] = municipios["Município"].apply(
            remove_punctuation
        )

        self.df["Município"] = self.df["Município"].astype("string", copy=False)
        self.df["Município_ASCII"] = (
            self.df["Município"].fillna("").apply(remove_punctuation)
        )

        self.df = pd.merge(
            self.df,
            municipios,
            on=["Município_ASCII", "UF"],
            how="left",
            copy=False,
        )

        self.df.rename(
            columns={
                "Município_x": "Município",
                "Código_Município_x": "Código_Município",
            },
            inplace=True,
        )

        bad_codes = ~self.log["empty_code"]
        bad_codes &= ~self.log["code_not_numeric"]
        bad_codes &= self.df["Município_IBGE"].isna()
        self.log.update({"invalid_code": bad_codes})

        # processing = 'Código do Município não consta no IBGE.'
        # processing += '\nMunicípio normalizado e UF serão usados como chave para validação.'
        # Base.register_log(self.df, processing, 'Código_Município', bad_codes)
        columns = ["Município_y", "Código_Município_y"]
        originals = ["Município", "Código_Município"]
        log = "Valor Inválido."
        # TODO: Verify if there isn't a wrong Município match, given Município is not unique
        self._replace_columns(columns, originals, log, bad_codes)

        self.df.drop(
            columns=["Município_y", "Município_ASCII", "Código_Município_y"],
            inplace=True,
        )

        self.df[["Município", "Código_Município"]] = self.df[
            ["Município", "Código_Município"]
        ].astype("string", copy=False)

    def merge_df_with_ibge(self):
        """It merges the instance df with the IBGE dfs based on `Código_Município`
        The additional columns are: `Latitude_IBGE`, `Longitude_IBGE`, `Município_IBGE`, `UF_IBGE`
        """

        municipios = pd.read_csv(
            self.ibge,
            usecols=["Código_Município", "Município", "Latitude", "Longitude", "UF"],
            dtype="string",
            dtype_backend="numpy_nullable",
        )

        self.df = pd.merge(
            self.df,
            municipios,
            on="Código_Município",
            how="left",
            copy=False,
        ).astype("string", copy=False)

        self.df.rename(
            columns={
                "Latitude_x": "Latitude",
                "Longitude_x": "Longitude",
                "Município_x": "Município",
                "UF_x": "UF",
                "UF_y": "UF_IBGE",
                "Latitude_y": "Latitude_IBGE",
                "Longitude_y": "Longitude_IBGE",
                "Município_y": "Município_IBGE",
            },
            inplace=True,
        )

        self.replace_bad_city_codes()

    def fill_missing_coords(self) -> None:
        """Fill the missing coordinates with the central coordinates of the city from IBGE"""
        rows = self.log["empty_coords"]
        rows &= self.log["city_normalized"]
        self.log.update({"filled_city_coords": rows})
        columns = ["Latitude_IBGE", "Longitude_IBGE"]
        originals = ["Latitude", "Longitude"]
        log = "Valor Nulo."
        # Lembre-se que colunas float não é possível comparar diretamente como strings
        self._replace_columns(columns, originals, log, rows, float_cols=True)

    def normalize_location_names(self) -> None:
        rows = self.df["Latitude_IBGE"].notna()
        rows &= self.df["Longitude_IBGE"].notna()
        self.log.update({"city_normalized": rows})
        originals = ["Município", "UF"]
        columns = ["Município_IBGE", "UF_IBGE"]
        self.df.loc[:, "Município"] = self.df.loc[:, "Município"].str.title()
        self.df.loc[:, "UF"] = self.df.loc[:, "UF"].str.upper()
        log = ""
        self._replace_columns(columns, originals, log, rows)

    def intersect_coordinates_on_poligon(self):
        """Intersect the coordinates with the shapefile of the IBGE
        Returns a geopandas dataframe with additional columns `CD_MUN, NM_MUN, SIGLA_UF`
        """
        regions = gpd.read_file(self.shapefile)
        # Convert pandas dataframe to geopandas df with geometry point given coordinates
        gdf_points = gpd.GeoDataFrame(
            self.df,
            geometry=gpd.points_from_xy(
                self.df.Longitude.astype("float").fillna(-1),
                self.df.Latitude.astype("float").fillna(-1),
            ),
            crs=regions.crs.to_string(),
        )
        # Set the same coordinate reference system (CRS) as the regions shapefile

        # Spatial join points to the regions
        gdf_joined = gdf_points.sjoin(regions, how="left", predicate="intersects")
        gdf_joined["CD_MUN"] = gdf_joined["CD_MUN"].astype("string", copy=False)
        gdf_joined["NM_MUN"] = gdf_joined["NM_MUN"].astype("string", copy=False)
        gdf_joined["SIGLA_UF"] = gdf_joined["SIGLA_UF"].astype("string", copy=False)
        gdf_joined["LAT"] = gdf_joined.geometry.centroid.y.astype("string", copy=False)
        gdf_joined["LON"] = gdf_joined.geometry.centroid.x.astype("string", copy=False)

        gdf_joined.drop(
            [
                "geometry",
                "AREA_KM2",
                "index_right",
            ],
            axis=1,
            inplace=True,
        )

        self.df = gdf_joined

    def fill_missing_city_info(self):
        """Fill the missing city code
        The missing ones are replaces with the city code derived from the intersection with the shapefile from IBGE"""
        rows = self.df["Código_Município"].isna()
        rows &= self.df["CD_MUN"].notna()
        self.log.update({"filled_city_info": rows})
        originals = ["Código_Município", "Município", "UF"]
        columns = ["CD_MUN", "NM_MUN", "SIGLA_UF"]
        log = "Valor retirado do polígono territorial, à partir das coordenadas."
        self._replace_columns(columns, originals, log, rows)

    def substitute_divergent_coordinates(self):
        """Substitute the coordinates with the centroid from the IBGE `municipios.csv`
        After the intersection of the coordinates with the shapefile, the city code from the data
        should match the one returned from the shapefile.
        If it doesn't the original coordinates are replaced by the ones representing the centroid of the original city code
        """
        # TODO: keep track of "unchanged divergent coordinates, i.e. with IBGE coords null"
        wrong_city_coords = self.df["Código_Município"].notna()
        wrong_city_coords &= self.df["Código_Município"] != self.df["CD_MUN"].fillna(
            "-1"
        )
        wrong_city_coords &= self.log["city_normalized"]
        self.log.update({"wrong_city_coords": wrong_city_coords})
        originals = ["Latitude", "Longitude"]
        columns = ["Latitude_IBGE", "Longitude_IBGE"]
        log = "Coordenadas do Município inseridas."
        self._replace_columns(columns, originals, log, wrong_city_coords, True)

    def input_info_from_coords(self):
        from geopy.geocoders import Nominatim

        rows = self.df["Código_Município"].isna()
        geolocator = Nominatim(user_agent="rfdatahub")
        pbar = tqdm(self.df[rows].itertuples(), total=len(self.df[rows]))
        pp(
            "[bold green]Logging: [/bold green][italic]Requisitando API geocoders para locais fora do perímetro brasileiro."
        )
        for row in pbar:
            pbar.set_description(
                f"Requisitando: ({float(row.Latitude):.4f}, {float(row.Longitude):.4f})"
            )
            location = geolocator.reverse(
                f"{row.Latitude}, {row.Longitude}", exactly_one=True, language="pt"
            )
            if location is None:
                continue
            address = location.raw["address"]
            city = address.get("city", "")
            state = address.get("state", "")
            country = address.get("country", "")
            if any([city, state, country]):
                self.df.loc[row.Index, "Município"] = city
                self.df.loc[row.Index, "UF"] = (
                    f"{state}-{country}" if country != "Brasil" else state
                )

        self.log.update({"city_state_country": rows})

    def _append_filters(self):
        for column, row_filter in self.log.items():
            self.df[column] = False
            self.df.loc[row_filter, column] = True

    def validate(self) -> pd.DataFrame:
        """Helper function to load the IBGE data, enrich and and validate the location information"""
        self.merge_df_with_ibge()
        self.normalize_location_names()
        self.fill_missing_coords()
        self.intersect_coordinates_on_poligon()
        self.substitute_divergent_coordinates()
        self.fill_missing_city_info()
        # self.input_info_from_coords()
        self._append_filters()
        return self.df
