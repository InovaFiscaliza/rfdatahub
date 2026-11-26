from __future__ import annotations

import ssl
import urllib.request
from functools import cached_property
from zipfile import ZipFile

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from dotenv import find_dotenv, load_dotenv
from fastcore.foundation import L
from fastcore.xtras import Path
from rich import print as pp
from tqdm.auto import tqdm

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
    def log(self) -> dict[str, pd.Series]:
        """Check the coordinates and city code availability"""
        empty_coords = self.df.Latitude.isna() | self.df.Longitude.isna()
        empty_code = self.df.Código_Município.isna()
        empty_coords = empty_coords & (~empty_code)
        empty_code = empty_code & (~empty_coords)
        no_location_info = empty_coords & empty_code
        return {
            "empty_coords": empty_coords,
            "empty_code": empty_code,
            "no_location_info": no_location_info,
        }

    def log_empty_coords(self):
        """Log the rows with empty coordinates"""
        rows = self.log["empty_coords"]
        processing = "Coordenadas nulas."
        Base.register_log(self.df, processing, row_filter=rows)

    def log_empty_code(self):
        """Log the rows with empty city code"""
        rows = self.log["empty_code"]
        processing = "Código do Município nulo."
        Base.register_log(self.df, processing, row_filter=rows)

    def drop_rows_without_location_info(self) -> None:
        rows = self.log["no_location_info"]
        processing = "Coordenadas e Código do Município nulos. Registro descartado"
        # # TODO: append to discarded
        Base.register_log(self.df, processing, row_filter=rows)
        self.discarded = self.df[rows]
        self.df = self.df[~rows]

    def validate_coordinates_as_number(self) -> None:
        for column in ["Latitude", "Longitude"]:
            self.df[column] = self.df[column].astype("string")
            self.df[column] = self.df[column].str.replace(",", ".")
            self.df[column] = self.df[column].str.strip()
            self.df[f"#{column}"] = self.df[column]
            self.df[column] = pd.to_numeric(self.df[column], errors="coerce").astype(
                "float"
            )
            bad_coords = ~self.log["empty_coords"]
            bad_coords &= self.df[column].isna()
            self.log.update({"coords_not_numeric": bad_coords})
            Base.register_log(
                self.df,
                f"Valor Não Numérico Descartado: {column}.",
                f"#{column}",
                bad_coords,
            )
            self.df.drop(columns=[f"#{column}"], inplace=True)

    def validate_codigo_municipio_as_number(self) -> None:
        self.df["Código_Município"] = (
            self.df["Código_Município"]
            .astype("string")
            .str.strip()
            .str.replace(".", "")
            .str[:7]
        )
        self.df["#Código_Município"] = self.df["Código_Município"]
        self.df["Código_Município"] = pd.to_numeric(
            self.df["Código_Município"], errors="coerce"
        ).astype("string")
        bad_codes = ~self.log["empty_code"]
        bad_codes &= self.df["Código_Município"].isna()
        self.log.update({"code_not_numeric": bad_codes})
        Base.register_log(
            self.df,
            "Valor Não Numérico Descartado: Código Município.",
            "#Código_Município",
            bad_codes,
        )
        self.df.drop(columns=["#Código_Município"], inplace=True)

    def _replace_columns(
        self, columns, originals, row_filter, log=None, float_cols=False
    ):
        for column, original in zip(columns, originals):
            col_filter = row_filter
            if not float_cols:
                col_filter = row_filter & (
                    self.df[original]
                    .astype("string")
                    .fillna("")
                    .str.strip()
                    .str.lower()
                    != self.df[column]
                    .astype("string")
                    .fillna("")
                    .str.strip()
                    .str.lower()
                )
            if log:
                self.df[f"#{original}"] = self.df[original].fillna("")
                Base.register_log(self.df, log, f"#{original}", col_filter)
                self.df.drop(columns=[f"#{original}"], inplace=True)
            # Use of col_filter instead of rows to avoid float comparison
            self.df.loc[col_filter, original] = self.df.loc[col_filter, column]

    # def replace_invalid_city_codes(self):
    #     """Tries to fix the wrong city codes by looking at the normalized city names"""
    #     import re
    #     import unicodedata

    #     def remove_punctuation(word):
    #         # Unicode normalize transforma um caracter em seu equivalente em latin.
    #         nfkd = unicodedata.normalize("NFKD", word.lower())
    #         decoded_word = "".join(c for c in nfkd if not unicodedata.combining(c))

    #         return re.sub(r"[^a-zA-Z0-9 ]", "", decoded_word)

    #     municipios = pd.read_csv(
    #         self.ibge,
    #         usecols=["Código_Município", "Município", "UF"],
    #         dtype="string",
    #         dtype_backend="numpy_nullable",
    #     )

    #     municipios["Município_ASCII"] = municipios["Município"].apply(
    #         remove_punctuation
    #     )

    #     self.df["UF"] = self.df["UF"].astype("string").str.strip().str.upper()
    #     self.df["Município"] = self.df["Município"].astype("string").str.strip()
    #     self.df["Município_ASCII"] = (
    #         self.df["Município"].fillna("").apply(remove_punctuation)
    #     )

    #     self.df = pd.merge(
    #         self.df,
    #         municipios,
    #         on=["Município_ASCII", "UF"],
    #         how="left",
    #     )

    #     self.df.rename(
    #         columns={
    #             "Município_x": "Município",
    #             "Código_Município_x": "Código_Município",
    #         },
    #         inplace=True,
    #     )

    #     invalid_code = ~self.log["empty_code"]
    #     invalid_code &= ~self.log["code_not_numeric"]
    #     invalid_code &= self.df["Município_IBGE"].isna()
    #     self.log.update({"invalid_code": invalid_code})

    #     log = "Código do Município não consta no IBGE."
    #     log += "\nMunicípio normalizado e UF serão usados como chave para substituição."
    #     Base.register_log(self.df, log, "Código_Município", invalid_code)
    #     columns = ["Município_y", "Código_Município_y"]
    #     originals = ["Município", "Código_Município"]
    #     # TODO: Verify if there isn't a wrong Município match, given Município is not unique
    #     has_replacement = invalid_code & self.df["Código_Município_y"].notna()
    #     self._replace_columns(columns, originals, log, has_replacement)

    #     self.df.drop(
    #         columns=["Município_y", "Município_ASCII", "Código_Município_y"],
    #         inplace=True,
    #     )

    #     if has_replacement.any():
    #         self.df.drop(
    #             columns=[
    #                 "Latitude_IBGE",
    #                 "Longitude_IBGE",
    #                 "Município_IBGE",
    #                 "UF_IBGE",
    #             ],
    #             inplace=True,
    #         )
    #         self.merge_df_with_ibge()

    #     self.df[["Município", "Código_Município"]] = self.df[
    #         ["Município", "Código_Município"]
    #     ].astype("string")

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
            self.df.astype("string"),
            municipios,
            on="Código_Município",
            how="left",
        ).astype("string")

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

    def fill_missing_coords(self) -> None:
        """Fill the missing coordinates with the central coordinates of the city from IBGE"""
        rows = self.log["empty_coords"]
        rows &= self.log["city_normalized"]
        self.log.update({"filled_city_coords": rows})
        columns = ["Latitude_IBGE", "Longitude_IBGE"]
        originals = ["Latitude", "Longitude"]
        log = "IBGE: Coordenadas Nulas -> Coordenadas do Município"
        Base.register_log(self.df, log, row_filter=rows)
        # Lembre-se que colunas float não é possível comparar diretamente como strings
        self._replace_columns(columns, originals, rows, float_cols=True)

    def normalize_location_names(self) -> None:
        self.df.loc[:, "UF"] = (
            self.df.loc[:, "UF"].astype("string").str.strip().str.upper()
        )
        self.df.loc[:, "Município"] = (
            self.df.loc[:, "Município"].astype("string").str.strip()
        )
        has_ibge_city = (
            self.df["Latitude_IBGE"].notna()
            & self.df["Longitude_IBGE"].notna()
            & self.df["Município_IBGE"].notna()
            & self.df["UF_IBGE"].notna()
        )
        self.log.update({"city_normalized": has_ibge_city})
        # name_or_uf_differs = (
        #     self.df["Município"].fillna("").str.lower()
        #     != self.df["Município_IBGE"].str.lower()
        # ) | (
        #     self.df["UF"].fillna("").str.upper()
        #     != self.df["UF_IBGE"].str.upper()
        # )
        # rows = has_ibge_city & name_or_uf_differs
        originals = ["Município", "UF"]
        columns = ["Município_IBGE", "UF_IBGE"]
        self.df.loc[has_ibge_city, originals] = self.df.loc[
            has_ibge_city, columns
        ].values
        # log = "Nomes de Municípios e UFs normalizados (IBGE)"
        # self._replace_columns(columns, originals, log, has_ibge_city)

    def intersect_coordinates_on_poligon(self):
        """Intersect the coordinates with the shapefile of the IBGE
        Returns a geopandas dataframe with additional columns `CD_MUN, NM_MUN, SIGLA_UF`
        """
        regions = gpd.read_file(self.shapefile)
        if crs := getattr(regions, "crs", None):
            crs = crs.to_string()
        # Convert pandas dataframe to geopandas df with geometry point given coordinates
        gdf_points = gpd.GeoDataFrame(
            self.df,
            geometry=gpd.points_from_xy(
                self.df.Longitude.astype("float").fillna(-1),
                self.df.Latitude.astype("float").fillna(-1),
            ),
            crs=crs,
        )
        # Set the same coordinate reference system (CRS) as the regions shapefile

        # Spatial join points to the regions
        gdf_joined = gdf_points.sjoin(regions, how="left", predicate="intersects")
        gdf_joined["CD_MUN"] = gdf_joined["CD_MUN"].astype("string")
        gdf_joined["NM_MUN"] = gdf_joined["NM_MUN"].astype("string")
        gdf_joined["SIGLA_UF"] = gdf_joined["SIGLA_UF"].astype("string")
        gdf_joined["LAT"] = gdf_joined.geometry.centroid.y.astype("string")
        gdf_joined["LON"] = gdf_joined.geometry.centroid.x.astype("string")

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
        self.fix_inverted_or_swapped_coordinates(regions)
        self.fix_decimal_shift_coordinates(regions)

        coords_inside_br = (
            self.df["CD_MUN"].notna()
            & self.df["LAT"].notna()
            & self.df["LON"].notna()
            & self.df["SIGLA_UF"].notna()
        )
        self.log.update({"coords_inside_br": coords_inside_br})
        Base.register_log(
            self.df,
            "Fora do Território Continental BR.",
            row_filter=~coords_inside_br,
        )

    def fix_inverted_or_swapped_coordinates(self, regions: gpd.GeoDataFrame) -> None:
        """Recover coordinates outside Brazilian land polygons caused by sign inversion or Lat/Lon swap.
        Only applies a candidate transform when the resulting point intersects an IBGE polygon matching
        the record's declared `Código_Município` (or `UF`), preventing offshore false positives.
        """
        lat = pd.to_numeric(self.df["Latitude"], errors="coerce")
        lon = pd.to_numeric(self.df["Longitude"], errors="coerce")
        outside = self.df["CD_MUN"].isna() & lat.notna() & lon.notna()
        has_decl = self.df["Código_Município"].notna() | self.df["UF"].notna()
        if not (outside & has_decl).any():
            return

        tree = shapely.STRtree(regions.geometry.to_numpy())
        transforms = (
            (
                "Sinal da Longitude invertido (Leste -> Oeste).",
                lambda la, lo: (la, -lo),
                ("Longitude",),
            ),
            (
                "Sinal da Latitude invertido.",
                lambda la, lo: (-la, lo),
                ("Latitude",),
            ),
            (
                "Sinais de Latitude e Longitude invertidos.",
                lambda la, lo: (-la, -lo),
                ("Latitude", "Longitude"),
            ),
            (
                "Latitude e Longitude trocadas entre si.",
                lambda la, lo: (lo, la),
                ("Latitude", "Longitude"),
            ),
        )

        for reason, fn, logged_cols in transforms:
            cand_mask = self.df["CD_MUN"].isna() & lat.notna() & lon.notna() & has_decl
            cand_idx = self.df.index[cand_mask]
            if len(cand_idx) == 0:
                break

            cand_lat, cand_lon = fn(
                lat.loc[cand_idx].to_numpy(dtype="float64"),
                lon.loc[cand_idx].to_numpy(dtype="float64"),
            )
            valid_wgs = (np.abs(cand_lat) <= 90.0) & (np.abs(cand_lon) <= 180.0)
            pts = shapely.points(
                np.where(valid_wgs, cand_lon, 0.0),
                np.where(valid_wgs, cand_lat, 0.0),
            )
            pt_i, poly_i = tree.query(pts, predicate="intersects")
            if len(pt_i) == 0:
                continue
            _, first_occ = np.unique(pt_i, return_index=True)
            pt_i, poly_i = pt_i[first_occ], poly_i[first_occ]
            keep = valid_wgs[pt_i]
            pt_i, poly_i = pt_i[keep], poly_i[keep]
            if len(pt_i) == 0:
                continue

            hit_df_idx = cand_idx[pt_i]
            hit_cd = regions["CD_MUN"].iloc[poly_i].astype("string").to_numpy()
            hit_nm = regions["NM_MUN"].iloc[poly_i].astype("string").to_numpy()
            hit_uf = regions["SIGLA_UF"].iloc[poly_i].astype("string").to_numpy()

            decl_uf = self.df.loc[hit_df_idx, "UF"].fillna("").to_numpy()
            confirmed = (decl_uf != "") & (decl_uf == hit_uf)
            if not confirmed.any():
                continue

            fix_idx = hit_df_idx[confirmed]
            row_filter = pd.Series(False, index=self.df.index)
            row_filter.loc[fix_idx] = True

            msg = f"Coordenadas corrigidas: {reason}"
            for col in logged_cols:
                self.df[f"#{col}"] = self.df[col].astype("string").fillna("")
                Base.register_log(self.df, msg, f"#{col}", row_filter)
                self.df.drop(columns=[f"#{col}"], inplace=True)

            new_la_str = pd.Series(cand_lat[pt_i[confirmed]], index=fix_idx).astype(
                "string"
            )
            new_lo_str = pd.Series(cand_lon[pt_i[confirmed]], index=fix_idx).astype(
                "string"
            )
            self.df.loc[fix_idx, "Latitude"] = new_la_str
            self.df.loc[fix_idx, "Longitude"] = new_lo_str
            self.df.loc[fix_idx, "CD_MUN"] = hit_cd[confirmed]
            self.df.loc[fix_idx, "NM_MUN"] = hit_nm[confirmed]
            self.df.loc[fix_idx, "SIGLA_UF"] = hit_uf[confirmed]
            self.df.loc[fix_idx, "LAT"] = new_la_str
            self.df.loc[fix_idx, "LON"] = new_lo_str

    def fix_decimal_shift_coordinates(self, regions: gpd.GeoDataFrame) -> None:
        """Shift one decimal place, preferring a municipality match over a UF match."""
        region_codes = regions["CD_MUN"].astype("string")
        eligible = self.df["CD_MUN"].isna() & self.df["Código_Município"].isin(
            region_codes.dropna()
        )
        filled_coords = self.log.get("filled_city_coords")
        if filled_coords is not None:
            eligible &= ~filled_coords

        columns = ["Latitude", "Longitude"]
        coords = (
            self.df.loc[eligible, columns]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype="float64", na_value=np.nan)
        )
        wgs84_limits = np.array([90.0, 180.0])
        # Values outside these bounds cannot be recovered by a single decimal shift.
        usable = ((np.abs(coords) <= wgs84_limits * 10) & (coords != -1)).all(axis=1)
        if not usable.any():
            return

        coords = coords[usable]
        source_index = self.df.index[eligible][usable]
        declared_cd = self.df.loc[source_index, "Código_Município"].to_numpy()
        declared_uf = self.df.loc[source_index, "UF"].fillna("").to_numpy()
        region_cd = region_codes.fillna("").to_numpy()
        region_uf = regions["SIGLA_UF"].astype("string").fillna("").to_numpy()
        polygons = regions.geometry.to_numpy()

        candidates = []
        for axis, column in enumerate(columns):
            for operation, scale in (("%", 10.0), ("x", 0.1)):
                shifted = coords.copy()
                shifted[:, axis] *= scale
                valid = ((np.abs(shifted) <= wgs84_limits) & (shifted != -1)).all(
                    axis=1
                )
                valid &= shifted[:, axis] != coords[:, axis]
                positions = np.flatnonzero(valid)
                points = shapely.points(shifted[positions, 1], shifted[positions, 0])
                pt_i, poly_i = regions.sindex.query(points, predicate="intersects")

                unique = np.bincount(pt_i, minlength=len(points))[pt_i] == 1
                inside = unique & shapely.within(points[pt_i], polygons[poly_i])
                source = positions[pt_i[inside]]
                polygon = poly_i[inside]
                exact = declared_cd[source] == region_cd[polygon]
                same_uf = (declared_uf[source] != "") & (
                    declared_uf[source] == region_uf[polygon]
                )
                confirmed = (exact | same_uf) & (region_cd[polygon] != "")
                if confirmed.any():
                    candidates.append(
                        pd.DataFrame(
                            {
                                "column": column,
                                "operation": operation,
                                "value": shifted[source[confirmed], axis],
                                "polygon": polygon[confirmed],
                                "exact": exact[confirmed],
                            },
                            index=source_index[source[confirmed]],
                        )
                    )

        if not candidates:
            return
        matches = pd.concat(candidates)
        best_level = matches.groupby(level=0)["exact"].transform("max")
        fixes = matches[matches["exact"] == best_level]
        fixes = fixes[~fixes.index.duplicated(keep=False)]
        if fixes.empty:
            return

        self.df["_decimal_coordinate"] = fixes["value"].astype("string")
        for (column, operation, exact), group in fixes.groupby(
            ["column", "operation", "exact"]
        ):
            scope = "mesmo município" if exact else "mesma UF"
            message = f"Coordenadas corrigidas ({scope}): {column} {operation} por 10."
            rows = pd.Series(self.df.index.isin(group.index), index=self.df.index)
            self._replace_columns(
                ["_decimal_coordinate"], [column], rows, message, float_cols=True
            )
        del self.df["_decimal_coordinate"]

        polygon_cols = ["CD_MUN", "NM_MUN", "SIGLA_UF"]
        self.df.loc[fixes.index, polygon_cols] = (
            regions.iloc[fixes["polygon"]][polygon_cols].astype("string").to_numpy()
        )
        self.df.loc[fixes.index, ["LAT", "LON"]] = self.df.loc[
            fixes.index, columns
        ].to_numpy()

    def substitute_invalid_outside_coordinates(self) -> None:
        """Replace physically impossible or Null Island coordinates (e.g. |Lat|>60, (0,0)) with IBGE centroid
        when a valid IBGE municipality exists, without touching legitimate offshore/border stations.
        """
        lat = pd.to_numeric(self.df["Latitude"], errors="coerce")
        lon = pd.to_numeric(self.df["Longitude"], errors="coerce")
        outside = self.df["CD_MUN"].isna() & lat.notna() & lon.notna()
        absurd = (
            (lat.abs() > 60.0)
            | (lon.abs() > 180.0)
            | ((lat.abs() < 1.0) & (lon.abs() < 1.0))
            | (lat.abs() < 1e-4)
            | (lon.abs() < 1e-4)
        )
        rows = outside & absurd & self.log["city_normalized"]
        self.log.update({"absurd_coords_replaced": rows})
        log = "IBGE: Coordenadas inválidas."
        self._replace_columns(
            ["Latitude_IBGE", "Longitude_IBGE"],
            ["Latitude", "Longitude"],
            rows,
            log,
            float_cols=True,
        )

    def fill_missing_city_info(self):
        """Fill the missing city code
        The missing ones are replaces with the city code derived from the intersection with the shapefile from IBGE"""
        rows = self.df["Código_Município"].isna()
        rows &= self.df["CD_MUN"].notna()
        self.log.update({"filled_city_info": rows})
        originals = ["Código_Município", "Município", "UF"]
        columns = ["CD_MUN", "NM_MUN", "SIGLA_UF"]
        log = "IBGE: Dado derivado das coordenadas."
        self._replace_columns(columns, originals, rows, log)

    def polygon2cities(self):
        """Substitute the city and or UF with the centroid from the IBGE `municipios.csv`
        After the intersection of the coordinates with the shapefile, the city code from the data
        should match the one returned from the shapefile.
        If it doesn't the divergence is logged but the original coordinates are kept replaced by the ones representing the centroid of the original city code
        """
        # TODO: keep track of "unchanged divergent coordinates, i.e. with IBGE coords null"
        wrong_city_coords = (
            self.df["Código_Município"].notna() & self.df["CD_MUN"].notna()
        )
        wrong_city_coords &= self.df["Código_Município"] != self.df["CD_MUN"]
        self.log.update({"wrong_city_coords": wrong_city_coords})
        log = "IBGE: Município distinto."
        Base.register_log(self.df, log, "NM_MUN", wrong_city_coords)

    # def input_info_from_coords(self):
    #     from geopy.geocoders import Nominatim

    #     rows = self.df["Código_Município"].isna()
    #     geolocator = Nominatim(user_agent="rfdatahub")
    #     pbar = tqdm(self.df[rows].itertuples(), total=len(self.df[rows]))
    #     pp(
    #         "[bold green]Logging: [/bold green][italic]Requisitando API geocoders para locais fora do perímetro brasileiro."
    #     )
    #     for row in pbar:
    #         pbar.set_description(
    #             f"Requisitando: ({float(row.Latitude):.4f}, {float(row.Longitude):.4f})"
    #         )
    #         location = geolocator.reverse(
    #             f"{row.Latitude}, {row.Longitude}", exactly_one=True, language=True
    #         )
    #         if location is None:
    #             continue
    #         address = location.raw["address"]
    #         city = address.get("city", "")
    #         state = address.get("state", "")
    #         country = address.get("country", "")
    #         if any([city, state, country]):
    #             self.df.loc[row.Index, "Município"] = city
    #             self.df.loc[row.Index, "UF"] = (
    #                 f"{state}-{country}" if country != "Brasil" else state
    #             )

    #     self.log.update({"city_state_country": rows})

    def _append_filters(self):
        for column, row_filter in self.log.items():
            self.df[column] = False
            self.df.loc[row_filter, column] = True

    def validate(self) -> pd.DataFrame:
        """Helper function to load the IBGE data, enrich and and validate the location information"""
        self.merge_df_with_ibge()
        # self.replace_invalid_city_codes()
        self.normalize_location_names()
        self.fill_missing_coords()
        self.intersect_coordinates_on_poligon()
        self.substitute_invalid_outside_coordinates()
        self.polygon2cities()
        self.fill_missing_city_info()

        # self.input_info_from_coords()
        self._append_filters()
        return self.df
