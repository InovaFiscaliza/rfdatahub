import json
import unittest
from unittest.mock import patch

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from extracao.format import merge_dicts
from extracao.location import Geography


class DecimalCoordinateTests(unittest.TestCase):
    def validate(self, rows):
        regions = gpd.GeoDataFrame(
            [
                ("3550308", "São Paulo", "SP", box(-46.7, -23.6, -46.5, -23.4)),
                ("3509502", "Campinas", "SP", box(-47.2, -23, -46.8, -22.8)),
                ("3106200", "Belo Horizonte", "MG", box(-44.1, -20, -43.9, -19.8)),
                ("1501402", "Belém", "PA", box(-53.1, -3.1, -52.9, -2.9)),
                ("1504208", "Marabá", "PA", box(-53.1, -0.04, -52.9, -0.02)),
                ("1500800", "Ananindeua", "PA", box(-52.6, -0.2, -52.4, -0.05)),
            ],
            columns=["CD_MUN", "NM_MUN", "SIGLA_UF", "geometry"],
            geometry="geometry",
        ).assign(AREA_KM2=1)
        source = pd.DataFrame(
            rows, columns=["Código_Município", "UF", "Latitude", "Longitude"]
        ).assign(Município=None)
        with (
            patch.object(Geography, "check_files"),
            patch("extracao.location.gpd.read_file", return_value=regions),
        ):
            return Geography(source).validate()

    def test_replaces_coordinates_and_logs_original_by_match_level(self):
        exact, same_uf, preferred = self.validate(
            [
                ("3550308", "SP", "-235", "-46.6"),
                ("3550308", "SP", "-22.9", "-4.7"),
                ("1504208", "PA", "-0.3", "-53"),
            ]
        ).itertuples(index=False)

        self.assertEqual(
            (float(exact.Latitude), float(exact.Longitude)), (-23.5, -46.6)
        )
        self.assertEqual(exact.CD_MUN, "3550308")
        self.assertEqual(
            json.loads(merge_dicts(exact.Log)),
            [
                {
                    "Log": "Coordenadas corrigidas (casa decimal, mesmo município): Latitude dividida por 10.",
                    "Coluna": "#Latitude",
                    "Original": "-235.0",
                }
            ],
        )

        self.assertEqual(
            (float(same_uf.Latitude), float(same_uf.Longitude)), (-22.9, -47.0)
        )
        self.assertEqual(
            (same_uf.Código_Município, same_uf.CD_MUN, same_uf.SIGLA_UF),
            ("3550308", "3509502", "SP"),
        )
        self.assertEqual(
            json.loads(merge_dicts(same_uf.Log)),
            [
                {
                    "Log": "Coordenadas corrigidas (casa decimal, mesma UF): Longitude multiplicada por 10.",
                    "Coluna": "#Longitude",
                    "Original": "-4.7",
                },
                {
                    "Log": "IBGE: Município distinto.",
                    "Coluna": "NM_MUN",
                    "Original": "Campinas",
                },
            ],
        )
        self.assertEqual(
            (preferred.CD_MUN, float(preferred.Latitude)), ("1504208", -0.03)
        )

    def test_preserves_inversions_centroid_fallback_and_unconfirmed_points(self):
        result = self.validate(
            [
                ("3550308", "SP", "23.5", "-46.6"),  # Existing sign inversion
                ("3550308", "SP", "-95", "-46.6"),  # Existing centroid fallback
                ("3550308", "SP", "-22.9", "-47"),  # Already inside a different city
                ("3550308", "SP", "-19.9", "-4.4"),  # Candidate is in another UF
                (None, None, "-2.35", "-46.6"),  # No declared location
                ("1500800", "PA", "-0.3", "-53"),  # Two same-UF candidates
                ("1500800", "PA", "-1", "-52.5"),  # A sentinel would create a candidate
                (
                    "3550308",
                    "SP",
                    "-23.4",
                    "-4.66",
                ),  # Candidate is on the polygon boundary
            ]
        )
        self.assertEqual(result.at[0, "Latitude"], "-23.5")
        self.assertIn("Sinal da Latitude invertido.", result.at[0, "Log"])
        self.assertEqual(result.at[1, "Latitude"], result.at[1, "Latitude_IBGE"])
        self.assertIn("IBGE: Coordenadas inválidas.", result.at[1, "Log"])
        coords = result.loc[2:, ["Latitude", "Longitude"]].astype(float)
        self.assertEqual(
            list(coords.itertuples(index=False, name=None)),
            [
                (-22.9, -47),
                (-19.9, -4.4),
                (-2.35, -46.6),
                (-0.3, -53),
                (-1, -52.5),
                (-23.4, -4.66),
            ],
        )
        self.assertFalse(result["Log"].str.contains("casa decimal").any())
