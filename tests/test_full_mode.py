import json
import unittest

import pandas as pd

from extracao.constants import AGG_LICENCIAMENTO
from extracao.datasources.mosaico import Mosaico


class MosaicoUngroupedTests(unittest.TestCase):
    @staticmethod
    def make_rows():
        return pd.DataFrame(
            {
                "Estação": ["2", "1", "3"],
                "Frequência": ["100", "100", "200"],
                "Fistel": ["10", "10", "20"],
                "Código_Município": ["1001", "1001", "1002"],
                "Longitude": ["-46", "-46", "-47"],
                "Latitude": ["-23", "-23", pd.NA],
                "Classe": ["FB", "FB", "FX"],
                "Serviço": ["010", "010", "011"],
                "Classe_Emissão": ["G7W", "G7W", "F3E"],
                "Largura_Emissão(kHz)": ["200", "200", "25"],
                "Log": ["[]", "[]", "[]"],
            },
            dtype="string",
        )

    def test_ungrouped_keeps_duplicates_and_rows_with_null_grouping_values(self):
        result = Mosaico.ungrouped(self.make_rows())

        self.assertEqual(len(result), 3)
        self.assertEqual(result["Estação"].tolist(), [1, 2, 3])
        self.assertEqual(result["Multiplicidade"].tolist(), [1, 1, 1])
        self.assertTrue(result["Log"].eq("[]").all())
        self.assertNotIn("#Estação", result.columns)
        self.assertEqual(result.loc[0, "Largura_Emissão(kHz)"], 200)
        self.assertTrue(pd.isna(result.loc[2, "Latitude"]))

    def test_aggregate_still_collapses_duplicates_and_logs_station_ids(self):
        result = Mosaico.aggregate(
            self.make_rows(), AGG_LICENCIAMENTO, "MOSAICO-TELECOM: Agrupamento"
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result["Multiplicidade"].tolist(), [2])
        self.assertEqual(
            json.loads(result.loc[0, "Log"])[0]["Coluna"], "#Estação"
        )
        self.assertEqual(
            json.loads(result.loc[0, "Log"])[0]["Original"], "[1, 2]"
        )


if __name__ == "__main__":
    unittest.main()
