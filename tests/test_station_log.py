import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from extracao.datasources.base import Base
from extracao.stations import Estacoes


class StationLogTests(unittest.TestCase):
    def test_processing_logs_are_saved_and_referenced_by_station_rows(self):
        stations = Estacoes(read_cache=True)
        df = pd.DataFrame(
            {
                "Latitude": ["invalid", "invalid", "valid"],
                "Longitude": ["?", "?", "0"],
            },
            dtype="string",
        )
        affected = df["Latitude"] == "invalid"
        Base.register_log(df, "Invalid coordinate", "Latitude", affected)
        Base.register_log(df, "Invalid coordinate", "Longitude", affected)
        Base.register_log(df, "Invalid coordinate", row_filter=affected)
        df.loc[affected, "Latitude"] = "still invalid"
        Base.register_log(df, "Invalid coordinate", "Latitude", affected)

        with tempfile.TemporaryDirectory() as folder:
            stations.folder = Path(folder)
            processed = stations.process_log(df)
            logs = pd.read_parquet(stations.folder / "log.parquet")["Log"].map(
                json.loads
            )

        self.assertEqual(len(logs), 2)
        self.assertEqual(processed["Log"].nunique(), 2)
        self.assertEqual(
            logs.iloc[processed["Log"].iloc[0]],
            [
                {
                    "Log": "Invalid coordinate",
                    "Coluna": "Latitude",
                    "Original": ["invalid", "still invalid"],
                },
                {
                    "Log": "Invalid coordinate",
                    "Coluna": "Longitude",
                    "Original": "?",
                },
                {"Log": "Invalid coordinate"},
            ],
        )
        self.assertEqual(processed["Log"].iloc[0], processed["Log"].iloc[1])
        self.assertEqual(logs.iloc[processed["Log"].iloc[2]], [])

    def test_grouped_station_ids_are_still_sorted_and_deduplicated(self):
        stations = Estacoes(read_cache=True)
        df = pd.DataFrame({"#Estação": ["['2', '1', '2']"]})
        Base.register_log(df, "Grouped stations", "#Estação")

        with tempfile.TemporaryDirectory() as folder:
            stations.folder = Path(folder)
            processed = stations.process_log(df)
            logs = pd.read_parquet(stations.folder / "log.parquet")["Log"]

        self.assertEqual(
            json.loads(logs.iloc[processed["Log"].iloc[0]]),
            [
                {
                    "Log": "Grouped stations",
                    "Coluna": "#Estação",
                    "Original": "['1', '2']",
                },
            ],
        )
