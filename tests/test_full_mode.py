import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from extracao.constants import AGG_LICENCIAMENTO, COLS_STEL
from extracao.datasources.mosaico import Mosaico
from extracao.datasources.smp import Smp
from extracao.datasources.telecom import Telecom


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


class FullOutputStemTests(unittest.TestCase):
    def test_full_sources_read_shared_raw_cache_and_save_separate_outputs(self):
        for source_class in (Telecom, Smp):
            with (
                self.subTest(source=source_class.__name__),
                tempfile.TemporaryDirectory() as folder,
            ):
                source = source_class(read_cache=True, full=True)
                self.assertEqual(source.output_stem, f"{source.stem}_full")
                source.folder = Path(folder)
                processed = pd.DataFrame(
                    {"result": ["full"]},
                    dtype="string",
                )
                with (
                    patch.object(
                        source,
                        "_read",
                        return_value=pd.DataFrame({"_id": ["raw"]}, dtype="string"),
                    ) as read,
                    patch.object(source, "_format", return_value=processed),
                ):
                    source.update()
                read.assert_called_once_with(f"{source.stem}_raw")

                source.save()

                output = Path(folder) / f"{source.stem}_full.parquet"
                discarded = Path(folder) / f"{source.stem}_full_discarded.parquet"
                self.assertTrue(output.is_file())
                self.assertTrue(discarded.is_file())
                self.assertEqual(pd.read_parquet(output)["result"].tolist(), ["full"])

                cached_source = source_class(read_cache=True, full=True)
                cached_source.folder = Path(folder)
                self.assertEqual(cached_source.df["result"].tolist(), ["full"])

    def test_grouped_sources_keep_their_existing_output_stems(self):
        for source_class in (Telecom, Smp):
            with (
                self.subTest(source=source_class.__name__),
                tempfile.TemporaryDirectory() as folder,
            ):
                source = source_class(read_cache=True)
                self.assertEqual(source.output_stem, source.stem)
                source.folder = Path(folder)
                source.df = pd.DataFrame({"result": ["grouped"]})
                source.save()
                self.assertTrue((Path(folder) / f"{source.stem}.parquet").is_file())
                self.assertTrue(
                    (Path(folder) / f"{source.stem}_discarded.parquet").is_file()
                )


class TelecomFullFormatTests(unittest.TestCase):
    @staticmethod
    def make_raw_rows():
        base = {
            "NumAto": "123",
            "NumFistel": "10",
            "NumServico": "011",
            "NomeEntidade": "Test Entity",
            "SiglaUf": "SP",
            "CodTipoClasseEstacao": "FB",
            "NomeMunicipio": "São Paulo",
            "CodMunicipio": "3550308",
            "DataValidade": "2030-01-01",
            "FreqTxMHz": "100",
            "formId": "1",
            "Tecnologia": "LTE",
            "Latitude": "-23.5",
            "Longitude": "-46.6",
            "DesignacaoEmissao": "200KG7W",
            "state": "LIC-LIC-01",
            "Log": "[]",
        }
        return pd.DataFrame(
            [
                dict(base, NumEstacao="2"),
                dict(base, NumEstacao="1"),
            ],
            dtype="string",
        )

    def test_full_format_preserves_duplicates_and_saves_without_multiplicity(self):
        source = Telecom(read_cache=True, full=True)
        result = source._format(self.make_raw_rows())

        self.assertEqual(len(result), 2)
        self.assertEqual(result["Estação"].tolist(), [1, 2])
        self.assertNotIn("Multiplicidade", result.columns)
        self.assertEqual(
            result.columns.tolist(),
            [column for column in COLS_STEL if column != "Multiplicidade"],
        )

        with tempfile.TemporaryDirectory() as folder:
            source.folder = Path(folder)
            source.df = result
            source.save()
            saved = pd.read_parquet(Path(folder) / "telecom_full.parquet")

        self.assertEqual(len(saved), 2)
        self.assertNotIn("Multiplicidade", saved.columns)


if __name__ == "__main__":
    unittest.main()
