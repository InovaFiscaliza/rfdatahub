import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from extracao.constants import DICT_SRD
from extracao.datasources.smp import Smp
from extracao.datasources.srd import SRD
from extracao.datasources.telecom import Telecom


class MosaicoCacheTests(unittest.TestCase):
    def test_cached_extraction_initializes_log_without_connecting(self):
        for source_class in (Smp, Telecom, SRD):
            with self.subTest(source=source_class.__name__):
                source = source_class(read_cache=True)
                raw = pd.DataFrame({"_id": ["1", "2"]}, dtype="string")
                with patch.object(
                    source, "_read", return_value=raw
                ) as read, patch.object(source, "connect") as connect:
                    result = source._extract(source.collection, [])
                read.assert_called_once_with(f"{source.stem}_raw")
                connect.assert_not_called()
                self.assertEqual(result["Log"].tolist(), ["[]", "[]"])
                self.assertEqual(result["_id"].tolist(), ["1", "2"])

    def test_cached_extraction_starts_a_new_processing_log(self):
        for source_class in (Smp, Telecom, SRD):
            with self.subTest(source=source_class.__name__):
                source = source_class(read_cache=True)
                raw = pd.DataFrame({"Log": ['[{"Processamento": "old"}]']})
                with patch.object(source, "_read", return_value=raw):
                    result = source._extract(source.collection, [])
                self.assertEqual(result["Log"].tolist(), ["[]"])

    def test_srd_update_from_raw_parquet_without_log(self):
        for discard_null_frequency in (False, True):
            with self.subTest(
                discard_null_frequency=discard_null_frequency
            ), tempfile.TemporaryDirectory() as folder:
                source = SRD(read_cache=True)
                source.folder = Path(folder)
                row = {key: "0" for key in DICT_SRD}
                row.update(
                    _id="1",
                    frequency="100",
                    NumServico="230",
                    state="TEST-C1",
                    DataValFreq="2030-01-01",
                    MedPotenciaOperacao="1",
                )
                rows = [row]
                if discard_null_frequency:
                    rows.append(dict(row, _id="2", frequency=None))
                raw = pd.DataFrame(rows, dtype="string")
                source._save(raw, source.folder, "srd_raw")

                with patch.object(source, "connect") as connect:
                    source.update()
                    source.save()

                connect.assert_not_called()
                self.assertNotIn(
                    "Log", pd.read_parquet(source.folder / "srd_raw.parquet")
                )
                result = pd.read_parquet(source.folder / "srd.parquet")
                self.assertEqual(result.columns.tolist(), source.columns)
                self.assertEqual(len(result), 1)
                self.assertEqual(result["Log"].tolist(), ["[]"])
                self.assertEqual(result["Frequência"].tolist(), [100.0])
                self.assertEqual(result["Largura_Emissão(kHz)"].tolist(), ["256.0"])
                self.assertEqual(result["Potência_Transmissor(W)"].tolist(), ["1000.0"])


if __name__ == "__main__":
    unittest.main()
