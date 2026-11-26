import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import full_pipeline_check


class OfflineInputTests(unittest.TestCase):
    def test_missing_shapefile_components_fail_before_processing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("telecom", "smp", "srd", "stel", "radcom"):
                (root / f"{name}_raw.parquet").touch()
            (root / "aero.parquet").touch()
            municipalities = root / "municipios.csv"
            municipalities.touch()
            shapefile = root / "municipios.shp"
            for suffix in (".shp", ".cpg", ".dbf", ".prj"):
                shapefile.with_suffix(suffix).touch()

            with (
                patch.object(full_pipeline_check, "CACHE_DIR", root),
                patch.object(full_pipeline_check, "IBGE_MUNICIPIOS", municipalities),
                patch.object(full_pipeline_check, "IBGE_POLIGONO", shapefile),
            ):
                with self.assertRaisesRegex(FileNotFoundError, r"municipios\.shx"):
                    full_pipeline_check._check_offline_inputs()
                shapefile.with_suffix(".shx").touch()
                full_pipeline_check._check_offline_inputs()


if __name__ == "__main__":
    unittest.main()
