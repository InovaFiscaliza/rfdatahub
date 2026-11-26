import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

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


class FrameComparisonTests(unittest.TestCase):
    def test_equal_frames_pass(self):
        frame = pd.DataFrame({"value": ["expected"]})

        full_pipeline_check.assert_frames_equal("tiny", frame, frame.copy())

    def test_perturbed_frame_reports_shape_columns_and_first_difference(self):
        actual = pd.DataFrame({"value": ["actual"]})
        expected = pd.DataFrame({"value": ["baseline"]})

        with self.assertRaisesRegex(
            AssertionError,
            "tiny differs from baseline.*first differing rows=.*actual.*baseline",
        ):
            full_pipeline_check.assert_frames_equal("tiny", actual, expected)


if __name__ == "__main__":
    unittest.main()
