import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from fastcore.foundation import L

from extracao.constants import (
    AGG_LICENCIAMENTO,
    COL_GRUPO,
    COL_GRUPO_PRINCIPAL,
    COLS_STEL,
)
from extracao.datasources.mosaico import Mosaico, full_columns
from extracao.datasources.smp import Smp
from extracao.datasources.telecom import Telecom
from extracao.stations import Estacoes


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
            full_columns(COLS_STEL, (COL_GRUPO, COL_GRUPO_PRINCIPAL)),
        )
        self.assertTrue(result["Tecnologia"].notna().all())
        self.assertEqual(result[COL_GRUPO].nunique(), 1)

        with tempfile.TemporaryDirectory() as folder:
            source.folder = Path(folder)
            source.df = result
            source.save()
            saved = pd.read_parquet(Path(folder) / "telecom_full.parquet")

        self.assertEqual(len(saved), 2)
        self.assertNotIn("Multiplicidade", saved.columns)
        self.assertEqual(result[COL_GRUPO_PRINCIPAL].tolist(), ["True", "False"])


class SmpFullFormatTests(unittest.TestCase):
    def test_full_format_preserves_duplicate_downlink_and_generated_uplink_rows(self):
        raw = TelecomFullFormatTests.make_raw_rows()
        raw["FreqTxMHz"] = "461.05"
        raw["Tecnologia"] = "4G"
        source = Smp(read_cache=True, full=True)
        result = source._format(raw)

        downlink = result[result["Fonte"] == "MOSAICO-LICENCIAMENTO"]
        uplink = result[result["Fonte"] == "CANALIZACAO-SMP"]
        self.assertEqual(len(downlink), 2)
        self.assertEqual(len(uplink), 2)
        self.assertEqual(uplink["Classe"].tolist(), ["ML", "ML"])
        self.assertEqual(downlink["Latitude"].tolist(), ["-23.5", "-23.5"])
        self.assertEqual(downlink["Longitude"].tolist(), ["-46.6", "-46.6"])
        self.assertNotIn("Multiplicidade", result.columns)
        self.assertEqual(
            result.columns.tolist(),
            full_columns(COLS_STEL, (COL_GRUPO, COL_GRUPO_PRINCIPAL)),
        )
        self.assertTrue(result["Tecnologia"].notna().all())
        self.assertEqual(result[COL_GRUPO].nunique(), 1)
        self.assertFalse(
            result["Log"].str.contains("Agrupamento: Latitude do Município").any()
        )

        with tempfile.TemporaryDirectory() as folder:
            source.folder = Path(folder)
            source.df = result
            source.save()
            saved = pd.read_parquet(Path(folder) / "smp_full.parquet")

        self.assertEqual(len(saved), 4)
        self.assertNotIn("Multiplicidade", saved.columns)


class StationConsolidationTests(unittest.TestCase):
    def test_consolidate_can_reuse_input_sources_without_mutating_the_list(self):
        anatel = pd.DataFrame(
            {
                "Frequência": ["100", "200"],
                "Latitude": ["-23", "-24"],
                "Longitude": ["-46", "-47"],
                "Fonte": ["STEL", "STEL"],
                "Log": ["[]", "[]"],
            },
            dtype="string",
        )
        aero = pd.DataFrame(
            {
                "Frequência": ["300"],
                "Latitude": ["-25"],
                "Longitude": ["-48"],
                "Fonte": ["AERO"],
                "Log": ["[]"],
            },
            dtype="string",
        )
        sources = L([anatel, aero])
        stations = Estacoes(read_cache=True)

        class IdentityGeography:
            def __init__(self, df):
                self.df = df

            def validate(self):
                return self.df

        with (
            patch(
                "extracao.stations.merge_on_frequency",
                side_effect=lambda left, right: pd.concat(
                    [left, right], ignore_index=True
                ),
            ),
            patch("extracao.stations.Geography", IdentityGeography),
            patch.object(Estacoes, "_simplify_sources", side_effect=lambda df: df),
            patch.object(
                Estacoes, "_remove_invalid_frequencies", side_effect=lambda df: df
            ),
        ):
            first = stations._consolidate(sources)
            second = stations._consolidate(sources)

        self.assertEqual(len(sources), 2)
        pd.testing.assert_frame_equal(sources[0], anatel)
        pd.testing.assert_frame_equal(sources[1], aero)
        self.assertEqual(len(first), 3)
        pd.testing.assert_frame_equal(first, second)


class InlineLogTests(unittest.TestCase):
    def test_inline_log_matches_process_log_and_handles_empty_sentinels(self):
        logs = [
            "[]",
            '[{"Log":"source","Coluna":"Latitude","Original":"-23"}]',
            '[{"Log":"source","Coluna":"Latitude","Original":"-24"}]',
        ]
        df = pd.DataFrame({"Log": logs}, dtype="string")
        stations = Estacoes(read_cache=True)

        with tempfile.TemporaryDirectory() as folder:
            stations.folder = Path(folder)
            processed = stations.process_log(df.copy())
            cached_logs = pd.read_parquet(Path(folder) / "log.parquet")

        with tempfile.TemporaryDirectory() as folder:
            stations.folder = Path(folder)
            inline = Estacoes.inline_log(df.copy())
            for index, label in enumerate(processed["Log"]):
                self.assertEqual(
                    json.loads(inline.loc[index, "Log"]),
                    json.loads(cached_logs.loc[label, "Log"]),
                )
            sentinel_logs = pd.DataFrame(
                {"Log": pd.Series(["-1", pd.NA], dtype="string")}
            )
            sentinels = Estacoes.inline_log(sentinel_logs)
            self.assertEqual(sentinels["Log"].tolist(), ["[]", "[]"])
            self.assertFalse((Path(folder) / "log.parquet").exists())


class TagGroupsTests(unittest.TestCase):
    def test_tags_match_the_rows_aggregate_keeps(self):
        rows = MosaicoUngroupedTests.make_rows()
        tagged = Mosaico.tag_groups(
            Mosaico.ungrouped(rows.copy()), AGG_LICENCIAMENTO, "telecom"
        )
        grouped = Mosaico.aggregate(rows, AGG_LICENCIAMENTO, "x")

        self.assertEqual(tagged[COL_GRUPO].tolist()[:2], ["telecom:0", "telecom:0"])
        self.assertTrue(pd.isna(tagged.loc[2, COL_GRUPO]))
        self.assertEqual(tagged[COL_GRUPO_PRINCIPAL].tolist(), ["True", "False", "False"])
        principal = tagged[tagged[COL_GRUPO_PRINCIPAL] == "True"]
        self.assertEqual(principal["Estação"].tolist(), grouped["Estação"].tolist())


class StationGroupRowsTests(unittest.TestCase):
    @staticmethod
    def make_consolidated():
        base = {
            "Frequência": "100.0",
            "Entidade": "Op",
            "Fistel": "10",
            "Serviço": "011",
            "Latitude": "-23.5",
            "Longitude": "-46.6",
            "Código_Município": "3550308",
            "Classe": "FB",
            "Fonte": "MOSAICO-LICENCIAMENTO",
            "Multiplicidade": "-1",
            "Log": "[]",
        }
        rows = [
            # Telecom group of two, the non principal row merged with an aero station
            dict(base, Estação="1", **{COL_GRUPO: "telecom:0", COL_GRUPO_PRINCIPAL: "True"}),
            dict(
                base,
                Estação="2",
                Entidade="Op | Aero",
                Fonte="MOSAICO-LICENCIAMENTO | AISWEB",
                **{COL_GRUPO: "telecom:0", COL_GRUPO_PRINCIPAL: "False"},
            ),
            # Row with null grouping values: aggregate discards it
            dict(base, Estação="3", **{COL_GRUPO: "-1", COL_GRUPO_PRINCIPAL: "False"}),
            # SMP group of two in different places of the same city
            dict(
                base,
                Estação="4",
                Frequência="900.0",
                Serviço="010",
                Latitude="-23.6",
                **{COL_GRUPO: "smp:0", COL_GRUPO_PRINCIPAL: "True"},
            ),
            dict(
                base,
                Estação="5",
                Frequência="900.0",
                Serviço="010",
                Latitude="-23.7",
                **{COL_GRUPO: "smp:0", COL_GRUPO_PRINCIPAL: "False"},
            ),
            # Source which is never grouped
            dict(
                base,
                Estação="6",
                Fonte="STEL",
                Multiplicidade="1",
                **{COL_GRUPO: "-1", COL_GRUPO_PRINCIPAL: "-1"},
            ),
        ]
        return pd.DataFrame(rows, dtype="string").astype("category")

    def test_group_rows_keeps_principals_and_applies_grouped_substitutions(self):
        stats = pd.DataFrame(
            {
                "Multiplicidade": [2, 2],
                "#Estação": ["[1, 2]", "[4, 5]"],
                "Processamento": ["TELECOM: Agrupamento", "SMP: Agrupamento"],
            },
            index=pd.Index(["telecom:0", "smp:0"], name=COL_GRUPO),
        )

        result = Estacoes._group_rows(self.make_consolidated(), stats)
        result = result.set_index("Estação")

        self.assertEqual(sorted(result.index), ["1", "4", "6"])
        self.assertEqual(result.loc["1", "Fonte"], "MOSAICO-LICENCIAMENTO | AISWEB")
        self.assertEqual(result.loc["1", "Entidade"], "Op | Aero")
        self.assertEqual(result.loc["1", "Multiplicidade"], "2")
        self.assertEqual(result.loc["6", "Multiplicidade"], "1")
        telecom_log = json.loads(result.loc["1", "Log"])
        self.assertEqual(
            telecom_log,
            [{"Log": "TELECOM: Agrupamento", "Coluna": "#Estação", "Original": "[1, 2]"}],
        )
        self.assertEqual(result.loc["4", "Latitude"], "-23.554753")
        self.assertEqual(result.loc["4", "Longitude"], "-46.579009")
        smp_log = json.loads(result.loc["4", "Log"])
        self.assertEqual(smp_log[0]["Log"], "SMP: Agrupamento")
        self.assertEqual(smp_log[1]["Original"], "-23.6")
        self.assertEqual(json.loads(result.loc["6", "Log"]), [])

    def test_group_stats_count_downlink_rows_per_group(self):
        smp = Smp(read_cache=True, full=True)
        df = pd.DataFrame(
            {
                "Estação": ["2", "1", "1", "3"],
                "Fonte": [
                    "MOSAICO-LICENCIAMENTO",
                    "MOSAICO-LICENCIAMENTO",
                    "CANALIZACAO-SMP",
                    "MOSAICO-LICENCIAMENTO",
                ],
                COL_GRUPO: ["smp:0", "smp:0", "smp:0", pd.NA],
            },
            dtype="string",
        )

        stats = Estacoes._group_stats(df, smp)

        self.assertEqual(stats.loc["smp:0", "Multiplicidade"], 2)
        self.assertEqual(stats.loc["smp:0", "#Estação"], "[2, 1]")
        self.assertTrue(stats.loc["smp:0", "Processamento"].startswith(smp.agg_log))


class StationFullUpdateTests(unittest.TestCase):
    def test_full_mode_uses_full_mosaico_sources_and_consolidates_once(self):
        stations = Estacoes(read_cache=True, parallel=False, full=True)
        self.assertTrue(stations.sources[0].full and stations.sources[1].full)
        frame = StationGroupRowsTests.make_consolidated().astype("string")
        columns = [*full_columns(stations.columns), "Multiplicidade", COL_GRUPO, COL_GRUPO_PRINCIPAL]
        frame = frame.reindex(columns=columns, fill_value="-1").astype("category")
        consolidate = patch.object(
            stations, "_consolidate", return_value=frame.copy()
        )
        stats = pd.DataFrame(
            {"Multiplicidade": [], "#Estação": [], "Processamento": []},
            index=pd.Index([], name=COL_GRUPO),
        )
        dfs = L([frame.astype("string")] * 2 + [pd.DataFrame()] * 4)

        with (
            consolidate as consolidated,
            patch.object(Estacoes, "_group_stats", return_value=stats),
            tempfile.TemporaryDirectory() as folder,
        ):
            stations.folder = Path(folder)
            stations.update_full(dfs)

        consolidated.assert_called_once()
        self.assertEqual(len(stations.df_full), len(frame))
        self.assertNotIn(COL_GRUPO, stations.df_full.columns)
        self.assertIn("Tecnologia", stations.df_full.columns)
        self.assertEqual(len(stations.df), 3)
        self.assertEqual(stations.df.columns.tolist(), stations.columns)


class StationSaveTests(unittest.TestCase):
    def test_save_persists_full_output_only_when_enabled(self):
        grouped = pd.DataFrame({"Multiplicidade": [2], "Log": [0]})
        full = pd.DataFrame({"Log": ["[]"]}, dtype="string")

        with tempfile.TemporaryDirectory() as folder:
            stations = Estacoes(read_cache=True, full=True)
            stations.folder = Path(folder)
            stations.df = grouped
            stations.df_full = full
            stations.save()

            grouped_saved = pd.read_parquet(Path(folder) / "estacoes.parquet")
            full_saved = pd.read_parquet(Path(folder) / "estacoes_full.parquet")
            self.assertIn("Multiplicidade", grouped_saved.columns)
            self.assertNotIn("Multiplicidade", full_saved.columns)
            self.assertEqual(full_saved["Log"].dtype.name, "category")
            self.assertEqual(json.loads(full_saved.loc[0, "Log"]), [])
            self.assertTrue(
                (Path(folder) / "estacoes_discarded.parquet").is_file()
            )
            self.assertFalse((Path(folder) / "log.parquet").exists())

        with tempfile.TemporaryDirectory() as folder:
            stations = Estacoes(read_cache=True, full=False)
            stations.folder = Path(folder)
            stations.df = grouped
            stations.save()

            self.assertTrue((Path(folder) / "estacoes.parquet").is_file())
            self.assertTrue(
                (Path(folder) / "estacoes_discarded.parquet").is_file()
            )
            self.assertFalse((Path(folder) / "estacoes_full.parquet").exists())


if __name__ == "__main__":
    unittest.main()
