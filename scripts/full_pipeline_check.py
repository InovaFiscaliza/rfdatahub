"""Offline regression helpers for the grouped and full station pipelines."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import pandas as pd
import typer
from fastcore.foundation import L

from extracao.constants import (
    AGG_LICENCIAMENTO,
    AGG_SMP,
    COLS_SRD,
    IBGE_MUNICIPIOS,
    IBGE_POLIGONO,
)
from extracao.datasources.aeronautica import Aero
from extracao.datasources.mosaico import full_columns
from extracao.datasources.sitarweb import Radcom, Stel
from extracao.datasources.smp import Smp
from extracao.datasources.srd import SRD
from extracao.datasources.telecom import Telecom
from extracao.stations import Estacoes

app = typer.Typer()
ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "extracao" / "datasources" / "arquivos" / "saida"
DEFAULT_OUTPUT_DIR = ROOT / "extracao" / "datasources" / "arquivos" / "baseline"


@app.callback()
def main() -> None:
    """Offline regression helpers for grouped and full station pipelines."""


def _sample_raw_cache(source, sample: int) -> None:
    if sample <= 0 or source.stem not in {"telecom", "smp"}:
        return

    read = source._read

    def read_sampled(stem: str, backend: str = "pyarrow") -> pd.DataFrame:
        df = read(stem, backend)
        if stem == f"{source.stem}_raw":
            return df.head(sample)
        return df

    source._read = read_sampled


def _process_source(source, output_dir: Path, sample: int) -> pd.DataFrame:
    source.folder = CACHE_DIR
    _sample_raw_cache(source, sample)
    source.update()
    if source.stem in {"telecom", "smp"}:
        source._save(source.df, output_dir, source.output_stem)
    return source.df


def _check_offline_inputs() -> None:
    shapefile = Path(IBGE_POLIGONO)
    required = [
        *(
            CACHE_DIR / f"{stem}_raw.parquet"
            for stem in ("telecom", "smp", "srd", "stel", "radcom")
        ),
        CACHE_DIR / "aero.parquet",
        Path(IBGE_MUNICIPIOS),
        *(
            shapefile.with_suffix(suffix)
            for suffix in (".shp", ".cpg", ".dbf", ".prj", ".shx")
        ),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing offline baseline inputs: {', '.join(missing)}")


def _load_offline_sources(output_dir: Path, sample: int) -> L:
    _check_offline_inputs()
    output_dir.mkdir(parents=True, exist_ok=True)

    telecom = _process_source(Telecom(read_cache=True), output_dir, sample)
    smp = _process_source(Smp(read_cache=True), output_dir, sample)
    srd = _process_source(SRD(read_cache=True), output_dir, sample)
    stel = _process_source(Stel(read_cache=True), output_dir, sample)
    radcom = _process_source(Radcom(read_cache=True), output_dir, sample)

    aero_source = Aero(read_cache=True)
    aero_source.folder = CACHE_DIR
    aero = aero_source.df

    return L([telecom, smp, srd, stel, radcom, aero])


def _run_grouped(output_dir: Path, sample: int) -> dict[str, pd.DataFrame]:
    telecom, smp, srd, stel, radcom, aero = _load_offline_sources(
        output_dir, sample
    )
    stations = Estacoes(read_cache=True, parallel=False)
    stations.folder = output_dir
    stations_df = stations._format(L([telecom, smp, srd, stel, radcom, aero]))
    stations._save(stations_df, output_dir, "estacoes")

    return {
        "telecom": telecom,
        "smp": smp,
        "estacoes": stations_df,
        "log": pd.read_parquet(output_dir / "log.parquet"),
    }


def _run_full(output_dir: Path, sample: int) -> dict[str, pd.DataFrame]:
    _check_offline_inputs()
    output_dir.mkdir(parents=True, exist_ok=True)
    telecom = _process_source(Telecom(read_cache=True, full=True), output_dir, sample)
    smp = _process_source(Smp(read_cache=True, full=True), output_dir, sample)
    others = [
        _process_source(source, output_dir, sample)
        for source in (SRD(read_cache=True), Stel(read_cache=True), Radcom(read_cache=True))
    ]
    aero_source = Aero(read_cache=True)
    aero_source.folder = CACHE_DIR

    stations = Estacoes(read_cache=True, parallel=False, full=True)
    stations.folder = output_dir
    for source, df in zip(stations.sources, [telecom, smp, *others, aero_source.df]):
        source.folder = output_dir
        source.df = df
    stations.update()
    stations._save(stations.df, output_dir, "estacoes")
    stations._save(stations.df_full, output_dir, "estacoes_full")

    return {
        "telecom_full": pd.read_parquet(output_dir / "telecom_full.parquet"),
        "smp_full": pd.read_parquet(output_dir / "smp_full.parquet"),
        "estacoes": pd.read_parquet(output_dir / "estacoes.parquet"),
        "log": pd.read_parquet(output_dir / "log.parquet"),
        "estacoes_full": pd.read_parquet(output_dir / "estacoes_full.parquet"),
    }


@app.command("full-invariants")
def full_invariants(
    sample: int = typer.Option(0, min=0, help="Process only the first N telecom/SMP raw rows."),
) -> None:
    """Check that Telecom full mode preserves all eligible ungrouped rows."""
    grouped_source = Telecom(read_cache=True)
    grouped_source.folder = CACHE_DIR
    _sample_raw_cache(grouped_source, sample)
    grouped_source.update()

    full_source = Telecom(read_cache=True, full=True)
    full_source.folder = CACHE_DIR
    _sample_raw_cache(full_source, sample)
    full_source.update()

    grouped = grouped_source.df
    full = full_source.df
    grouped_multiplicity = pd.to_numeric(
        grouped["Multiplicidade"], errors="raise"
    ).sum()
    full_eligible_rows = full[AGG_LICENCIAMENTO].notna().all(axis=1).sum()
    if len(full) < len(grouped):
        raise AssertionError(
            f"Telecom full rows ({len(full)}) are fewer than grouped rows "
            f"({len(grouped)})."
        )
    if grouped_multiplicity != full_eligible_rows:
        raise AssertionError(
            f"Grouped Telecom multiplicity sum ({grouped_multiplicity}) does not "
            f"equal eligible full rows ({full_eligible_rows})."
        )

    grouped_smp_source = Smp(read_cache=True)
    grouped_smp_source.folder = CACHE_DIR
    _sample_raw_cache(grouped_smp_source, sample)
    grouped_smp_source.update()

    full_smp_source = Smp(read_cache=True, full=True)
    full_smp_source.folder = CACHE_DIR
    _sample_raw_cache(full_smp_source, sample)
    full_smp_source.update()

    grouped_smp = grouped_smp_source.df
    grouped_downlink = grouped_smp["Fonte"] == "MOSAICO-LICENCIAMENTO"
    grouped_smp_multiplicity = pd.to_numeric(
        grouped_smp.loc[grouped_downlink, "Multiplicidade"], errors="raise"
    ).sum()
    full_smp = full_smp_source.df
    full_downlink = full_smp["Fonte"] == "MOSAICO-LICENCIAMENTO"
    missing_smp_agg_cols = set(AGG_SMP) - set(full_smp.columns)
    if missing_smp_agg_cols:
        raise AssertionError(
            f"Full SMP output is missing aggregation columns: "
            f"{sorted(missing_smp_agg_cols)}"
        )
    full_smp_eligible = full_smp.loc[full_downlink, AGG_SMP].notna().all(axis=1).sum()
    if grouped_smp_multiplicity != full_smp_eligible:
        raise AssertionError(
            f"Grouped SMP downlink multiplicity sum ({grouped_smp_multiplicity}) "
            f"does not equal eligible full downlink rows ({full_smp_eligible})."
        )
    print(
        f"Telecom invariants hold: grouped={len(grouped):,}, "
        f"full={len(full):,}, eligible={full_eligible_rows:,}"
    )
    print(
        f"SMP downlink invariant holds: grouped multiplicity="
        f"{grouped_smp_multiplicity:,}, full eligible={full_smp_eligible:,}"
    )


def assert_frames_equal(
    name: str, actual: pd.DataFrame, expected: pd.DataFrame
) -> None:
    try:
        pd.testing.assert_frame_equal(actual, expected, check_dtype=True)
    except AssertionError as error:
        actual_columns = actual.columns.tolist()
        expected_columns = expected.columns.tolist()
        differing_rows = []
        for index in range(max(len(actual), len(expected))):
            has_actual = index < len(actual)
            has_expected = index < len(expected)
            actual_row = actual.iloc[index] if has_actual else "<missing>"
            expected_row = expected.iloc[index] if has_expected else "<missing>"
            rows_differ = (
                not has_actual
                or not has_expected
                or not actual_row.equals(expected_row)
            )
            if rows_differ:
                differing_rows.append(
                    {
                        "index": index,
                        "actual": (
                            actual_row.to_dict() if has_actual else actual_row
                        ),
                        "baseline": (
                            expected_row.to_dict() if has_expected else expected_row
                        ),
                    }
                )
                if len(differing_rows) == 3:
                    break

        raise AssertionError(
            f"{name} differs from baseline: "
            f"actual shape={actual.shape}, baseline shape={expected.shape}; "
            f"actual columns={actual_columns}, baseline columns={expected_columns}; "
            f"first differing rows={differing_rows or 'none (likely dtype/index mismatch)'}; "
            f"pandas diff: {error}"
        ) from error


def _compare_grouped_outputs(actual_dir: Path, baseline_dir: Path) -> None:
    differences = []
    for name in ("telecom", "smp", "estacoes", "log"):
        actual = pd.read_parquet(actual_dir / f"{name}.parquet")
        expected = pd.read_parquet(baseline_dir / f"{name}.parquet")
        try:
            assert_frames_equal(name, actual, expected)
        except AssertionError as error:
            differences.append(str(error))

    if differences:
        raise AssertionError("\n".join(differences))


def _compare_grouped_rows(actual_dir: Path, baseline_dir: Path) -> None:
    """Compare grouped stations ignoring row order and `Log`.
    The full pipeline derives them from validated rows, so some logs legitimately differ.
    """

    def read(folder: Path) -> pd.DataFrame:
        df = pd.read_parquet(folder / "estacoes.parquet").drop(columns="Log")
        df = df.astype("string")
        return df.sort_values(df.columns.tolist(), ignore_index=True)

    assert_frames_equal("estacoes", read(actual_dir), read(baseline_dir))


@app.command()
def baseline(
    sample: int = typer.Option(0, min=0, help="Process only the first N telecom/SMP raw rows."),
    out: Path = typer.Option(
        DEFAULT_OUTPUT_DIR, help="Directory for baseline parquet files."
    ),
) -> None:
    """Generate the current grouped pipeline baseline entirely from local caches."""
    output_dir = out / f"sample_{sample}" if sample else out
    started = time.perf_counter()
    results = _run_grouped(output_dir, sample)
    elapsed = time.perf_counter() - started

    print(f"Baseline written to {output_dir}")
    for name, df in results.items():
        print(f"{name}: {len(df):,} rows")
    print(f"Elapsed: {elapsed:.1f} seconds")


@app.command()
def compare(
    sample: int = typer.Option(0, min=0, help="Process only the first N telecom/SMP raw rows."),
    out: Path = typer.Option(
        DEFAULT_OUTPUT_DIR, help="Directory containing the grouped baseline."
    ),
) -> None:
    """Compare a fresh offline grouped run against the saved baseline."""
    baseline_dir = out / f"sample_{sample}" if sample else out
    with tempfile.TemporaryDirectory(prefix="rfdatahub-compare-") as temp_dir:
        actual_dir = Path(temp_dir)
        _run_grouped(actual_dir, sample)
        try:
            _compare_grouped_outputs(actual_dir, baseline_dir)
        except AssertionError as error:
            typer.echo(str(error), err=True)
            raise typer.Exit(code=1) from error
    print(f"Grouped outputs match baseline: {baseline_dir}")


@app.command()
def full(
    sample: int = typer.Option(0, min=0, help="Process only the first N telecom/SMP raw rows."),
    out: Path = typer.Option(
        DEFAULT_OUTPUT_DIR / "full", help="Directory for grouped and full outputs."
    ),
) -> None:
    """Run the full station pipeline offline and check its grouped and full outputs."""
    output_dir = out / f"sample_{sample}" if sample else out
    baseline_dir = DEFAULT_OUTPUT_DIR / f"sample_{sample}" if sample else DEFAULT_OUTPUT_DIR
    results = _run_full(output_dir, sample)

    try:
        _compare_grouped_rows(output_dir, baseline_dir)
    except AssertionError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error

    full = results["estacoes_full"]
    expected_columns = full_columns(COLS_SRD)
    if full.columns.tolist() != expected_columns:
        raise AssertionError(
            f"Full station columns differ: {full.columns.tolist()} != {expected_columns}"
        )
    invalid_logs = [
        index
        for index, value in enumerate(full["Log"])
        if not isinstance(json.loads(value), list)
    ]
    if invalid_logs:
        raise AssertionError(f"Full station Log values are not JSON arrays: {invalid_logs[:5]}")
    if len(full) < len(results["estacoes"]):
        raise AssertionError(
            f"Full station rows ({len(full)}) are fewer than grouped rows "
            f"({len(results['estacoes'])})."
        )

    print(f"Grouped station rows match baseline: {baseline_dir}")
    print(f"Full station output written to {output_dir / 'estacoes_full.parquet'}")


if __name__ == "__main__":
    app()
