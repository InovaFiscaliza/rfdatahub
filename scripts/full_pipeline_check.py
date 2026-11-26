"""Offline regression helpers for the grouped and full station pipelines."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pandas as pd
import typer
from fastcore.foundation import L

from extracao.constants import AGG_LICENCIAMENTO, IBGE_MUNICIPIOS, IBGE_POLIGONO
from extracao.datasources.aeronautica import Aero
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
        source._save(source.df, output_dir, source.stem)
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


def _run_grouped(output_dir: Path, sample: int) -> dict[str, pd.DataFrame]:
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
    print(
        f"Telecom invariants hold: grouped={len(grouped):,}, "
        f"full={len(full):,}, eligible={full_eligible_rows:,}"
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

        differences = []
        for name in ("telecom", "smp", "estacoes", "log"):
            actual = pd.read_parquet(actual_dir / f"{name}.parquet")
            expected = pd.read_parquet(baseline_dir / f"{name}.parquet")
            try:
                assert_frames_equal(name, actual, expected)
            except AssertionError as error:
                differences.append(str(error))

    if differences:
        typer.echo("\n".join(differences), err=True)
        raise typer.Exit(code=1)
    print(f"Grouped outputs match baseline: {baseline_dir}")


if __name__ == "__main__":
    app()
