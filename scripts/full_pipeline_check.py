"""Offline regression helpers for the grouped and full station pipelines."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import typer
from fastcore.foundation import L

from extracao.constants import IBGE_MUNICIPIOS, IBGE_POLIGONO
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


if __name__ == "__main__":
    app()
