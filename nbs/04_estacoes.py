import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    #| default_exp stations
    # magic command not supported in marimo; please file an issue to add support
    # %load_ext autoreload
    # '%autoreload 2' command supported automatically in marimo

    import sys
    from pathlib import Path

    return Path, sys


@app.cell
def _(Path, sys):
    # Insert in Path Project Directory
    sys.path.insert(0, str(Path().cwd().parent))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Anatel

    > Este módulo consolida as bases da Anatel e realiza pós-processamento dos dados obtidos.
    """)
    return


@app.cell
def _():
    # | export
    import pandas as pd
    from dotenv import find_dotenv, load_dotenv
    from fastcore.foundation import L
    from fastcore.parallel import parallel

    from extracao.constants import COLS_SRD
    from extracao.location import Geography

    from extracao.datasources.aeronautica import Aero
    from extracao.datasources.base import Base
    from extracao.datasources.mosaico import MONGO_URI
    from extracao.datasources.sitarweb import SQLSERVER_PARAMS, Radcom, Stel
    from extracao.datasources.smp import Smp
    from extracao.datasources.srd import SRD
    from extracao.datasources.telecom import Telecom
    from extracao.format import LIMIT_FREQ, merge_on_frequency, merge_dicts

    return (
        Aero,
        Base,
        COLS_SRD,
        Geography,
        L,
        LIMIT_FREQ,
        MONGO_URI,
        Radcom,
        SQLSERVER_PARAMS,
        SRD,
        Smp,
        Stel,
        Telecom,
        find_dotenv,
        load_dotenv,
        merge_dicts,
        merge_on_frequency,
        parallel,
        pd,
    )


@app.cell
def _(find_dotenv, load_dotenv, pd):
    # | export
    load_dotenv(find_dotenv(), override=True)
    pd.options.mode.copy_on_write = True
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Base Consolidada ANATEL
    """)
    return


@app.cell
def _(
    Aero,
    Base,
    COLS_SRD,
    Geography,
    L,
    LIMIT_FREQ,
    MONGO_URI,
    Radcom,
    SQLSERVER_PARAMS,
    SRD,
    Smp,
    Stel,
    Telecom,
    merge_dicts,
    merge_on_frequency,
    parallel,
    pd,
):
    #| export
    class Estacoes(Base):
        """Helper Class to aggregate and process the data from different sources"""

        def __init__(
            self,
            sql_params: dict = SQLSERVER_PARAMS,
            mongo_uri: str = MONGO_URI,
            limit: int = 0,
            parallel: bool = True,
            read_cache: bool = False,
            reprocess_sources: bool = False,
        ):
            self.sql_params = sql_params
            self.mongo_uri = mongo_uri
            self.limit = limit
            self.parallel = parallel
            self.read_cache = read_cache
            self.reprocess_sources = reprocess_sources
            self.init_data_sources()

        @property
        def columns(self):
            return COLS_SRD

        @property
        def stem(self):
            return "estacoes"

        @staticmethod
        def _update_source(class_instance):
            """Helper functions to update and save the individual data sources"""
            try:
                class_instance.update()
                class_instance.save()
            except Exception as e:
                print(
                    f"Erro ao atualizar a classe {class_instance.__class__.__name__}: {e}"
                )
            return class_instance

        def init_data_sources(self):
            """Initializes the individual classes and saves them in a property list"""
            self.sources = L(
                [
                    Telecom(self.mongo_uri, self.limit, self.read_cache),
                    Smp(self.mongo_uri, self.limit, self.read_cache),
                    SRD(self.mongo_uri, self.limit, self.read_cache),
                    Stel(self.sql_params, self.read_cache),
                    Radcom(self.sql_params, self.read_cache),
                    Aero(read_cache=self.read_cache),
                ]
            )

        def extraction(self) -> L:
            if not self.reprocess_sources and self.read_cache:
                return self.sources.attrgot("df")
            if self.parallel:
                self.sources = parallel(
                    Estacoes._update_source,
                    self.sources,
                    n_workers=len(self.sources),
                    progress=False,
                )
            else:
                self.sources = self.sources.map(Estacoes._update_source)
            return self.sources.attrgot("df")

        def update(self):
            dfs = self.extraction()
            self.df = self._format(dfs)

        @staticmethod
        def _simplify_sources(df):
            df["Fonte"] = df["Fonte"].str.replace(
                "ICAO-CANALIZACAO-VOR/ILS/DME | AISWEB-CANALIZACAO-VOR/ILS/DME",
                "CANALIZACAO-VOR/ILS/DME",
            )
            df["Fonte"] = df["Fonte"].str.replace(
                r"(ICAO-)?(AISWEB-)?CANALIZACAO-VOR/ILS/DME",
                "CANALIZACAO-VOR/ILS/DME",
                regex=True,
            )

            return df

        @staticmethod
        def _remove_invalid_frequencies(df):
            df["Frequência"] = df["Frequência"].astype("float")
            df.sort_values(
                ["Frequência", "Latitude", "Longitude"], ignore_index=True, inplace=True
            )
            row_filter = df["Frequência"] > LIMIT_FREQ
            Base.register_log(
                df,
                f"Frequência removida. Valor superior a {LIMIT_FREQ}MHz.",
                "Frequência",
                row_filter,
            )
            return df[~row_filter].reset_index(drop=True)

        def process_log(self, df):
            """Process the log and save in a different dataframe"""
            log = df["Log"].progress_apply(merge_dicts)
            labels, uniques = pd.factorize(log, sort=False)
            log = pd.DataFrame({"Log": uniques}, dtype="string")
            log.to_parquet(f"{self.folder}/log.parquet", compression="gzip", index=False)
            df["Log"] = pd.to_numeric(labels, downcast="integer")
            return df

        def _format(self, dfs: L) -> pd.DataFrame:
            aero = dfs.pop()
            anatel = pd.concat(dfs, ignore_index=True).astype(
                "string"
            )
            df = merge_on_frequency(anatel, aero)
            df = Geography(df).validate()
            df = Estacoes._simplify_sources(df)
            df = Estacoes._remove_invalid_frequencies(df)
            df = df.astype("string")
            for column in df.columns:
                df[column] = df[column].str.replace(r"^-1\.0$|^\s$", "-1", regex=True)
            df = df.fillna("-1").astype("category")
            df = self.process_log(df)
            return df.loc[:, self.columns]

    return (Estacoes,)


@app.cell
def _(Estacoes):
    data = Estacoes(read_cache=True, reprocess_sources=True)
    return (data,)


@app.cell
def _(data):
    dfs = data.extraction()
    return (dfs,)


@app.cell
def _(dfs, merge_on_frequency, pd):
    aero = dfs.pop()
    anatel = pd.concat(dfs, ignore_index=True).astype('string')
    df = merge_on_frequency(anatel, aero)
    return (df,)


@app.cell
def _(df):
    df.info()
    return


@app.cell
def _(df):
    df.isna().sum()
    return


@app.cell
def _(Geography, df):
    g = Geography(df)
    return (g,)


@app.cell
def _(g):
    g.df.isna().sum()
    return


@app.cell
def _(g):
    c = g.df['Código_Município'].isna()
    g.df[c]
    return


@app.cell
def _(g):
    g.merge_df_with_ibge()
    return


@app.cell
def _(g):
    g.log.keys()
    return


@app.cell
def _(g):
    g.df[g.log['empty_code']]
    return


@app.cell
def _(g):
    g.df[g.log["empty_coords"]]
    return


@app.cell
def _(g):
    g.df[g.log["invalid_code"]]
    return


@app.cell
def _(g):
    g.df.isna().sum()
    return


@app.cell
def _(g):
    g.normalize_location_names()
    return


@app.cell
def _(g):
    g.fill_missing_coords()
    return


@app.cell
def _(g):
    g.df.isna().sum()
    return


@app.cell
def _(g):
    g.intersect_coordinates_on_poligon()
    return


@app.cell
def _(g):
    g.fill_missing_city_info()
    return


@app.cell
def _(g):
    g.log['filled_city_info'].sum()
    return


@app.cell
def _(g):
    g.df.isna().sum()
    return


@app.cell
def _(g):
    g.substitute_divergent_coordinates()
    return


@app.cell
def _(g):
    a = g.df['Código_Município'] != g.df['CD_MUN']
    g.df[a]
    return


@app.cell
def _(g):
    g.input_info_from_coords()
    return


@app.cell
def _(g):
    g.df.isna().sum()
    return


if __name__ == "__main__":
    app.run()
