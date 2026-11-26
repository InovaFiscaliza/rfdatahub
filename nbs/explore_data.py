# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "altair>=6.0.0",
#     "folium>=1.0.0",
#     "geopandas>=1.0.0",
#     "marimo>=0.26.0",
#     "numpy>=2.0.0",
#     "pandas>=2.2.0",
#     "pyarrow>=18.0.0",
#     "pyogrio>=0.10.0",
#     "shapely>=2.0.0",
# ]
#
# [tool.marimo.runtime]
# output_max_bytes = 50_000_000
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import html
    import json
    import os
    from datetime import datetime
    from pathlib import Path

    import altair as alt
    import folium
    import folium.plugins as plugins
    import geopandas as gpd
    import marimo as mo
    import numpy as np
    import pandas as pd
    import shapely


@app.cell(hide_code=True)
def _(DATA_DIR, estacoes_mtime_str, estacoes_size_mb, log_size_mb):
    mo.md(f"""
    # 📡 RFDataHub — Análise Exploratória, Rastreabilidade & Auditoria de Qualidade
    **Diretório ativo:** `{DATA_DIR}` | **Última atualização (`estacoes.parquet`):** `{estacoes_mtime_str}` ({estacoes_size_mb:.1f} MB) | **`log.parquet`:** {log_size_mb:.2f} MB

    Este aplicativo interativo atualiza automaticamente todos os indicadores, tabelas, gráficos e mapas a cada nova execução do pipeline:
    1. **Visão Geral & Completude do Esquema** — volume consolidado vs. emissores físicos (`Multiplicidade`), preenchimento por coluna e sentinelas `"-1"`.
    2. **Espectro, Serviços, Fontes & Operadoras** — distribuição por faixas de frequência, serviços Anatel/Aero, UFs e principais entidades licenciadas.
    3. **Explorador Estruturado de `log.parquet`** — desaninhamento dos templates JSON fatorados, fluxos de processamento (`Pipeline`), deslocamento ERB → município no SMP e inspeção antes/depois.
    4. **Investigação Especial — Coordenadas Divergentes (`IBGE_COORD_DIVERGENTE`)** — diagnóstico geoespacial das estações cujas coordenadas caem em outro polígono municipal do IBGE (apenas registrado no `Log`; coordenadas preservadas), com amplificação por múltiplos canais, efeito de fronteira/topo de serra vs. erros cadastrais e mapa coordenada → centroide declarado.
    5. **Auditoria de Qualidade Residual & Mapa Interativo Folium** — anomalias remanescentes pós-processamento com inspeção individual de estações.
    """)
    return


@app.cell
def _():
    OFFICIAL_UFS = frozenset(
        {
            "AC",
            "AL",
            "AM",
            "AP",
            "BA",
            "CE",
            "DF",
            "ES",
            "GO",
            "MA",
            "MG",
            "MS",
            "MT",
            "PA",
            "PB",
            "PE",
            "PI",
            "PR",
            "RJ",
            "RN",
            "RO",
            "RR",
            "RS",
            "SC",
            "SE",
            "SP",
            "TO",
        }
    )

    SERVICO_NOMES = {
        "019": "019 - SLP (Limitado Privado)",
        "010": "010 - SMP (Móvel Pessoal 2G-5G)",
        "046": "046 - SCM (Comunicação Multimídia)",
        "011": "011 - SLE (Limitado Especializado)",
        "053": "053 - STFC (Telefonia Fixa)",
        "801": "801 - Retransmissão de TV (RTV)",
        "023": "023 - SLMP (Limitado Móvel Privativo)",
        "800": "800 - TV (Sons e Imagens)",
        "175": "175 - Radioenlaces Associados",
        "231": "231 - RadCom (Comunitária)",
        "125": "125 - Monitoramento / Segurança",
        "230": "230 - Rádio FM",
        "728": "728 - Serviço Auxiliar de Radiodifusão",
        "-1": "-1 - Sem Código (Aero/ICAO)",
    }

    # Sentinel used by every "(Todas)" dropdown filter
    ALL_OPTION = "(Todas)"

    SPECTRUM_BAND_BINS = [0, 0.3, 3, 30, 300, 1000, 3000, 6000, 30000, 100000]
    SPECTRUM_BAND_LABELS = [
        "1. LF (<300 kHz)",
        "2. MF/OM (0.3–3 MHz)",
        "3. HF (3–30 MHz)",
        "4. VHF (30–300 MHz)",
        "5. UHF Baixo (300–1000 MHz)",
        "6. UHF/L/S (1–3 GHz)",
        "7. Banda C/5G (3–6 GHz)",
        "8. SHF Micro-ondas (6–30 GHz)",
        "9. EHF mmWave (30–85 GHz)",
    ]

    # Engineering columns that should be numeric after normalization
    ENG_NUMERIC_COLS = [
        "Potência_Transmissor(W)",
        "Altura_Antena(m)",
        "Cota_Base_Torre(m)",
        "Ganho_Antena(dBd)",
        "Ângulo_Elevação_Antena",
        "Azimute_Antena",
        "Atenuação_Linha(db/100m)",
        "Perdas_Acessórias_Linha(db)",
        "Comprimento_Linha(m)",
    ]

    # Section 4 regimes: code -> (color, description). Codes are produced by classify_divergent_regime().
    DIVERGENT_REGIME_META = {
        "1_Micro_Fronteira_100m": (
            "#2b8cbe",
            "1. Micro-fronteira / Datum (≤ 100 m da divisa)",
        ),
        "2_Topo_Serra_Borda_1km": (
            "#4eb3d3",
            "2. Borda / Crista de Serra (100 m – 1 km da divisa)",
        ),
        "3_Limitrofe_1_10km": (
            "#fdae61",
            "3. Município Limítrofe Próximo (1 – 10 km da divisa)",
        ),
        "4_Limitrofe_Maior_10km": (
            "#f46d43",
            "4. Município Limítrofe Distante (> 10 km da divisa)",
        ),
        "5_Nao_Vizinho_Mesma_UF": ("#d73027", "5. Município Não-Vizinho na Mesma UF"),
        "6_Outra_UF_Nao_Vizinho": (
            "#762a83",
            "6. Outra UF Não-Limítrofe (Erro Cadastral / Digitação)",
        ),
    }

    # Section 5 residual issues: code -> (color, description). Codes are produced by classify_primary_issue().
    ISSUE_META = {
        "01_Null_Island_ou_Polar_Invalido": (
            "#d62728",
            "Crítico: Null Island (0,0) ou Latitude Polar (±89°)",
        ),
        "02_Lat_Lon_Invertidas": (
            "#e31a1c",
            "Crítico: Latitude ↔ Longitude Trocadas (no Oceano)",
        ),
        "03_Sinal_Invertido_Hemisferio": (
            "#ff7f00",
            "Crítico: Sinal Invertido (Hemisfério Norte / Leste +Lon)",
        ),
        "04_Fora_Fronteira_Terrestre": (
            "#984ea3",
            "Alerta: Coordenada em País Vizinho / Fronteira",
        ),
        "05_Poligono_UF_Divergente": (
            "#e7298a",
            "Erro Geo: Ponto cai em OUTRA UF pelo Polígono IBGE",
        ),
        "06_Poligono_Municipio_Divergente": (
            "#f781bf",
            "Erro Geo: Ponto cai em OUTRO Município IBGE (mesma UF)",
        ),
        "07_Offshore_Maritimo_Ilhas": (
            "#1f78b4",
            "Contexto: Offshore / Pré-Sal / Costeiro (fora da malha terrestre)",
        ),
        "08_Texto_Sujo_Coluna_Numerica": (
            "#b15928",
            "Erro Tipo: Strings não normalizadas em Engenharia (vírgula/unidade)",
        ),
        "09_Outlier_Fisico_Antena_Torre": (
            "#a65628",
            "Alerta Físico: Azimute >360°, Elevação >90° ou Cota >3000m",
        ),
        "10_UF_Com_Espaco_ou_Divergente": (
            "#fdbf6f",
            "Erro String: UF com espaço ou divergente do IBGE",
        ),
        "11_Sem_Codigo_Municipio_ou_UF": (
            "#6a3d9a",
            "Incompleto: Código_Município ou UF == '-1'",
        ),
        "00_Sem_Coordenadas_-1": (
            "#7f7f7f",
            "Incompleto: Sem Latitude/Longitude ('-1', não plotável)",
        ),
        "00_OK_Consistente": (
            "#2ca02c",
            "OK: Coordenada consistente com Polígono IBGE",
        ),
    }

    DEFAULT_MAP_ISSUES = [
        "01_Null_Island_ou_Polar_Invalido",
        "02_Lat_Lon_Invertidas",
        "03_Sinal_Invertido_Hemisferio",
        "04_Fora_Fronteira_Terrestre",
        "05_Poligono_UF_Divergente",
        "06_Poligono_Municipio_Divergente",
        "08_Texto_Sujo_Coluna_Numerica",
        "09_Outlier_Fisico_Antena_Torre",
        "10_UF_Com_Espaco_ou_Divergente",
    ]
    return (
        ALL_OPTION,
        DEFAULT_MAP_ISSUES,
        DIVERGENT_REGIME_META,
        ENG_NUMERIC_COLS,
        ISSUE_META,
        OFFICIAL_UFS,
        SERVICO_NOMES,
        SPECTRUM_BAND_BINS,
        SPECTRUM_BAND_LABELS,
    )


@app.function
def cat_to_float(s: pd.Series) -> np.ndarray:
    """Fast vectorized conversion of categorical string Series to float64 (-1 -> NaN)."""
    if hasattr(s, "cat"):
        cats = s.cat.categories
        conv = pd.to_numeric(cats, errors="coerce").to_numpy(dtype="float64", copy=True)
        conv[cats == "-1"] = np.nan
        codes = s.cat.codes.to_numpy()
        out = np.full(len(s), np.nan, dtype="float64")
        valid = codes >= 0
        out[valid] = conv[codes[valid]]
        return out
    s_num = pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64", copy=True)
    s_num[s.astype("string").to_numpy() == "-1"] = np.nan
    return s_num


@app.function
def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vectorized great-circle distance in km (NaN inputs propagate to NaN)."""
    la1, lo1, la2, lo2 = (
        np.radians(np.asarray(v, dtype="float64")) for v in (lat1, lon1, lat2, lon2)
    )
    a = (
        np.sin((la2 - la1) / 2.0) ** 2
        + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2.0) ** 2
    )
    return 2.0 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


@app.function
def valid_wgs84_mask(lat, lon) -> np.ndarray:
    """True where (lat, lon) is a finite, plottable WGS84 coordinate."""
    lat = np.asarray(lat, dtype="float64")
    lon = np.asarray(lon, dtype="float64")
    return (
        (~np.isnan(lat))
        & (~np.isnan(lon))
        & (np.abs(lat) <= 90.0)
        & (np.abs(lon) <= 180.0)
    )


@app.function
def top_value(s: pd.Series) -> str:
    """Most frequent value of a Series, as string."""
    return str(s.astype(str).value_counts().index[0])


@app.function
def summarize_by(
    frame: pd.DataFrame,
    by: str,
    *,
    sort: bool = True,
    observed: bool = True,
    **extra_aggs,
) -> pd.DataFrame:
    """Group `frame` (which must contain an `Emissores` column) into Registros / Emissores_Fisicos (+ extra aggs)."""
    out = frame.groupby(by, observed=observed).agg(
        Registros=("Emissores", "size"),
        Emissores_Fisicos=("Emissores", "sum"),
        **extra_aggs,
    )
    if sort:
        out = out.sort_values("Registros", ascending=False)
    return out.reset_index()


@app.function
def selection_mask(values: pd.Series, selected: str, all_option: str) -> np.ndarray:
    """Boolean mask for a dropdown filter; `all_option` disables the filter."""
    if selected == all_option:
        return np.ones(len(values), dtype=bool)
    return (values.astype(str) == selected).to_numpy()


@app.function
def text_search_mask(frame: pd.DataFrame, columns: list[str], query: str) -> np.ndarray:
    """Case-insensitive substring match of `query` over any of `columns` (empty query matches all)."""
    q = query.strip().lower()
    if not q:
        return np.ones(len(frame), dtype=bool)
    mask = np.zeros(len(frame), dtype=bool)
    for col in columns:
        mask |= (
            frame[col].astype(str).str.lower().str.contains(q, regex=False).to_numpy()
        )
    return mask


@app.function
def kpi_row(items: list[tuple[str, str, str]]):
    """Row of equally sized `mo.stat` cards from (label, value, caption) tuples."""
    return mo.hstack(
        [
            mo.stat(label=label, value=value, caption=caption)
            for label, value, caption in items
        ],
        widths="equal",
    )


@app.function
def meta_color_scale(meta: dict[str, tuple[str, str]]) -> alt.Scale:
    """Altair color scale from a `{code: (color, description)}` metadata dict."""
    return alt.Scale(domain=list(meta.keys()), range=[v[0] for v in meta.values()])


@app.function
def format_distance(
    km: float, *, meters_below_1km: bool = True, decimals: int = 2
) -> str:
    """Human-readable distance ('N/A' for NaN, meters below 1 km when enabled)."""
    if km is None or pd.isna(km):
        return "N/A"
    if meters_below_1km and km < 1.0:
        return f"{km * 1000.0:,.0f} m"
    return f"{km:,.{decimals}f} km"


@app.function
def base_folium_map(
    location: list[float], zoom_start: int, *, light_basemap: bool = False
) -> folium.Map:
    """OpenStreetMap-based Folium map, optionally with an extra CARTO light basemap layer."""
    fmap = folium.Map(
        location=location,
        zoom_start=zoom_start,
        tiles="OpenStreetMap",
        control_scale=True,
    )
    if light_basemap:
        folium.TileLayer(
            tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
            attr="&copy; OpenStreetMap contributors &copy; CARTO",
            name="Mapa Claro (CARTO)",
            subdomains="abcd",
        ).add_to(fmap)
    return fmap


@app.function
def folium_iframe(fmap: folium.Map, height: str):
    """Embed a Folium map as a self-contained marimo iframe."""
    return mo.iframe(fmap.get_root().render(), width="100%", height=height)


@app.function
def resolve_arq_dir() -> Path:
    """Locate the RFDataHub `arquivos` directory (override with env var RFDATAHUB_ARQ_DIR)."""
    env_dir = os.environ.get("RFDATAHUB_ARQ_DIR")
    if env_dir and Path(env_dir).exists():
        return Path(env_dir)
    nb_dir = mo.notebook_dir() or Path.cwd()
    fallback = Path("/code/rfdatahub/extracao/datasources/arquivos")
    for candidate in (
        nb_dir.parent / "extracao" / "datasources" / "arquivos",
        Path.cwd() / "extracao" / "datasources" / "arquivos",
        Path.cwd() / "arquivos",
        fallback,
    ):
        if (candidate / "saida" / "estacoes.parquet").exists():
            return candidate
    return fallback


@app.function
def classify_log_step(proc: str) -> tuple[str, bool]:
    """Return (short_tag, is_data_quality_event) for a log message (JSON key "Log")."""
    if proc.startswith(("MOSAICO: Registros agrupados", "MOSAICO: Agrupamento")):
        return ("MOSAICO_AGRUPADO", False)
    if proc.startswith(
        (
            "MOSAICO-TELECOM: Registros Agrupados",
            "MOSAICO-TELECOM: Agrupamento",
            "Registros Agrupados",
        )
    ):
        return ("TELECOM_AGRUPADO", False)
    if "Colunas criadas à partir da Canalização" in proc:
        return ("SMP_CANALIZACAO", False)
    if "Criado registro adicional com Frequência de Recepção" in proc:
        return ("SMP_RX_ESPELHADO", False)
    if "Latitude do Município inserida" in proc:
        return ("SMP_LAT_MUNICIPIO", False)
    if "Longitude do Município inserida" in proc:
        return ("SMP_LON_MUNICIPIO", False)
    if proc.startswith("RADCOM:"):
        return ("RADCOM_FASE_SITUACAO", False)
    if proc.startswith(
        (
            "Código do município retirado do polígono",
            "IBGE: Dado derivado das coordenadas",
        )
    ):
        return ("IBGE_POLIGONO_RECUPERADO", True)
    if proc.startswith("Coordenadas e Código do Município nulos"):
        return ("MUN_E_COORD_NULOS", True)
    if proc.startswith("Código do Município nulo"):
        return ("MUN_CODIGO_NULO", True)
    if proc.startswith(("Coordenadas divergentes", "IBGE: Município distinto")):
        return ("IBGE_COORD_DIVERGENTE", True)
    if proc.startswith("Nomes de Municípios e UFs normalizados"):
        return ("IBGE_NOME_NORMALIZADO", True)
    if proc.startswith(
        (
            "Valor Nulo de Coordenadas",
            "IBGE: Coordenadas Nulas -> Coordenadas do Município",
        )
    ):
        return ("COORD_NULA_PREENCHIDA_IBGE", True)
    if proc.startswith("Coordenadas nulas"):
        return ("COORD_NULAS", True)
    if proc.startswith("Coordenadas corrigidas automaticamente"):
        return ("COORD_SINAL_OU_TROCA_CORRIGIDA", True)
    if proc.startswith(("Coordenadas inválidas", "IBGE: Coordenadas inválidas")):
        return ("COORD_ABSURDA_SUBSTITUIDA_IBGE", True)
    if proc.startswith(("Coordenadas Fora do Território", "Fora do Território")):
        return ("FORA_TERRITORIO_BR", False)
    if proc.startswith("Código do Município não consta"):
        return ("MUN_NAO_CONSTA_IBGE", True)
    if proc.startswith("Valor Não Numérico Descartado"):
        return ("COORD_NAO_NUMERICA", True)
    return ("OUTRO", True)


@app.function
def parse_log_templates(
    log_raw_df: pd.DataFrame, rows_per_log: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Unnest factorized JSON log templates into (templates, events) tables.

    `rows_per_log` maps Log ID -> number of rows in estacoes referencing it.
    """
    tmpl_rows, event_rows = [], []
    for lid, raw in enumerate(log_raw_df["Log"]):
        items = json.loads(raw)
        n_rows = int(rows_per_log.get(lid, 0))
        tags, pretty_parts, has_dq = [], [], False
        for step_i, it in enumerate(items):
            proc = it.get("Log", "")
            tag, is_dq = classify_log_step(proc)
            has_dq = has_dq or is_dq
            tags.append(tag)
            # A step may touch several columns at once: Coluna/Valor are then parallel lists
            cols, vals = (
                it.get("Coluna", ""),
                it.get("Original", it.get("Original", "")),
            )
            if not isinstance(cols, list):
                cols, vals = [cols], [vals]
            detail = ", ".join(f"{c}: '{v}'" for c, v in zip(cols, vals) if c)
            pretty_parts.append(
                f"{step_i + 1}. {tag}" + (f" [{detail}]" if detail else "")
            )
            for col, val in zip(cols, vals):
                event_rows.append(
                    {
                        "Log": lid,
                        "Passo": step_i + 1,
                        "Tag": tag,
                        "Alerta_Qualidade": is_dq,
                        "Processamento": proc.replace("\n", " "),
                        "Coluna": col,
                        "Original": val,
                        "Estacoes_Afetadas": n_rows,
                    }
                )
        tmpl_rows.append(
            {
                "Log": lid,
                "Qtd_Passos": len(items),
                "Pipeline": " → ".join(dict.fromkeys(tags))
                if tags
                else "SEM_ALTERACAO",
                "Alerta_Qualidade": has_dq,
                "Resumo_Log": " | ".join(pretty_parts)
                if pretty_parts
                else "Sem alterações registradas",
                "JSON_Bruto": raw,
                "Estacoes_Afetadas": n_rows,
            }
        )
    return pd.DataFrame(tmpl_rows), pd.DataFrame(event_rows)


@app.cell
def _():
    ARQ_DIR = resolve_arq_dir()
    DATA_DIR = ARQ_DIR / "saida"

    _shp_candidates = sorted(ARQ_DIR.glob("BR_Municipios_*/*.shp"))
    SHP_PATH = (
        str(_shp_candidates[-1])
        if _shp_candidates
        else str(ARQ_DIR / "BR_Municipios_2025" / "BR_Municipios_2025.shp")
    )

    estacoes_parquet_path = DATA_DIR / "estacoes.parquet"
    log_parquet_path = DATA_DIR / "log.parquet"
    estacoes_size_mb = estacoes_parquet_path.stat().st_size / (1024**2)
    log_size_mb = log_parquet_path.stat().st_size / (1024**2)
    estacoes_mtime_str = datetime.fromtimestamp(
        estacoes_parquet_path.stat().st_mtime
    ).strftime("%Y-%m-%d %H:%M:%S")

    estacoes = pd.read_parquet(estacoes_parquet_path)
    log_raw_df = pd.read_parquet(log_parquet_path)
    ibge_mun = pd.read_csv(ARQ_DIR / "municipios.csv", dtype=str)

    log_templates, log_events = parse_log_templates(
        log_raw_df, estacoes["Log"].value_counts()
    )

    # Core numeric arrays shared by all sections
    freq_mhz = cat_to_float(estacoes["Frequência"])
    lat_num = cat_to_float(estacoes["Latitude"])
    lon_num = cat_to_float(estacoes["Longitude"])
    mult_num = np.nan_to_num(cat_to_float(estacoes["Multiplicidade"]), nan=1.0)
    return (
        DATA_DIR,
        SHP_PATH,
        estacoes,
        estacoes_mtime_str,
        estacoes_size_mb,
        freq_mhz,
        ibge_mun,
        lat_num,
        log_events,
        log_size_mb,
        log_templates,
        lon_num,
        mult_num,
    )


@app.function
def schema_profile(df: pd.DataFrame, max_top_len: int = 42) -> pd.DataFrame:
    """Per-column dtype, cardinality, '-1' sentinel count, fill % and most common value."""
    n_rows = max(len(df), 1)
    rows = []
    for col in df.columns:
        s = df[col]
        if str(s.dtype) == "category":
            n_sent = int((s == "-1").sum()) if "-1" in s.cat.categories else 0
            vc = s.value_counts()
            top_val = str(vc.index[0]) if len(vc) > 0 else ""
        else:
            n_sent = int((s == -1).sum())
            modes = s.mode()
            top_val = str(modes.iloc[0]) if len(modes) > 0 else ""
        if len(top_val) > max_top_len:
            top_val = top_val[: max_top_len - 3] + "..."
        rows.append(
            {
                "Coluna": col,
                "Tipo_Parquet": str(s.dtype),
                "Valores_Unicos": int(s.nunique()),
                "Sentinela_-1": n_sent,
                "Preenchido_%": round((1.0 - n_sent / n_rows) * 100.0, 2),
                "Valor_Mais_Comum": top_val,
            }
        )
    return pd.DataFrame(rows)


@app.cell(hide_code=True)
def _(
    OFFICIAL_UFS,
    estacoes,
    estacoes_size_mb,
    log_events,
    log_templates,
    mult_num,
):
    schema_profile_df = schema_profile(estacoes)

    _n_rows = max(len(estacoes), 1)
    _n_cols = len(estacoes.columns)
    _n_emit = int(mult_num.sum())
    _mun_missing = int((estacoes["Município"].astype(str) == "-1").sum())
    _mun_filled_pct = (1.0 - _mun_missing / _n_rows) * 100.0
    _uf_vals = set(estacoes["UF"].astype(str).unique())
    _valid_ufs_count = len(_uf_vals & OFFICIAL_UFS)
    _dirty_ufs = sorted(_uf_vals - OFFICIAL_UFS - {"-1"})
    _uf_missing = int((estacoes["UF"].astype(str) == "-1").sum())
    _low_fill_count = int((schema_profile_df["Preenchido_%"] < 50.0).sum())
    _srd_share_pct = float(
        estacoes["Fonte"].astype(str).isin(["MOSAICO-SRD", "SRD"]).mean() * 100.0
    )

    _uf_status_txt = (
        f"nenhuma UF fora do padrão IBGE ({_valid_ufs_count} UFs oficiais presentes e **{_uf_missing:,}** registros offshore/fronteira com `UF == '-1'`)"
        if not _dirty_ufs
        else f"**{len(_dirty_ufs)}** UFs anômalas detectadas (`{_dirty_ufs}`) além de {_valid_ufs_count} UFs oficiais e **{_uf_missing:,}** com `'-1'`"
    )

    _kpis = kpi_row(
        [
            (
                "Registros Consolidados",
                f"{len(estacoes):,}",
                f"{estacoes_size_mb:.1f} MB Parquet ({_n_cols} colunas)",
            ),
            (
                "Emissores Físicos (Σ Multiplicidade)",
                f"{_n_emit:,}",
                f"Média {mult_num.mean():.1f}x por linha agrupada",
            ),
            (
                "Templates em log.parquet",
                f"{len(log_templates):,}",
                f"{len(log_events):,} eventos JSON fatorados",
            ),
            (
                "Município Preenchido",
                f"{_mun_filled_pct:.2f}%",
                f"{_mun_missing:,} registros com sentinela '-1' ({100.0 - _mun_filled_pct:.2f}%)",
            ),
        ]
    )

    _miss_chart = (
        alt.Chart(schema_profile_df)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X(
                "Preenchido_%:Q",
                title="% Preenchido (não '-1')",
                scale=alt.Scale(domain=[0, 100]),
            ),
            y=alt.Y("Coluna:N", sort="-x", title=None),
            color=alt.condition(
                alt.datum["Preenchido_%"] < 50,
                alt.value("#e45756"),
                alt.value("#4c78a8"),
            ),
            tooltip=[
                "Coluna",
                "Preenchido_%",
                alt.Tooltip("Sentinela_-1:Q", format=","),
                alt.Tooltip("Valores_Unicos:Q", format=","),
                "Valor_Mais_Comum",
            ],
        )
        .properties(
            height=500,
            title=f"Completude por Coluna ({len(estacoes):,} registros | Sentinela '-1' = Ausente)",
        )
    )

    mo.vstack(
        [
            mo.md(
                f"""
    ## 1. Visão Geral & Completude do Esquema
    > **Síntese da extração atual:** A base consolidada soma **{len(estacoes):,} registros** (**{_n_emit:,} emissores físicos** considerando `Multiplicidade`). O campo `Município` está preenchido em **{_mun_filled_pct:.2f}%** da base (restando **{_mun_missing:,}** registros com `"-1"`), com {_uf_status_txt}. Atualmente **{_low_fill_count} de {_n_cols} colunas** apresentam preenchimento inferior a 50% — concentradas nos atributos técnicos de engenharia exclusivos das fontes de Radiodifusão (`MOSAICO-SRD` e `SRD`, que representam **{_srd_share_pct:.2f}%** dos registros).
    """
            ),
            _kpis,
            mo.ui.tabs(
                {
                    "📊 Gráfico de Completude": _miss_chart,
                    "📋 Tabela de Perfil das Colunas": mo.ui.table(
                        schema_profile_df, pagination=True, page_size=15
                    ),
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(
    OFFICIAL_UFS,
    SERVICO_NOMES,
    SPECTRUM_BAND_BINS,
    SPECTRUM_BAND_LABELS,
    estacoes,
    freq_mhz,
    mult_num,
):
    _n_total = max(len(estacoes), 1)
    _emit_total = max(float(mult_num.sum()), 1.0)

    _band_df = summarize_by(
        pd.DataFrame(
            {
                "Faixa": pd.cut(
                    freq_mhz, bins=SPECTRUM_BAND_BINS, labels=SPECTRUM_BAND_LABELS
                ),
                "Emissores": mult_num,
            }
        ),
        "Faixa",
        sort=False,
        observed=False,
    )

    _fonte_df = summarize_by(
        pd.DataFrame({"Fonte": estacoes["Fonte"].astype(str), "Emissores": mult_num}),
        "Fonte",
    ).head(12)
    _fonte_df["%_Registros"] = (_fonte_df["Registros"] / _n_total * 100.0).round(2)

    _serv_df = summarize_by(
        pd.DataFrame(
            {
                "Serviço": estacoes["Serviço"].astype(str),
                "Emissores": mult_num,
                "Freq_MHz": freq_mhz,
            }
        ),
        "Serviço",
        Freq_Mediana_MHz=("Freq_MHz", "median"),
    ).head(12)
    _serv_df["Descricao"] = _serv_df["Serviço"].map(
        lambda c: SERVICO_NOMES.get(c, f"Serviço {c}")
    )

    _uf_df = summarize_by(
        pd.DataFrame({"UF": estacoes["UF"].astype(str), "Emissores": mult_num}), "UF"
    )
    _uf_df["Anomala"] = ~_uf_df["UF"].isin(OFFICIAL_UFS)

    _ent_df = summarize_by(
        pd.DataFrame(
            {
                "Entidade": estacoes["Entidade"].astype(str),
                "Emissores": mult_num,
                "UF": estacoes["UF"].astype(str),
                "Servico": estacoes["Serviço"].astype(str),
            }
        ),
        "Entidade",
        UFs_Atendidas=("UF", "nunique"),
        Servico_Principal=("Servico", top_value),
    ).head(20)

    _top1_s = _serv_df.iloc[0]
    _top2_s = _serv_df.iloc[1] if len(_serv_df) > 1 else _top1_s
    _top1_pct = float(_top1_s["Registros"] / _n_total * 100.0)
    _top2_pct = (
        float(_top2_s["Registros"] / _n_total * 100.0) if len(_serv_df) > 1 else 0.0
    )
    _top2_emit_pct = float(
        _serv_df["Emissores_Fisicos"].iloc[:2].sum() / _emit_total * 100.0
    )
    _sub3_pct = float(np.nanmean((freq_mhz > 0) & (freq_mhz <= 3000.0)) * 100.0)
    _shf_pct = float(np.nanmean(freq_mhz >= 6000.0) * 100.0)

    _chart_bands = (
        alt.Chart(_band_df)
        .mark_bar(color="#2b8cbe", cornerRadiusEnd=3)
        .encode(
            x=alt.X("Registros:Q", title="Registros"),
            y=alt.Y("Faixa:N", sort=None, title="Faixa de Espectro"),
            tooltip=[
                "Faixa",
                alt.Tooltip("Registros:Q", format=","),
                alt.Tooltip("Emissores_Fisicos:Q", format=",.0f"),
            ],
        )
        .properties(width=350, height=270, title="Distribuição por Faixa de Frequência")
    )

    _chart_serv = (
        alt.Chart(_serv_df)
        .mark_bar(color="#54a24b", cornerRadiusEnd=3)
        .encode(
            x=alt.X("Registros:Q", title="Registros"),
            y=alt.Y("Descricao:N", sort="-x", title="Serviço Anatel"),
            tooltip=[
                "Descricao",
                alt.Tooltip("Registros:Q", format=","),
                alt.Tooltip("Emissores_Fisicos:Q", format=",.0f"),
                alt.Tooltip("Freq_Mediana_MHz:Q", format=".2f"),
            ],
        )
        .properties(
            width=350, height=270, title=f"Top {len(_serv_df)} Serviços (Anatel / Aero)"
        )
    )

    _chart_uf = (
        alt.Chart(_uf_df)
        .mark_bar(cornerRadiusEnd=2)
        .encode(
            x=alt.X("UF:N", sort="-y", title="Unidade da Federação"),
            y=alt.Y("Registros:Q", title="Registros"),
            color=alt.condition(
                alt.datum["Anomala"],
                alt.value("#e45756"),
                alt.value("#4c78a8"),
            ),
            tooltip=[
                "UF",
                alt.Tooltip("Registros:Q", format=","),
                alt.Tooltip("Emissores_Fisicos:Q", format=",.0f"),
                "Anomala",
            ],
        )
        .properties(
            height=250,
            title=f"Registros por UF (barras vermelhas = fora das {len(OFFICIAL_UFS)} UFs oficiais)",
        )
    )

    mo.vstack(
        [
            mo.md(
                f"""
    ## 2. Espectro, Serviços, Fontes & Operadoras
    - **Concentração por Serviço:** `{_top1_s["Descricao"]}` (**{_top1_pct:.1f}%**, {int(_top1_s["Registros"]):,} linhas) e `{_top2_s["Descricao"]}` (**{_top2_pct:.1f}%**, {int(_top2_s["Registros"]):,} linhas) concentram juntos **{_top1_pct + _top2_pct:.1f}%** dos registros e **{_top2_emit_pct:.1f}%** dos **{int(mult_num.sum()):,}** emissores físicos.
    - **Perfil Espectral:** **{_sub3_pct:.1f}%** das frequências válidas operam abaixo de 3 GHz (rádio privativo VHF/UHF, radiodifusão e cobertura móvel 2G/3G/4G) e **{_shf_pct:.1f}%** operam acima de 6 GHz (enlaces ponto-a-ponto SHF/EHF de *backhaul*).
    """
            ),
            mo.ui.tabs(
                {
                    "📡 Espectro & Serviços": mo.vstack(
                        [
                            mo.hstack([_chart_bands, _chart_serv], justify="start"),
                            _chart_uf,
                        ]
                    ),
                    f"🏢 Top {len(_ent_df)} Entidades": mo.ui.table(
                        _ent_df, pagination=False
                    ),
                    "🗂️ Top Fontes": mo.ui.table(_fonte_df, pagination=False),
                }
            ),
        ]
    )
    return


@app.function
def pipeline_summary(log_templates: pd.DataFrame, n_total: int) -> pd.DataFrame:
    """Stations affected per distinct processing pipeline (ordered unique tag sequence)."""
    out = (
        log_templates.groupby(["Pipeline", "Alerta_Qualidade"], as_index=False)
        .agg(
            Templates=("Log", "count"),
            Estacoes_Afetadas=("Estacoes_Afetadas", "sum"),
            Exemplo_Log_ID=("Log", "min"),
        )
        .sort_values("Estacoes_Afetadas", ascending=False)
    )
    out["%_da_Base"] = (out["Estacoes_Afetadas"] / max(n_total, 1) * 100).round(3)
    return out


@app.function
def tag_summary(log_events: pd.DataFrame, log_templates: pd.DataFrame) -> pd.DataFrame:
    """Unique stations and column-level events per Tag.

    Deduplicates (Log, Tag) first so multi-column templates (#Latitude + #Longitude)
    do not double-count stations.
    """
    stations_per_log = log_templates.set_index("Log")["Estacoes_Afetadas"]
    per_log = log_events.groupby(
        ["Tag", "Alerta_Qualidade", "Log"], as_index=False
    ).agg(
        Colunas_Afetadas=("Coluna", "count"),
        Exemplo_Processamento=("Processamento", "first"),
    )
    per_log["Estacoes_Unicas"] = (
        per_log["Log"].map(stations_per_log).fillna(0).astype(int)
    )
    per_log["Eventos_Coluna"] = per_log["Estacoes_Unicas"] * per_log["Colunas_Afetadas"]
    return (
        per_log.groupby(["Tag", "Alerta_Qualidade"], as_index=False)
        .agg(
            Templates_com_Evento=("Log", "nunique"),
            Estacoes_Unicas=("Estacoes_Unicas", "sum"),
            Eventos_Coluna=("Eventos_Coluna", "sum"),
            Processamento=("Exemplo_Processamento", "first"),
        )
        .sort_values("Estacoes_Unicas", ascending=False)
    )


@app.function
def log_original_values(
    log_events: pd.DataFrame, tag: str, coluna: str | None = None
) -> pd.Series:
    """Logged `Valor` (pre-change value or, for checks, the reference value) per Log ID for a given Tag (and optionally Coluna)."""
    ev = log_events[log_events["Tag"] == tag]
    if coluna is not None:
        ev = ev[ev["Coluna"] == coluna]
    return ev.drop_duplicates("Log").set_index("Log")["Original"]


@app.function
def smp_centroid_shift(
    log_events: pd.DataFrame, estacoes: pd.DataFrame
) -> pd.DataFrame:
    """Distance between the original SMP tower coordinate and the municipal centroid assigned on grouping."""
    orig_lat = log_original_values(log_events, "SMP_LAT_MUNICIPIO")
    orig_lon = log_original_values(log_events, "SMP_LON_MUNICIPIO")
    lids = orig_lat.index.intersection(orig_lon.index)
    out = (
        estacoes[estacoes["Log"].isin(lids)][
            ["Log", "UF", "Município", "Latitude", "Longitude"]
        ]
        .drop_duplicates("Log")
        .copy()
    )
    out["Orig_Lat"] = pd.to_numeric(out["Log"].map(orig_lat), errors="coerce")
    out["Orig_Lon"] = pd.to_numeric(out["Log"].map(orig_lon), errors="coerce")
    # "-1" is the missing-coordinate sentinel, not a centroid
    out["Mun_Lat"] = pd.to_numeric(
        out["Latitude"].astype(str).replace("-1", None), errors="coerce"
    )
    out["Mun_Lon"] = pd.to_numeric(
        out["Longitude"].astype(str).replace("-1", None), errors="coerce"
    )
    out = out[
        valid_wgs84_mask(out["Orig_Lat"], out["Orig_Lon"])
        & valid_wgs84_mask(out["Mun_Lat"], out["Mun_Lon"])
    ].copy()
    out["Deslocamento_km"] = np.round(
        haversine_km(out["Orig_Lat"], out["Orig_Lon"], out["Mun_Lat"], out["Mun_Lon"]),
        2,
    )
    return out


@app.cell
def _(estacoes, log_events, log_templates):
    pipe_summary_df = pipeline_summary(log_templates, len(estacoes))
    tag_summary_df = tag_summary(log_events, log_templates)
    smp_shift_df = smp_centroid_shift(log_events, estacoes)
    return pipe_summary_df, smp_shift_df, tag_summary_df


@app.cell(hide_code=True)
def _(log_events, log_templates):
    # Controls for exploring log.parquet & linked estacoes
    _all_tags = sorted(log_events["Tag"].unique().tolist())
    _max_lid = max(0, int(len(log_templates) - 1))

    log_tag_filter = mo.ui.dropdown(
        options=["(Todos os Eventos)"] + _all_tags,
        value="(Todos os Eventos)",
        label="Filtrar por Evento (Tag):",
    )
    log_only_dq = mo.ui.checkbox(
        value=True,
        label="Apenas eventos de correção / alerta de qualidade (ignorar rotina SMP/Agrupamento)",
    )
    log_id_selector = mo.ui.number(
        start=0,
        stop=_max_lid,
        step=1,
        value=min(8, _max_lid),
        label=f"Inspecionar Log ID (0–{_max_lid:,}):",
    )
    return log_id_selector, log_only_dq, log_tag_filter


@app.cell(hide_code=True)
def _(
    estacoes,
    log_events,
    log_id_selector,
    log_only_dq,
    log_tag_filter,
    log_templates,
    pipe_summary_df,
    smp_shift_df,
    tag_summary_df,
):
    _tag_stations = dict(zip(tag_summary_df["Tag"], tag_summary_df["Estacoes_Unicas"]))
    _tag_col_events = dict(zip(tag_summary_df["Tag"], tag_summary_df["Eventos_Coluna"]))

    _n_smp = len(smp_shift_df)
    _smp_med_km = float(smp_shift_df["Deslocamento_km"].median()) if _n_smp else 0.0
    _smp_p90_km = (
        float(smp_shift_df["Deslocamento_km"].quantile(0.9)) if _n_smp else 0.0
    )
    _smp_max_km = float(smp_shift_df["Deslocamento_km"].max()) if _n_smp else 0.0
    _smp_shift_top = smp_shift_df.sort_values("Deslocamento_km", ascending=False).head(
        30
    )

    _ev_view = log_events
    if log_only_dq.value:
        _ev_view = _ev_view[_ev_view["Alerta_Qualidade"]]
    if log_tag_filter.value != "(Todos os Eventos)":
        _ev_view = _ev_view[_ev_view["Tag"] == log_tag_filter.value]

    _sel_lid = min(max(0, int(log_id_selector.value)), max(0, len(log_templates) - 1))
    _sel_tmpl = log_templates.loc[log_templates["Log"] == _sel_lid].iloc[0]
    _sel_steps = log_events[log_events["Log"] == _sel_lid][
        ["Passo", "Tag", "Coluna", "Original", "Processamento"]
    ]
    _linked_stations = (
        estacoes[estacoes["Log"] == _sel_lid][
            [
                "Frequência",
                "Entidade",
                "Serviço",
                "UF",
                "Município",
                "Código_Município",
                "Latitude",
                "Longitude",
                "Fonte",
                "Multiplicidade",
                "Log",
            ]
        ]
        .head(50)
        .astype(str)
    )

    _tag_chart = (
        alt.Chart(tag_summary_df)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X(
                "Estacoes_Unicas:Q",
                title="Estações Únicas Afetadas (escala symlog)",
                scale=alt.Scale(type="symlog"),
            ),
            y=alt.Y("Tag:N", sort="-x", title="Evento de Log"),
            color=alt.condition(
                alt.datum["Alerta_Qualidade"],
                alt.value("#f58518"),
                alt.value("#4c78a8"),
            ),
            tooltip=[
                "Tag",
                "Processamento",
                alt.Tooltip("Estacoes_Unicas:Q", format=",", title="Estações Únicas"),
                alt.Tooltip("Eventos_Coluna:Q", format=",", title="Eventos de Coluna"),
                alt.Tooltip("Templates_com_Evento:Q", format=",", title="Templates"),
            ],
        )
        .properties(
            height=360,
            title=f"Estações Únicas por Tipo de Evento em log.parquet ({len(log_templates):,} Templates)",
        )
    )

    _n_div_st = int(_tag_stations.get("IBGE_COORD_DIVERGENTE", 0))
    _n_div_ev = int(_tag_col_events.get("IBGE_COORD_DIVERGENTE", 0))
    _n_poly_rec = int(_tag_stations.get("IBGE_POLIGONO_RECUPERADO", 0))
    _n_auto_fix = int(_tag_stations.get("COORD_SINAL_OU_TROCA_CORRIGIDA", 0)) + int(
        _tag_stations.get("COORD_ABSURDA_SUBSTITUIDA_IBGE", 0)
    )
    _n_null_fill = int(_tag_stations.get("COORD_NULA_PREENCHIDA_IBGE", 0))

    mo.vstack(
        [
            mo.md(
                f"""
    ## 3. Explorador Estruturado de `log.parquet` (Rastreabilidade Antes → Depois)
    O dicionário fatorado `log.parquet` contém **{len(log_templates):,} templates distintos** (**{len(log_events):,} eventos atômicos**), permitindo auditar cada transformação aplicada às **{len(estacoes):,}** estações:

    - **`IBGE_COORD_DIVERGENTE`:** **{_n_div_st:,} estações únicas** ({_n_div_ev:,} eventos `NM_MUN`) apresentaram coordenada em polígono municipal distinto do `Código_Município` declarado — a divergência é apenas registrada no `Log` (com o nome do município do polígono, `NM_MUN`, em `Valor`) e as coordenadas são preservadas — analisadas em detalhe na **Seção 4** abaixo.
    - **Recuperação Geoespacial Automática:** **{_n_poly_rec:,} estações** sem código IBGE recuperaram `Código_Município`/`Município`/`UF` via interseção com o polígono territorial (`IBGE_POLIGONO_RECUPERADO`), **{_n_auto_fix:,} estações** tiveram sinal/troca de coordenadas ou Null Island corrigidos automaticamente, e **{_n_null_fill:,} estações** sem coordenadas receberam o centroide municipal.
    - **Deslocamento ERB → Centroide no `MOSAICO-SMP`:** Em **{_n_smp:,} templates SMP** agrupados por município, o deslocamento mediano entre a coordenada original da ERB e o centroide municipal é de **{_smp_med_km:.2f} km** (p90 = **{_smp_p90_km:.2f} km**, máx = **{_smp_max_km:,.1f} km**).
    """
            ),
            mo.ui.tabs(
                {
                    "🔀 Pipelines & Gráfico de Eventos": mo.vstack(
                        [
                            _tag_chart,
                            mo.md(
                                "### Resumo por Tag (Estações Únicas vs. Eventos de Coluna)"
                            ),
                            mo.ui.table(tag_summary_df, pagination=False),
                            mo.md(
                                "### Top 25 Sequências de Processamento (`Pipeline`)"
                            ),
                            mo.ui.table(
                                pipe_summary_df.head(25), pagination=True, page_size=15
                            ),
                        ]
                    ),
                    "📏 Deslocamento Real ERB → Município (SMP)": mo.vstack(
                        [
                            mo.md(
                                "Distância geodésica entre a coordenada original da torre (`Original` preservado no `Log`) e o centroide municipal atribuído no agrupamento SMP:"
                            ),
                            mo.ui.table(_smp_shift_top, pagination=True, page_size=12),
                        ]
                    ),
                    "🔬 Eventos Desaninhados (Antes vs Depois)": mo.vstack(
                        [
                            mo.hstack([log_tag_filter, log_only_dq], justify="start"),
                            mo.ui.table(_ev_view, pagination=True, page_size=12),
                        ]
                    ),
                    "🔗 Inspetor de Log ID + Join com Estações": mo.vstack(
                        [
                            log_id_selector,
                            mo.callout(
                                mo.md(
                                    f"**Log ID `{_sel_lid}`** — **{_sel_tmpl['Estacoes_Afetadas']:,} estações** vinculadas | "
                                    f"**Pipeline:** `{_sel_tmpl['Pipeline']}`\n\n"
                                    f"**JSON bruto:** `{_sel_tmpl['JSON_Bruto']}`"
                                ),
                                kind="warn"
                                if _sel_tmpl["Alerta_Qualidade"]
                                else "info",
                            ),
                            mo.md("**Passos desaninhados deste Log ID:**"),
                            mo.ui.table(_sel_steps, pagination=False),
                            mo.md(
                                f"**Amostra de estações com `Log == {_sel_lid}` (até 50 de {_sel_tmpl['Estacoes_Afetadas']:,}):**"
                            ),
                            mo.ui.table(
                                _linked_stations, pagination=True, page_size=10
                            ),
                        ]
                    ),
                }
            ),
        ]
    )
    return


@app.function
@mo.persistent_cache
def compute_divergent_forensics(
    unique_pairs: pd.DataFrame, shp_path_str: str
) -> pd.DataFrame:
    """Intersect original pre-substitution coordinates with IBGE polygons and measure exact distance to declared border."""
    if len(unique_pairs) == 0:
        return pd.DataFrame(
            columns=[
                "Cod_Decl",
                "Orig_Lat",
                "Orig_Lon",
                "Real_CD_MUN",
                "Real_NM_MUN",
                "Real_UF",
                "Dist_Borda_km",
                "Vizinho",
            ]
        )

    regions = gpd.read_file(
        shp_path_str, columns=["CD_MUN", "NM_MUN", "SIGLA_UF", "geometry"]
    )
    regions["CD_MUN"] = regions["CD_MUN"].astype(str)

    pts_wgs = shapely.points(
        unique_pairs["Orig_Lon"].to_numpy(), unique_pairs["Orig_Lat"].to_numpy()
    )
    tree = shapely.STRtree(regions.geometry.to_numpy())
    pt_i, poly_i = tree.query(pts_wgs, predicate="intersects")
    _, first_occ = np.unique(pt_i, return_index=True)
    pt_i, poly_i = pt_i[first_occ], poly_i[first_occ]

    res = unique_pairs.copy()
    res["Real_CD_MUN"] = ""
    res["Real_NM_MUN"] = ""
    res["Real_UF"] = ""
    res.loc[pt_i, "Real_CD_MUN"] = regions["CD_MUN"].iloc[poly_i].to_numpy()
    res.loc[pt_i, "Real_NM_MUN"] = regions["NM_MUN"].iloc[poly_i].to_numpy()
    res.loc[pt_i, "Real_UF"] = regions["SIGLA_UF"].iloc[poly_i].to_numpy()

    # Exact metric distance in SIRGAS 2000 / Brazil Polyconic (EPSG:5880)
    regions_m = regions.to_crs("EPSG:5880").set_index("CD_MUN")
    pts_m = (
        gpd.GeoDataFrame(
            res,
            geometry=gpd.points_from_xy(res["Orig_Lon"], res["Orig_Lat"]),
            crs=regions.crs,
        )
        .to_crs("EPSG:5880")
        .geometry.to_numpy()
    )
    decl_geoms = regions_m.geometry.reindex(res["Cod_Decl"]).to_numpy()
    real_geoms = regions_m.geometry.reindex(res["Real_CD_MUN"]).to_numpy()

    valid_g = pd.notna(decl_geoms) & pd.notna(real_geoms)
    dist_km = np.full(len(res), np.nan, dtype="float64")
    vizinho = np.zeros(len(res), dtype=bool)
    if valid_g.any():
        dist_km[valid_g] = (
            shapely.distance(pts_m[valid_g], decl_geoms[valid_g]) / 1000.0
        )
        vizinho[valid_g] = shapely.intersects(decl_geoms[valid_g], real_geoms[valid_g])

    res["Dist_Borda_km"] = dist_km
    res["Vizinho"] = vizinho
    return res


@app.function
def classify_divergent_regime(
    dist_borda_km: pd.Series, vizinho: pd.Series, mesma_uf: pd.Series
) -> np.ndarray:
    """Spatial regime code (keys of DIVERGENT_REGIME_META) from border distance and adjacency."""
    conditions = [
        dist_borda_km <= 0.1,
        dist_borda_km <= 1.0,
        vizinho & (dist_borda_km <= 10.0),
        vizinho & (dist_borda_km > 10.0),
        (~vizinho) & mesma_uf,
    ]
    codes = [
        "1_Micro_Fronteira_100m",
        "2_Topo_Serra_Borda_1km",
        "3_Limitrofe_1_10km",
        "4_Limitrofe_Maior_10km",
        "5_Nao_Vizinho_Mesma_UF",
    ]
    return np.select(conditions, codes, default="6_Outra_UF_Nao_Vizinho")


@app.function
def build_divergent_records(
    estacoes: pd.DataFrame,
    log_events: pd.DataFrame,
    ibge_mun: pd.DataFrame,
    shp_path: str,
    regime_meta: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    """One row per IBGE_COORD_DIVERGENTE station: kept coordinate vs. declared municipality centroid, plus forensic diagnostics.

    The pipeline only logs the divergence (polygon `NM_MUN` in `Valor`) and keeps the
    original coordinates, so `New_Lat`/`New_Lon` is the declared municipality's IBGE centroid.
    """
    log_nm_mun = log_original_values(log_events, "IBGE_COORD_DIVERGENTE")

    base = estacoes[estacoes["Log"].isin(log_nm_mun.index)].copy()
    base["Cod_Decl"] = base["Código_Município"].astype(str)
    base["UF_Decl"] = base["UF"].astype(str)
    base["Mun_Decl"] = base["Município"].astype(str)
    base["Log_NM_MUN"] = base["Log"].map(log_nm_mun).astype(str)
    base["Orig_Lat"] = pd.to_numeric(base["Latitude"].astype(str), errors="coerce")
    base["Orig_Lon"] = pd.to_numeric(base["Longitude"].astype(str), errors="coerce")
    centroids = ibge_mun.set_index("Código_Município")[["Latitude", "Longitude"]]
    base["New_Lat"] = pd.to_numeric(
        base["Cod_Decl"].map(centroids["Latitude"]), errors="coerce"
    )
    base["New_Lon"] = pd.to_numeric(
        base["Cod_Decl"].map(centroids["Longitude"]), errors="coerce"
    )
    base["Mult_Num"] = np.nan_to_num(cat_to_float(base["Multiplicidade"]), nan=1.0)

    keys = ["Cod_Decl", "Orig_Lat", "Orig_Lon"]
    unique_pairs = base[keys].dropna().drop_duplicates().reset_index(drop=True)
    rec = base.merge(
        compute_divergent_forensics(unique_pairs, shp_path), on=keys, how="left"
    )

    rec["Shift_Centroid_km"] = haversine_km(
        rec["Orig_Lat"], rec["Orig_Lon"], rec["New_Lat"], rec["New_Lon"]
    )
    rec["Mesma_UF"] = rec["UF_Decl"] == rec["Real_UF"].astype(str)
    rec["Vizinho"] = rec["Vizinho"].fillna(False).astype(bool)
    rec["Regime_Code"] = classify_divergent_regime(
        rec["Dist_Borda_km"], rec["Vizinho"], rec["Mesma_UF"]
    )
    rec["Regime_Desc"] = rec["Regime_Code"].map(lambda k: regime_meta[k][1])
    return rec


@app.function
def aggregate_divergent_sites(records: pd.DataFrame) -> pd.DataFrame:
    """Collapse divergent records (one per channel) into unique physical sites."""
    site_keys = [
        "Cod_Decl",
        "UF_Decl",
        "Mun_Decl",
        "Orig_Lat",
        "Orig_Lon",
        "New_Lat",
        "New_Lon",
        "Real_CD_MUN",
        "Real_NM_MUN",
        "Real_UF",
        "Vizinho",
        "Mesma_UF",
        "Regime_Code",
        "Regime_Desc",
    ]
    aggs = dict(
        Canais_Afetados=("Log", "size"),
        Emissores_Fisicos=("Mult_Num", "sum"),
        Estacoes_IDs=("Estação", lambda s: int(s.astype(str).nunique())),
        Dist_Borda_km=("Dist_Borda_km", "first"),
        Shift_Centroid_km=("Shift_Centroid_km", "first"),
        Entidade_Principal=("Entidade", top_value),
        Fonte_Principal=("Fonte", top_value),
        Servico_Principal=("Serviço", top_value),
        Exemplo_Estacao=("Estação", lambda s: str(s.astype(str).iloc[0])),
        Exemplo_Log_ID=("Log", "min"),
    )
    if records.empty:
        return pd.DataFrame(columns=site_keys + list(aggs))
    return (
        records.groupby(site_keys, observed=True, as_index=False)
        .agg(**aggs)
        .sort_values("Canais_Afetados", ascending=False)
        .reset_index(drop=True)
    )


@app.function
def summarize_divergent_regimes(
    records: pd.DataFrame,
    sites: pd.DataFrame,
    regime_meta: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    """Records, share, unique sites and median distances per spatial regime."""
    rows = []
    for code, (_color, desc) in regime_meta.items():
        sub_r = records[records["Regime_Code"] == code]
        n = len(sub_r)
        rows.append(
            {
                "Regime_Code": code,
                "Regime": desc,
                "Registros": n,
                "%_Divergentes": round(n / max(len(records), 1) * 100.0, 2),
                "Sites_Fisicos_Unicos": int((sites["Regime_Code"] == code).sum()),
                "Dist_Divisa_Mediana_km": round(
                    float(sub_r["Dist_Borda_km"].median()), 3
                )
                if n > 0
                else 0.0,
                "Shift_Centroide_Mediano_km": round(
                    float(sub_r["Shift_Centroid_km"].median()), 2
                )
                if n > 0
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


@app.function
def divergent_concentration_metrics(
    records: pd.DataFrame,
    sites: pd.DataFrame,
    servico_nomes: dict[str, str],
    heavy_site_channels: int = 50,
) -> dict:
    """Headline numbers for the Section 4 narrative (border effect, adjacency, concentration)."""
    n_div, n_sites = len(records), len(sites)
    denom = max(n_div, 1)
    dist = records["Dist_Borda_km"]
    shift = records["Shift_Centroid_km"]
    heavy = sites[sites["Canais_Afetados"] >= heavy_site_channels]

    def top_share(values: pd.Series) -> tuple[str, int, float]:
        if n_div == 0:
            return ("N/A", 0, 0.0)
        vc = values.astype(str).value_counts()
        return (str(vc.index[0]), int(vc.iloc[0]), int(vc.iloc[0]) / n_div * 100.0)

    top_fonte, top_fonte_n, top_fonte_pct = top_share(records["Fonte"])
    top_serv, _top_serv_n, top_serv_pct = top_share(records["Serviço"])
    top_uf, top_uf_n, top_uf_pct = top_share(records["UF_Decl"])
    top_ent, top_ent_n, _top_ent_pct = top_share(records["Entidade"])
    top_ent_sites = (
        int(
            records.loc[
                records["Entidade"].astype(str) == top_ent, ["Orig_Lat", "Orig_Lon"]
            ]
            .drop_duplicates()
            .shape[0]
        )
        if n_div
        else 0
    )
    n_viz = int(records["Vizinho"].sum())
    n_100m = int((dist <= 0.1).sum())
    n_1km = int((dist <= 1.0).sum())
    n_heavy_rows = int(heavy["Canais_Afetados"].sum())

    return {
        "n_div": n_div,
        "n_sites": n_sites,
        "emit": int(records["Mult_Num"].sum()) if n_div else 0,
        "avg_ch_per_site": n_div / max(n_sites, 1),
        "max_ch_site": int(sites["Canais_Afetados"].max()) if n_sites else 0,
        "n_100m": n_100m,
        "n_1km": n_1km,
        "pct_1km": n_1km / denom * 100.0,
        "n_viz": n_viz,
        "pct_viz": n_viz / denom * 100.0,
        "n_non_viz": n_div - n_viz,
        "pct_non_viz": (n_div - n_viz) / denom * 100.0,
        "med_borda": float(dist.median()) if n_div else 0.0,
        "med_shift": float(shift.median()) if n_div else 0.0,
        "pct_shift_worse": float((shift > dist).mean() * 100.0) if n_div else 0.0,
        "heavy_site_channels": heavy_site_channels,
        "n_heavy_sites": len(heavy),
        "n_heavy_rows": n_heavy_rows,
        "pct_heavy_rows": n_heavy_rows / denom * 100.0,
        "top_fonte": top_fonte,
        "top_fonte_n": top_fonte_n,
        "top_fonte_pct": top_fonte_pct,
        "top_serv_lbl": servico_nomes.get(top_serv, f"Serviço {top_serv}")
        if n_div
        else "N/A",
        "top_serv_pct": top_serv_pct,
        "top_uf": top_uf,
        "top_uf_n": top_uf_n,
        "top_uf_pct": top_uf_pct,
        "top_ent": top_ent,
        "top_ent_n": top_ent_n,
        "top_ent_sites": top_ent_sites,
    }


@app.function
def divergent_top_entities(records: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    """Entities with most divergent records, with unique physical sites and median border distance."""
    if records.empty:
        out = pd.DataFrame(
            columns=["Entidade", "Registros", "Sites_Unicos", "Dist_Divisa_Med_km"]
        )
    else:
        out = (
            records.groupby("Entidade", observed=True)
            .agg(
                Registros=("Log", "size"),
                Sites_Unicos=(
                    "Orig_Lat",
                    lambda s: int(
                        records.loc[s.index, ["Orig_Lat", "Orig_Lon"]]
                        .drop_duplicates()
                        .shape[0]
                    ),
                ),
                Dist_Divisa_Med_km=("Dist_Borda_km", "median"),
            )
            .sort_values("Registros", ascending=False)
            .head(n)
            .reset_index()
        )
    out["Entidade_Curta"] = out["Entidade"].astype(str).str.slice(0, 38)
    return out


@app.function
def divergent_top_pairs(records: pd.DataFrame, n: int = 40) -> pd.DataFrame:
    """Most frequent (declared municipality -> real polygon) pairs."""
    if records.empty:
        return pd.DataFrame()
    out = (
        records.groupby(
            ["UF_Decl", "Mun_Decl", "Real_UF", "Real_NM_MUN", "Vizinho"],
            observed=True,
            as_index=False,
        )
        .agg(
            Registros=("Log", "size"),
            Sites_Unicos=("Orig_Lat", "nunique"),
            Dist_Divisa_Med_km=("Dist_Borda_km", "median"),
            Shift_Centroide_Med_km=("Shift_Centroid_km", "median"),
            Entidade_Principal=("Entidade", top_value),
        )
        .sort_values("Registros", ascending=False)
        .head(n)
    )
    out["Dist_Divisa_Med_km"] = out["Dist_Divisa_Med_km"].round(3)
    out["Shift_Centroide_Med_km"] = out["Shift_Centroide_Med_km"].round(2)
    return out


@app.function
def filter_divergent_sites(
    sites: pd.DataFrame,
    regime_codes: set[str],
    uf: str,
    fonte: str,
    query: str,
    all_option: str,
) -> pd.DataFrame:
    """Apply the Section 4 UI filters to the physical-site table (only plottable original coordinates)."""
    mask = sites["Regime_Code"].isin(regime_codes).to_numpy(copy=True)
    mask &= valid_wgs84_mask(sites["Orig_Lat"], sites["Orig_Lon"])
    mask &= selection_mask(sites["UF_Decl"], uf, all_option)
    mask &= selection_mask(sites["Fonte_Principal"], fonte, all_option)
    mask &= text_search_mask(
        sites, ["Entidade_Principal", "Mun_Decl", "Real_NM_MUN"], query
    )
    return sites[mask].copy()


@app.function
def divergent_site_tooltip(site, color: str, desc: str) -> str:
    """HTML tooltip for one physical site (row from `div_sites_df.itertuples()`)."""
    esc = lambda v: html.escape(str(v))
    return (
        f"<div style='font-family:sans-serif;font-size:12px;min-width:290px;max-width:380px;'>"
        f"<div style='background:{color};color:#fff;padding:4px 7px;border-radius:4px;font-weight:bold;margin-bottom:5px;'>"
        f"{esc(desc)}</div>"
        f"<b>Canais neste Site:</b> {site.Canais_Afetados:,} (Σ Mult: {int(site.Emissores_Fisicos):,}) | "
        f"<b>Estação ex.:</b> {esc(site.Exemplo_Estacao)}<br>"
        f"<b>Entidade:</b> {esc(site.Entidade_Principal)}<br>"
        f"<b>Fonte:</b> {esc(site.Fonte_Principal)} | <b>Serviço:</b> {esc(site.Servico_Principal)} | "
        f"<b>Log ID:</b> {site.Exemplo_Log_ID}<br>"
        f"<hr style='margin:4px 0'>"
        f"<b>Município Declarado:</b> {esc(site.Mun_Decl)}/{esc(site.UF_Decl)} ({esc(site.Cod_Decl)})<br>"
        f"<b>Polígono da Coordenada:</b> {esc(site.Real_NM_MUN)}/{esc(site.Real_UF)} ({esc(site.Real_CD_MUN)})<br>"
        f"<b>Coordenada (preservada):</b> ({float(site.Orig_Lat):.6f}, {float(site.Orig_Lon):.6f})<br>"
        f"<b>Centroide do Município Declarado:</b> ({float(site.New_Lat):.6f}, {float(site.New_Lon):.6f})<br>"
        f"<b>Distância até Divisa Declarada:</b> <span style='color:#d95f02;font-weight:bold'>{format_distance(site.Dist_Borda_km)}</span> | "
        f"<b>Distância ao Centroide Declarado:</b> {format_distance(site.Shift_Centroid_km, meters_below_1km=False)}"
        f"</div>"
    )


@app.function
def build_divergent_map(
    sites: pd.DataFrame,
    regime_meta: dict[str, tuple[str, str]],
    regime_codes: set[str],
    show_vectors: bool,
) -> folium.Map:
    """Circle at the (preserved) tower coordinate, dashed vector to the declared municipality's IBGE centroid."""
    fmap = base_folium_map([-16.5, -47.5], 5, light_basemap=True)
    layers = {
        code: folium.FeatureGroup(name=regime_meta[code][1], show=True).add_to(fmap)
        for code in sorted(regime_codes)
    }
    for site in sites.itertuples():
        color, desc = regime_meta[site.Regime_Code]
        orig = [float(site.Orig_Lat), float(site.Orig_Lon)]
        new = [float(site.New_Lat), float(site.New_Lon)]
        layer = layers[site.Regime_Code]
        if show_vectors and valid_wgs84_mask(new[0], new[1]):
            folium.PolyLine(
                locations=[orig, new],
                color=color,
                weight=1.5,
                opacity=0.45,
                dash_array="4,4",
            ).add_to(layer)
        folium.CircleMarker(
            location=orig,
            radius=float(
                min(12.0, max(4.5, 3.5 + np.log1p(site.Canais_Afetados) * 1.3))
            ),
            color=color,
            weight=1.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.82,
            tooltip=folium.Tooltip(
                divergent_site_tooltip(site, color, desc), sticky=True
            ),
        ).add_to(layer)
    folium.LayerControl(collapsed=True).add_to(fmap)
    return fmap


@app.cell
def _(
    DIVERGENT_REGIME_META,
    SERVICO_NOMES,
    SHP_PATH,
    estacoes,
    ibge_mun,
    log_events,
):
    div_records_df = build_divergent_records(
        estacoes, log_events, ibge_mun, SHP_PATH, DIVERGENT_REGIME_META
    )
    div_sites_df = aggregate_divergent_sites(div_records_df)
    div_regime_summary_df = summarize_divergent_regimes(
        div_records_df, div_sites_df, DIVERGENT_REGIME_META
    )
    div_metrics = divergent_concentration_metrics(
        div_records_df, div_sites_df, SERVICO_NOMES
    )
    div_top_entities_df = divergent_top_entities(div_records_df)
    div_top_pairs_df = divergent_top_pairs(div_records_df)
    return (
        div_metrics,
        div_regime_summary_df,
        div_sites_df,
        div_top_entities_df,
        div_top_pairs_df,
    )


@app.cell(hide_code=True)
def _(ALL_OPTION, div_regime_summary_df, div_sites_df):
    _div_reg_opts = {
        f"{r.Regime} [{r.Registros:,} reg / {r.Sites_Fisicos_Unicos:,} sites]": r.Regime_Code
        for r in div_regime_summary_df.itertuples()
        if r.Registros > 0
    }
    div_regime_select = mo.ui.multiselect(
        options=_div_reg_opts,
        value=list(_div_reg_opts.keys()),
        label="Regimes Espaciais:",
    )

    div_uf_select = mo.ui.dropdown(
        options=[ALL_OPTION]
        + sorted(div_sites_df["UF_Decl"].astype(str).unique().tolist()),
        value=ALL_OPTION,
        label="UF Declarada:",
    )
    div_fonte_select = mo.ui.dropdown(
        options=[ALL_OPTION]
        + sorted(div_sites_df["Fonte_Principal"].astype(str).unique().tolist()),
        value=ALL_OPTION,
        label="Fonte Principal:",
    )
    div_entity_search = mo.ui.text(
        value="",
        placeholder="Ex: POLICIA MILITAR, TELEFONICA, VALE, Nova Lima...",
        label="Buscar Entidade ou Município:",
    )
    div_max_sites = mo.ui.slider(
        start=100, stop=2000, step=100, value=600, label="Máx. Sites Físicos no Mapa:"
    )
    div_show_vectors = mo.ui.checkbox(
        value=True,
        label="Traçar vetor de deslocamento (Torre Original → Centroide IBGE)",
    )
    return (
        div_entity_search,
        div_fonte_select,
        div_max_sites,
        div_regime_select,
        div_show_vectors,
        div_uf_select,
    )


@app.cell(hide_code=True)
def _(
    ALL_OPTION,
    DIVERGENT_REGIME_META,
    div_entity_search,
    div_fonte_select,
    div_max_sites,
    div_metrics,
    div_regime_select,
    div_regime_summary_df,
    div_show_vectors,
    div_sites_df,
    div_top_entities_df,
    div_top_pairs_df,
    div_uf_select,
    estacoes,
):
    _m = div_metrics
    _div_pct_base = _m["n_div"] / max(len(estacoes), 1) * 100.0

    _div_kpis = kpi_row(
        [
            (
                "Estações c/ Coord. Divergente",
                f"{_m['n_div']:,}",
                f"{_div_pct_base:.2f}% da base ({_m['emit']:,} emissores)",
            ),
            (
                "Sites Físicos Únicos (Coord.)",
                f"{_m['n_sites']:,}",
                f"Média {_m['avg_ch_per_site']:.1f} canais/site (máx {_m['max_ch_site']:,})",
            ),
            (
                "Na Divisa / Topo de Serra (≤ 1 km)",
                f"{_m['n_1km']:,} ({_m['pct_1km']:.1f}%)",
                f"{_m['n_100m']:,} registros a ≤ 100 m da fronteira",
            ),
            (
                "Em Município Limítrofe",
                f"{_m['n_viz']:,} ({_m['pct_viz']:.1f}%)",
                f"Mediana {_m['med_borda']:.2f} km da divisa → {_m['med_shift']:.2f} km p/ centroide",
            ),
        ]
    )

    _chart_regimes = (
        alt.Chart(div_regime_summary_df)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("Registros:Q", title="Estações Afetadas"),
            y=alt.Y(
                "Regime:N",
                sort=[v[1] for v in DIVERGENT_REGIME_META.values()],
                title=None,
            ),
            color=alt.Color(
                "Regime_Code:N",
                scale=meta_color_scale(DIVERGENT_REGIME_META),
                legend=None,
            ),
            tooltip=[
                "Regime",
                alt.Tooltip("Registros:Q", format=","),
                alt.Tooltip("%_Divergentes:Q", format=".1f"),
                alt.Tooltip("Sites_Fisicos_Unicos:Q", format=","),
                alt.Tooltip("Dist_Divisa_Mediana_km:Q", format=".3f"),
                alt.Tooltip("Shift_Centroide_Mediano_km:Q", format=".2f"),
            ],
        )
        .properties(width=380, height=250, title="Decomposição por Regime Geoespacial")
    )

    _chart_div_ent = (
        alt.Chart(div_top_entities_df)
        .mark_bar(color="#e6550d", cornerRadiusEnd=3)
        .encode(
            x=alt.X("Registros:Q", title="Registros Divergentes"),
            y=alt.Y(
                "Entidade_Curta:N",
                sort="-x",
                title=f"Top {len(div_top_entities_df)} Entidades",
            ),
            tooltip=[
                "Entidade",
                alt.Tooltip("Registros:Q", format=","),
                alt.Tooltip("Sites_Unicos:Q", format=",", title="Sites Físicos Únicos"),
                alt.Tooltip(
                    "Dist_Divisa_Med_km:Q",
                    format=".3f",
                    title="Dist. Mediana Divisa (km)",
                ),
            ],
        )
        .properties(
            width=340, height=250, title="Concentração por Entidade (Canais vs. Sites)"
        )
    )

    _sel_reg_codes = set(div_regime_select.value)
    _filtered_sites = filter_divergent_sites(
        div_sites_df,
        _sel_reg_codes,
        div_uf_select.value,
        div_fonte_select.value,
        div_entity_search.value,
        ALL_OPTION,
    )
    _total_filtered_sites = len(_filtered_sites)
    _total_filtered_rows = (
        int(_filtered_sites["Canais_Afetados"].sum())
        if _total_filtered_sites > 0
        else 0
    )
    _plot_sites = _filtered_sites.head(int(div_max_sites.value))
    _div_map = build_divergent_map(
        _plot_sites, DIVERGENT_REGIME_META, _sel_reg_codes, div_show_vectors.value
    )

    _sites_display = _filtered_sites[
        [
            "Regime_Desc",
            "Canais_Afetados",
            "UF_Decl",
            "Mun_Decl",
            "Real_UF",
            "Real_NM_MUN",
            "Vizinho",
            "Dist_Borda_km",
            "Shift_Centroid_km",
            "Entidade_Principal",
            "Fonte_Principal",
            "Servico_Principal",
            "Orig_Lat",
            "Orig_Lon",
            "Exemplo_Estacao",
            "Exemplo_Log_ID",
        ]
    ].round({"Dist_Borda_km": 3, "Shift_Centroid_km": 2, "Orig_Lat": 6, "Orig_Lon": 6})

    mo.vstack(
        [
            mo.md(
                f"""
    ## 4. Investigação Especial — Por que há `{_m["n_div"]:,}` estações com `IBGE_COORD_DIVERGENTE`?
    A mensagem `"IBGE: Município distinto."` é registrada sempre que o ponto `(Latitude, Longitude)` cai em um polígono municipal do IBGE diferente do `Código_Município` declarado (o nome do município do polígono, `NM_MUN`, fica em `Valor`). As coordenadas **não são substituídas** — são consideradas a melhor fonte de verdade. A auditoria geoespacial dessas coordenadas revela **quatro fatores estruturais**:

    1. **Amplificação por multicanalização (`{_m["n_sites"]:,}` sites físicos → `{_m["n_div"]:,}` linhas):** Como o agrupamento preserva uma linha por `Frequência`, estações repetidoras multicanal multiplicam o evento. `{_m["top_fonte"]}` concentra **{_m["top_fonte_pct"]:.1f}%** dos casos ({_m["top_fonte_n"]:,} linhas) e `{_m["top_serv_lbl"]}` responde por **{_m["top_serv_pct"]:.1f}%**. Apenas **{_m["n_heavy_sites"]:,} sites físicos** com ≥{_m["heavy_site_channels"]} frequências somam **{_m["n_heavy_rows"]:,} linhas ({_m["pct_heavy_rows"]:.1f}%)** — liderados em `{_m["top_uf"]}` (**{_m["top_uf_pct"]:.1f}%**, {_m["top_uf_n"]:,} linhas) por `{_m["top_ent"]}` (**{_m["top_ent_n"]:,} linhas** em apenas **{_m["top_ent_sites"]:,} coordenadas únicas**).
    2. **Efeito de Fronteira, Crista de Serra e Datum (`{_m["n_1km"]:,}` linhas = `{_m["pct_1km"]:.1f}%` a ≤ 1 km da divisa):** **{_m["n_100m"]:,} linhas** estão a **≤ 100 m** do limite municipal declarado (margem típica de deslocamento SAD69→SIRGAS2000 ~65 m, arredondamento de segundos DMS ~30 m e generalização da malha IBGE), e outras **{_m["n_1km"] - _m["n_100m"]:,}** entre **100 m e 1 km** (parques de antenas em topos de serra divisórios).
    3. **Conurbação e Plantas Limítrofes (`{_m["n_viz"]:,}` linhas = `{_m["pct_viz"]:.1f}%` em município vizinho):** A grande maioria ocorre entre municípios com fronteira comum (redes metropolitanas, mineradoras, usinas e fazendas intermunicipais). Como a regra não aplica *buffer* de tolerância na divisa, em **{_m["pct_shift_worse"]:.1f}%** dos registros a distância até o centroide do município declarado (mediana **{_m["med_shift"]:.2f} km**) é maior que a distância original até a divisa (mediana **{_m["med_borda"]:.2f} km**).
    4. **Erros Cadastrais Genuínos à Distância (`{_m["n_non_viz"]:,}` linhas = `{_m["pct_non_viz"]:.1f}%` em municípios não-vizinhos / outra UF):** Coordenadas de sede copiadas para licenças do interior ou erros de digitação em graus/minutos — casos em que a coordenada e o município declarado divergem por de dezenas a centenas de quilômetros.
    """
            ),
            _div_kpis,
            mo.ui.tabs(
                {
                    "📊 Regimes, Entidades & Pares": mo.vstack(
                        [
                            mo.hstack(
                                [_chart_regimes, _chart_div_ent], justify="start"
                            ),
                            mo.md("### Síntese Quantitativa por Regime Geoespacial"),
                            mo.ui.table(div_regime_summary_df, pagination=False),
                            mo.md(
                                "### Top Pares (Município Declarado → Polígono Real da Coordenada Original)"
                            ),
                            mo.ui.table(
                                div_top_pairs_df, pagination=True, page_size=12
                            ),
                        ]
                    ),
                    "🗺️ Mapa Coordenada → Centroide Declarado (Vetores)": mo.vstack(
                        [
                            mo.hstack(
                                [div_uf_select, div_fonte_select, div_entity_search],
                                widths="equal",
                            ),
                            mo.hstack(
                                [
                                    div_regime_select,
                                    mo.vstack([div_max_sites, div_show_vectors]),
                                ],
                                widths=[2, 1],
                            ),
                            mo.callout(
                                mo.md(
                                    f"Exibindo **{len(_plot_sites):,}** de **{_total_filtered_sites:,} sites físicos** filtrados "
                                    f"(representando **{_total_filtered_rows:,}** estações/canais). "
                                    f"Os círculos marcam a **coordenada original da torre** e as linhas tracejadas indicam o deslocamento até o **centroide IBGE**."
                                ),
                                kind="info",
                            ),
                            folium_iframe(_div_map, "560px"),
                        ]
                    ),
                    "🗼 Tabela de Sites Físicos Filtrados": mo.vstack(
                        [
                            mo.md(
                                f"Detalhamento dos **{_total_filtered_sites:,} sites físicos** (**{_total_filtered_rows:,} canais**) nos filtros atuais:"
                            ),
                            mo.ui.table(_sites_display, pagination=True, page_size=15),
                        ]
                    ),
                }
            ),
        ]
    )
    return


@app.function
@mo.persistent_cache
def compute_polygon_lookup(
    coords_frame: pd.DataFrame, shp_path_str: str
) -> pd.DataFrame:
    """Intersect unique (Latitude, Longitude) pairs against official IBGE municipal polygons."""
    regions = gpd.read_file(shp_path_str, columns=["CD_MUN", "NM_MUN", "SIGLA_UF"])
    lat_u = pd.to_numeric(coords_frame["Latitude"], errors="coerce")
    lon_u = pd.to_numeric(coords_frame["Longitude"], errors="coerce")
    valid_u = (
        (coords_frame["Latitude"] != "-1")
        & (coords_frame["Longitude"] != "-1")
        & lat_u.notna()
        & lon_u.notna()
        & (lat_u.abs() <= 90)
        & (lon_u.abs() <= 180)
    )
    pts = shapely.points(lon_u[valid_u].to_numpy(), lat_u[valid_u].to_numpy())
    tree = shapely.STRtree(regions.geometry.to_numpy())
    pt_idx, poly_idx = tree.query(pts, predicate="intersects")
    _, first_occ = np.unique(pt_idx, return_index=True)
    pt_idx = pt_idx[first_occ]
    poly_idx = poly_idx[first_occ]

    res = coords_frame.copy()
    res["poly_cd"] = ""
    res["poly_nm"] = ""
    res["poly_uf"] = ""
    valid_indices = res.index[valid_u].to_numpy()
    matched_idx = valid_indices[pt_idx]
    res.loc[matched_idx, "poly_cd"] = regions["CD_MUN"].iloc[poly_idx].to_numpy()
    res.loc[matched_idx, "poly_nm"] = regions["NM_MUN"].iloc[poly_idx].to_numpy()
    res.loc[matched_idx, "poly_uf"] = regions["SIGLA_UF"].iloc[poly_idx].to_numpy()
    return res


@app.function
def map_polygon_lookup(
    estacoes: pd.DataFrame, lookup: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Broadcast the unique-coordinate polygon lookup back to every station row -> (cd, nm, uf)."""
    idx = pd.MultiIndex.from_frame(estacoes[["Latitude", "Longitude"]].astype(str))
    lkp = lookup.set_index(["Latitude", "Longitude"])
    return tuple(
        lkp[c].reindex(idx).fillna("").to_numpy(dtype=object)
        for c in ("poly_cd", "poly_nm", "poly_uf")
    )


@app.function
def declared_ibge_attributes(
    codes: pd.Series, ibge_mun: pd.DataFrame
) -> dict[str, np.ndarray]:
    """Centroid lat/lon, UF and name of the declared IBGE municipality code for each row."""
    cod = codes.astype(str)
    key = ibge_mun["Código_Município"]
    return {
        "lat": cod.map(dict(zip(key, ibge_mun["Latitude"].astype(float)))).to_numpy(
            dtype="float64"
        ),
        "lon": cod.map(dict(zip(key, ibge_mun["Longitude"].astype(float)))).to_numpy(
            dtype="float64"
        ),
        "uf": cod.map(dict(zip(key, ibge_mun["UF"]))).fillna("").to_numpy(dtype=object),
        "name": cod.map(dict(zip(key, ibge_mun["Município"])))
        .fillna("")
        .to_numpy(dtype=object),
    }


@app.function
def detect_dirty_numeric_strings(
    df: pd.DataFrame, columns: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Rows whose engineering columns hold non-numeric text (besides '-1') -> (mask, 'col=value; ...' detail)."""
    mask = np.zeros(len(df), dtype=bool)
    detail = np.full(len(df), "", dtype=object)
    for col in columns:
        cats = df[col].cat.categories
        bad = set(cats[pd.to_numeric(cats, errors="coerce").isna() & (cats != "-1")])
        if not bad:
            continue
        m = df[col].isin(bad).to_numpy()
        mask |= m
        vals = df[col].astype(str).to_numpy()
        detail[m] = np.where(
            detail[m] == "",
            col + "=" + vals[m],
            detail[m] + "; " + col + "=" + vals[m],
        )
    return mask, detail


@app.function
def physical_outlier_mask(df: pd.DataFrame) -> np.ndarray:
    """Antenna/tower values outside physical ranges (azimuth, elevation, base height, antenna height)."""
    limits = {
        "Azimute_Antena": (0, 360),
        "Ângulo_Elevação_Antena": (-90, 90),
        "Cota_Base_Torre(m)": (-50, 3000),
        "Altura_Antena(m)": (0, 500),
    }
    mask = np.zeros(len(df), dtype=bool)
    for col, (lo, hi) in limits.items():
        v = cat_to_float(df[col])
        mask |= (~np.isnan(v)) & ((v < lo) | (v > hi))
    return mask


@app.function
def classify_primary_issue(
    estacoes: pd.DataFrame,
    lat: np.ndarray,
    lon: np.ndarray,
    poly_cd: np.ndarray,
    poly_uf: np.ndarray,
    declared_uf: np.ndarray,
    dirty_eng: np.ndarray,
    phys_outlier: np.ndarray,
) -> np.ndarray:
    """Assign one residual-issue code (keys of ISSUE_META) per row; later assignments have higher priority."""
    miss_coord = (estacoes["Latitude"] == "-1").to_numpy() | (
        estacoes["Longitude"] == "-1"
    ).to_numpy()
    has_coord = (~miss_coord) & (~np.isnan(lat)) & (~np.isnan(lon))
    outside = has_coord & (poly_cd == "")

    polar = outside & ((np.abs(lat) > 60) | (np.abs(lon) > 180))
    null_island = outside & ~polar & (np.abs(lat) < 1.0) & (np.abs(lon) < 1.0)
    swapped = (
        outside
        & ~polar
        & ~null_island
        & (lat < -34.0)
        & (lat >= -74.5)
        & (lon >= -34.0)
        & (lon <= 6.0)
    )
    sign_lon = outside & ~polar & ~null_island & ~swapped & (lon > 28.0) & (lon <= 75.0)
    sign_lat = (
        outside
        & ~polar
        & ~null_island
        & ~swapped
        & ~sign_lon
        & (lat > 5.35)
        & (lon >= -74.5)
        & (lon <= -34.0)
    )
    offshore = (
        outside
        & ~(polar | null_island | swapped | sign_lon | sign_lat)
        & (lon > -53.0)
        & (lon <= -28.5)
        & (lat >= -34.5)
        & (lat <= 6.0)
    )
    foreign = outside & ~(
        polar | null_island | swapped | sign_lon | sign_lat | offshore
    )

    cod = estacoes["Código_Município"].astype(str).to_numpy()
    uf = estacoes["UF"].astype(str).to_numpy()
    uf_trimmed = np.char.strip(uf.astype(str))
    inside = has_coord & (poly_cd != "")
    wrong_uf_poly = inside & (uf != "-1") & (uf_trimmed != poly_uf)
    wrong_mun_poly = inside & ~wrong_uf_poly & (cod != "-1") & (cod != poly_cd)
    uf_whitespace = (uf != uf_trimmed) | (
        (cod != "-1") & (declared_uf != "") & (uf_trimmed != declared_uf)
    )
    missing_mun_or_uf = (cod == "-1") | (uf == "-1")

    issue = np.full(len(estacoes), "00_OK_Consistente", dtype=object)
    for mask, code in (
        (missing_mun_or_uf, "11_Sem_Codigo_Municipio_ou_UF"),
        (uf_whitespace, "10_UF_Com_Espaco_ou_Divergente"),
        (phys_outlier, "09_Outlier_Fisico_Antena_Torre"),
        (dirty_eng, "08_Texto_Sujo_Coluna_Numerica"),
        (offshore, "07_Offshore_Maritimo_Ilhas"),
        (wrong_mun_poly, "06_Poligono_Municipio_Divergente"),
        (wrong_uf_poly, "05_Poligono_UF_Divergente"),
        (foreign, "04_Fora_Fronteira_Terrestre"),
        (sign_lat | sign_lon, "03_Sinal_Invertido_Hemisferio"),
        (swapped, "02_Lat_Lon_Invertidas"),
        (null_island | polar, "01_Null_Island_ou_Polar_Invalido"),
        (miss_coord, "00_Sem_Coordenadas_-1"),
    ):
        issue[mask] = code
    return issue


@app.function
def summarize_issues(
    primary_issue: np.ndarray, issue_meta: dict[str, tuple[str, str]]
) -> pd.DataFrame:
    """Record count and share for every issue code in `issue_meta` (zero-filled)."""
    counts = pd.Series(primary_issue).value_counts()
    n_total = max(len(primary_issue), 1)
    return pd.DataFrame(
        [
            {
                "Codigo": code,
                "Descricao": desc,
                "Registros": int(counts.get(code, 0)),
                "%_Total": round(int(counts.get(code, 0)) / n_total * 100, 4),
                "Plotavel_no_Mapa": code != "00_Sem_Coordenadas_-1",
            }
            for code, (_color, desc) in issue_meta.items()
        ]
    )


@app.function
def build_issue_points(
    estacoes: pd.DataFrame,
    idx: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    primary_issue: np.ndarray,
    poly_cd: np.ndarray,
    poly_nm: np.ndarray,
    poly_uf: np.ndarray,
    ibge_name: np.ndarray,
    dist_km: np.ndarray,
    dirty_detail: np.ndarray,
) -> pd.DataFrame:
    """Stations at row positions `idx` (as strings) enriched with diagnostics for map and table."""
    pts = estacoes.iloc[idx].astype(str).reset_index(drop=True)
    pts["Row_ID"] = idx
    pts["Diagnostico"] = primary_issue[idx]
    pts["Lat"] = lat[idx]
    pts["Lon"] = lon[idx]
    pts["Poly_CD"] = poly_cd[idx]
    pts["Poly_NM"] = poly_nm[idx]
    pts["Poly_UF"] = poly_uf[idx]
    pts["IBGE_Nome"] = ibge_name[idx]
    pts["Dist_km"] = dist_km[idx]
    pts["Dirty"] = dirty_detail[idx]
    pts["Log_ID"] = estacoes["Log"].to_numpy()[idx]
    return pts


@app.function
def issue_points_table(points: pd.DataFrame) -> pd.DataFrame:
    """Inspection table for the stations currently plotted on the Section 5 map."""
    return pd.DataFrame(
        {
            "Row_ID": points["Row_ID"],
            "Diagnostico": points["Diagnostico"],
            "Entidade": points["Entidade"],
            "Frequência_MHz": points["Frequência"],
            "Serviço": points["Serviço"],
            "Fonte": points["Fonte"],
            "UF_Decl": points["UF"],
            "Mun_Decl": points["Município"],
            "Cod_IBGE_Decl": points["Código_Município"],
            "Latitude": np.round(points["Lat"], 6),
            "Longitude": np.round(points["Lon"], 6),
            "UF_Poligono_Real": points["Poly_UF"],
            "Mun_Poligono_Real": points["Poly_NM"],
            "Dist_Centroide_km": np.round(points["Dist_km"], 1),
            "Detalhe_Engenharia": points["Dirty"],
            "Log_ID": points["Log_ID"],
        }
    )


@app.function
def station_tooltip(
    p: dict, color: str, desc: str, log_summary: str, max_log_chars: int = 180
) -> str:
    """HTML tooltip for one station (a record from `build_issue_points(...).to_dict('records')`)."""
    esc = lambda v: html.escape(str(v))
    log_txt = esc(log_summary)
    if len(log_txt) > max_log_chars:
        log_txt = log_txt[: max_log_chars - 3] + "..."
    mun_decl = p["Município"]
    if mun_decl == "-1" and p["IBGE_Nome"]:
        mun_decl = f"-1 (IBGE p/ código: {p['IBGE_Nome']})"
    poly_info = (
        f"{p['Poly_NM']} / {p['Poly_UF']} ({p['Poly_CD']})"
        if p["Poly_CD"]
        else "FORA DA MALHA TERRESTRE IBGE"
    )
    dirty_line = (
        f"<br><b style='color:#b15928'>Texto Sujo Eng.:</b> {esc(p['Dirty'])}"
        if p["Dirty"]
        else ""
    )
    return (
        f"<div style='font-family:sans-serif;font-size:12px;min-width:290px;max-width:380px;'>"
        f"<div style='background:{color};color:#fff;padding:4px 7px;border-radius:4px;font-weight:bold;margin-bottom:5px;'>"
        f"{esc(desc)}</div>"
        f"<b>Linha #</b>{p['Row_ID']} | <b>Estação:</b> {esc(p['Estação'])} | <b>Fistel:</b> {esc(p['Fistel'])}<br>"
        f"<b>Entidade:</b> {esc(p['Entidade'])}<br>"
        f"<b>Freq:</b> {esc(p['Frequência'])} MHz | <b>BW:</b> {esc(p['Largura_Emissão(kHz)'])} kHz | "
        f"<b>Serviço:</b> {esc(p['Serviço'])} ({esc(p['Classe'])})<br>"
        f"<b>Fonte:</b> {esc(p['Fonte'])} | <b>Status:</b> {esc(p['Status'])} | <b>Mult:</b> {esc(p['Multiplicidade'])}<br>"
        f"<hr style='margin:4px 0'>"
        f"<b>Coord:</b> ({p['Lat']:.6f}, {p['Lon']:.6f})<br>"
        f"<b>Declarado:</b> {esc(mun_decl)} / '{esc(p['UF'])}' (Cód: {esc(p['Código_Município'])})<br>"
        f"<b>Polígono Real:</b> {esc(poly_info)} | <b>Dist. Centroide:</b> {format_distance(p['Dist_km'], meters_below_1km=False, decimals=1)}"
        f"{dirty_line}"
        f"<hr style='margin:4px 0'>"
        f"<b>Log [{p['Log_ID']}]:</b> {log_txt}"
        f"</div>"
    )


@app.function
def build_issue_map(
    points: pd.DataFrame,
    issue_meta: dict[str, tuple[str, str]],
    selected_codes: set[str],
    cluster: bool,
    log_summaries: dict[int, str],
) -> folium.Map:
    """Folium map of residual-issue stations, one layer per issue (or a single MarkerCluster)."""
    fmap = base_folium_map([-14.235, -51.925], 4)
    if cluster:
        cluster_layer = plugins.MarkerCluster(name="Estações Filtradas").add_to(fmap)
        layer_for = lambda code: cluster_layer
    else:
        layers = {
            code: folium.FeatureGroup(name=issue_meta[code][1], show=True).add_to(fmap)
            for code in sorted(selected_codes)
        }
        layer_for = lambda code: layers[code]
    for p in points.to_dict("records"):
        color, desc = issue_meta[p["Diagnostico"]]
        folium.CircleMarker(
            location=[float(p["Lat"]), float(p["Lon"])],
            radius=5,
            color=color,
            weight=1.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            tooltip=folium.Tooltip(
                station_tooltip(
                    p, color, desc, log_summaries.get(int(p["Log_ID"]), "")
                ),
                sticky=True,
            ),
        ).add_to(layer_for(p["Diagnostico"]))
    folium.LayerControl(collapsed=True).add_to(fmap)
    return fmap


@app.cell
def _(
    ENG_NUMERIC_COLS,
    ISSUE_META,
    SHP_PATH,
    estacoes,
    ibge_mun,
    lat_num,
    lon_num,
):
    coord_poly_lookup = compute_polygon_lookup(
        estacoes[["Latitude", "Longitude"]]
        .astype(str)
        .drop_duplicates()
        .reset_index(drop=True),
        SHP_PATH,
    )
    poly_cd_arr, poly_nm_arr, poly_uf_arr = map_polygon_lookup(
        estacoes, coord_poly_lookup
    )

    _decl = declared_ibge_attributes(estacoes["Código_Município"], ibge_mun)
    ibge_mun_name_arr = _decl["name"]
    ibge_mun_uf_arr = _decl["uf"]

    # Distance from each station to the centroid of its declared IBGE municipality
    _valid_hav = valid_wgs84_mask(lat_num, lon_num) & ~np.isnan(_decl["lat"])
    dist_ibge_km = np.full(len(estacoes), np.nan, dtype="float64")
    dist_ibge_km[_valid_hav] = haversine_km(
        lat_num[_valid_hav],
        lon_num[_valid_hav],
        _decl["lat"][_valid_hav],
        _decl["lon"][_valid_hav],
    )

    dirty_eng_mask, dirty_eng_detail = detect_dirty_numeric_strings(
        estacoes, ENG_NUMERIC_COLS
    )
    phys_outlier_arr = physical_outlier_mask(estacoes)

    primary_issue = classify_primary_issue(
        estacoes,
        lat_num,
        lon_num,
        poly_cd_arr,
        poly_uf_arr,
        ibge_mun_uf_arr,
        dirty_eng_mask,
        phys_outlier_arr,
    )
    issue_summary_df = summarize_issues(primary_issue, ISSUE_META)
    return (
        dirty_eng_detail,
        dist_ibge_km,
        ibge_mun_name_arr,
        issue_summary_df,
        poly_cd_arr,
        poly_nm_arr,
        poly_uf_arr,
        primary_issue,
    )


@app.cell(hide_code=True)
def _(ALL_OPTION, DEFAULT_MAP_ISSUES, estacoes, issue_summary_df):
    _issue_options = {
        f"{row.Descricao} [{row.Registros:,}]": row.Codigo
        for row in issue_summary_df.itertuples()
        if row.Registros > 0 and row.Plotavel_no_Mapa
    }
    map_issue_select = mo.ui.multiselect(
        options=_issue_options,
        value=[lbl for lbl, cod in _issue_options.items() if cod in DEFAULT_MAP_ISSUES],
        label="Categorias para Plotar no Mapa:",
    )
    map_uf_select = mo.ui.dropdown(
        options=[ALL_OPTION] + sorted(estacoes["UF"].astype(str).unique().tolist()),
        value=ALL_OPTION,
        label="Filtrar UF:",
    )
    map_fonte_select = mo.ui.dropdown(
        options=[ALL_OPTION] + sorted(estacoes["Fonte"].astype(str).unique().tolist()),
        value=ALL_OPTION,
        label="Filtrar Fonte:",
    )
    map_max_points = mo.ui.slider(
        start=200, stop=100000, step=200, value=50000, label="Máx. Pontos no Mapa:"
    )
    map_cluster_toggle = mo.ui.checkbox(
        value=False, label="Agrupar marcadores (MarkerCluster)"
    )
    map_entity_search = mo.ui.text(
        value="",
        placeholder="Ex: PETROBRAS, PRYMUS, CLARO...",
        label="Buscar Entidade:",
    )
    return (
        map_cluster_toggle,
        map_entity_search,
        map_fonte_select,
        map_issue_select,
        map_max_points,
        map_uf_select,
    )


@app.cell(hide_code=True)
def _(
    ALL_OPTION,
    ISSUE_META,
    dirty_eng_detail,
    dist_ibge_km,
    estacoes,
    ibge_mun_name_arr,
    issue_summary_df,
    lat_num,
    log_templates,
    lon_num,
    map_cluster_toggle,
    map_entity_search,
    map_fonte_select,
    map_issue_select,
    map_max_points,
    map_uf_select,
    poly_cd_arr,
    poly_nm_arr,
    poly_uf_arr,
    primary_issue,
):
    _selected_codes = set(map_issue_select.value)
    _mask = np.isin(primary_issue, list(_selected_codes)) & valid_wgs84_mask(
        lat_num, lon_num
    )
    _mask &= selection_mask(estacoes["UF"], map_uf_select.value, ALL_OPTION)
    _mask &= selection_mask(estacoes["Fonte"], map_fonte_select.value, ALL_OPTION)
    _mask &= text_search_mask(estacoes, ["Entidade"], map_entity_search.value)

    _matched_idx = np.flatnonzero(_mask)
    _total_matched = len(_matched_idx)
    # Sorting by issue code keeps the most severe anomalies when sampling down to map_max_points
    if _total_matched > map_max_points.value:
        _order = np.argsort(primary_issue[_matched_idx].astype(str))
        _matched_idx = _matched_idx[_order[: int(map_max_points.value)]]

    _points = build_issue_points(
        estacoes,
        _matched_idx,
        lat_num,
        lon_num,
        primary_issue,
        poly_cd_arr,
        poly_nm_arr,
        poly_uf_arr,
        ibge_mun_name_arr,
        dist_ibge_km,
        dirty_eng_detail,
    )
    _fmap = build_issue_map(
        _points,
        ISSUE_META,
        _selected_codes,
        map_cluster_toggle.value,
        dict(zip(log_templates["Log"], log_templates["Resumo_Log"])),
    )
    _table_out = issue_points_table(_points)

    _issue_counts = dict(zip(issue_summary_df["Codigo"], issue_summary_df["Registros"]))
    _count = lambda *codes: sum(int(_issue_counts.get(c, 0)) for c in codes)
    _n_ok = _count("00_OK_Consistente")
    _pct_ok = _n_ok / max(len(estacoes), 1) * 100.0
    _n_offshore = _count("07_Offshore_Maritimo_Ilhas")
    _n_poly_div_res = _count(
        "05_Poligono_UF_Divergente", "06_Poligono_Municipio_Divergente"
    )
    _n_crit_coord = _count(
        "01_Null_Island_ou_Polar_Invalido",
        "02_Lat_Lon_Invertidas",
        "03_Sinal_Invertido_Hemisferio",
    )
    _n_foreign = _count("04_Fora_Fronteira_Terrestre")
    _n_dirty_eng = _count("08_Texto_Sujo_Coluna_Numerica")
    _n_phys_out = _count("09_Outlier_Fisico_Antena_Torre")
    _n_uf_dirty = _count(
        "10_UF_Com_Espaco_ou_Divergente", "11_Sem_Codigo_Municipio_ou_UF"
    )
    _n_no_coord = _count("00_Sem_Coordenadas_-1")

    _res_kpis = kpi_row(
        [
            (
                "Consistentes c/ Malha IBGE",
                f"{_n_ok:,}",
                f"{_pct_ok:.2f}% do total consolidado",
            ),
            (
                "Offshore / Costeiro",
                f"{_n_offshore:,}",
                "Plataformas marítimas, pré-sal e ilhas",
            ),
            (
                "Fronteira / Críticos no Mar",
                f"{_n_foreign + _n_crit_coord:,}",
                f"{_n_foreign:,} fronteira/exterior + {_n_crit_coord:,} inversões/nulos",
            ),
            (
                "Resíduos Eng. / Sem Coord.",
                f"{_n_dirty_eng + _n_phys_out + _n_no_coord:,}",
                f"{_n_dirty_eng:,} texto sujo | {_n_phys_out:,} outliers | {_n_no_coord:,} sem coord.",
            ),
        ]
    )

    _issue_chart = (
        alt.Chart(
            issue_summary_df[
                (issue_summary_df["Codigo"] != "00_OK_Consistente")
                & (issue_summary_df["Registros"] > 0)
            ]
        )
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X(
                "Registros:Q",
                title="Registros Residualmente Sinalizados (escala symlog)",
                scale=alt.Scale(type="symlog"),
            ),
            y=alt.Y("Descricao:N", sort="-x", title=None),
            color=alt.Color(
                "Codigo:N", scale=meta_color_scale(ISSUE_META), legend=None
            ),
            tooltip=[
                "Codigo",
                "Descricao",
                alt.Tooltip("Registros:Q", format=","),
                "%_Total",
            ],
        )
        .properties(
            height=300,
            title="Panorama Atual de Registros Fora da Malha Terrestre ou com Alertas de Engenharia",
        )
    )

    mo.vstack(
        [
            mo.md(
                f"""
    ## 5. Auditoria de Qualidade Residual (Pós-Processamento) & Mapa Interativo Folium
    Após todas as etapas de padronização e validação espacial em `estacoes.parquet`, **{_n_ok:,} estações ({_pct_ok:.2f}%)** encontram-se 100% consistentes com os polígonos municipais IBGE e sem alertas técnicos:

    - **Divergência Residual Polígono × Município/UF em Terra:** **{_n_poly_div_res:,} registros** (e **{_n_uf_dirty:,}** com `UF`/código inconsistente em terra).
    - **Operações Marítimas & Fronteiriças:** **{_n_offshore:,} registros** correspondem a plataformas *offshore*/costeiras legítimas no Atlântico, **{_n_foreign:,}** situam-se em faixa de fronteira internacional/exterior, **{_n_crit_coord:,}** apresentam coordenadas críticas residuais fora de terra (ex.: `Lat ↔ Lon` cujo par invertido também cai no mar/ilha) e **{_n_no_coord:,}** permanecem sem coordenadas (`"-1"`).
    - **Atributos de Engenharia (Radiodifusão):** **{_n_dirty_eng:,} registros** contêm strings não normalizadas em colunas numéricas (vírgula decimal ou unidade textual) e **{_n_phys_out:,}** exibem valores físicos extremos de antena/torre.
    """
            ),
            _res_kpis,
            mo.ui.tabs(
                {
                    "🗺️ Mapa Folium Interativo (Tooltips)": mo.vstack(
                        [
                            mo.hstack(
                                [map_uf_select, map_fonte_select, map_entity_search],
                                widths="equal",
                            ),
                            mo.hstack(
                                [
                                    map_issue_select,
                                    mo.vstack([map_max_points, map_cluster_toggle]),
                                ],
                                widths=[2, 1],
                            ),
                            mo.callout(
                                mo.md(
                                    f"Exibindo **{len(_matched_idx):,}** de **{_total_matched:,}** registros filtrados no mapa. "
                                    f"Passe o mouse sobre qualquer ponto para inspecionar o registro completo, polígono real IBGE, distância ao centroide e `Log`."
                                ),
                                kind="info",
                            ),
                            folium_iframe(_fmap, "580px"),
                        ]
                    ),
                    "📊 Resumo Quantitativo de Anomalias": mo.vstack(
                        [
                            _issue_chart,
                            mo.ui.table(issue_summary_df, pagination=False),
                        ]
                    ),
                    "🧾 Registros Problemáticos Filtrados": mo.vstack(
                        [
                            mo.md(
                                f"Tabela detalhada dos **{len(_table_out):,}** registros atualmente exibidos no filtro do mapa:"
                            ),
                            mo.ui.table(_table_out, pagination=True, page_size=15),
                        ]
                    ),
                }
            ),
        ]
    )
    return


@app.cell
def _():
    mo.md(r"""
    ## 6. Explorador Geral de Estações (Todos os Pontos)

    Mapa de **toda a base**, não apenas dos resíduos problemáticos. Como a base tem mais de um milhão de registros, filtre por
    **Serviço**, **faixa de Frequência**, **UF**, **Município declarado** e/ou **busca livre** (aplicada a *todas* as colunas;
    múltiplos termos separados por espaço devem todos aparecer, em qualquer coluna). Cada ponto é colorido pelo seu diagnóstico
    da Seção 5 — `00_OK_Consistente` representa as estações normais.

    Nenhum ponto é descartado: o resultado filtrado é dividido em **páginas** de *Pontos por página* registros, ordenadas por
    UF/Município, e o seletor **Página** percorre todo o conjunto (mudar qualquer filtro volta à página 1).
    """)
    return


@app.function(hide_code=True)
def any_column_search_mask(frame: pd.DataFrame, query: str) -> np.ndarray:
    """Case-insensitive search over every column; each whitespace-separated term must match some column."""
    mask = np.ones(len(frame), dtype=bool)
    for term in query.lower().split():
        term_mask = np.zeros(len(frame), dtype=bool)
        for col in frame.columns:
            values = frame[col]
            if isinstance(values.dtype, pd.CategoricalDtype):
                # Match on the (few) categories, then map back through the codes
                hits = (
                    values.cat.categories.astype(str)
                    .str.lower()
                    .str.contains(term, regex=False)
                )
                codes = values.cat.codes.to_numpy()
                term_mask |= (codes >= 0) & np.asarray(hits)[np.clip(codes, 0, None)]
            else:
                term_mask |= (
                    values.astype(str)
                    .str.lower()
                    .str.contains(term, regex=False)
                    .to_numpy()
                )
        mask &= term_mask
    return mask


@app.cell(hide_code=True)
def _(ALL_OPTION, SERVICO_NOMES, estacoes):
    _serv_counts = estacoes["Serviço"].astype(str).value_counts()
    all_serv_select = mo.ui.multiselect(
        options={
            f"{SERVICO_NOMES.get(code, code)} [{n:,}]": code
            for code, n in _serv_counts.items()
        },
        value=[],
        label="Serviço (vazio = todos):",
    )
    all_freq_min = mo.ui.number(
        start=0, stop=100000, step=0.001, value=0, label="Freq. mín. (MHz):"
    )
    all_freq_max = mo.ui.number(
        start=0, stop=100000, step=0.001, value=100000, label="Freq. máx. (MHz):"
    )
    all_uf_select = mo.ui.dropdown(
        options=[ALL_OPTION] + sorted(estacoes["UF"].astype(str).unique().tolist()),
        value=ALL_OPTION,
        label="UF:",
    )
    all_text_search = mo.ui.text(
        value="",
        placeholder="Ex: CLARO 3500, 50450314308, PETROBRAS...",
        label="Busca livre (todas as colunas):",
        full_width=True,
    )
    all_max_points = mo.ui.slider(
        start=100, stop=20000, step=100, value=3000, label="Pontos por página:"
    )
    all_cluster_toggle = mo.ui.checkbox(
        value=True, label="Agrupar marcadores (MarkerCluster)"
    )
    return (
        all_cluster_toggle,
        all_freq_max,
        all_freq_min,
        all_max_points,
        all_serv_select,
        all_text_search,
        all_uf_select,
    )


@app.cell(hide_code=True)
def _(ALL_OPTION, all_uf_select, estacoes):
    _pairs = estacoes[["Município", "UF"]].astype(str).drop_duplicates()
    if all_uf_select.value != ALL_OPTION:
        _pairs = _pairs[_pairs["UF"] == all_uf_select.value]
    _pairs = _pairs.sort_values(["Município", "UF"])
    all_mun_select = mo.ui.dropdown(
        options={ALL_OPTION: None}
        | {f"{m} / {u}": (m, u) for m, u in zip(_pairs["Município"], _pairs["UF"])},
        value=ALL_OPTION,
        searchable=True,
        label="Município declarado:",
    )
    return (all_mun_select,)


@app.cell
def _(
    ALL_OPTION,
    all_freq_max,
    all_freq_min,
    all_mun_select,
    all_serv_select,
    all_text_search,
    all_uf_select,
    estacoes,
    freq_mhz,
    lat_num,
    lon_num,
):
    _mask = valid_wgs84_mask(lat_num, lon_num)
    if all_serv_select.value:
        _mask &= np.isin(
            estacoes["Serviço"].astype(str).to_numpy(), all_serv_select.value
        )
    _fmin, _fmax = all_freq_min.value, all_freq_max.value
    if _fmin > 0 or _fmax < 100000:
        _mask &= (freq_mhz >= _fmin) & (freq_mhz <= _fmax)
    _mask &= selection_mask(estacoes["UF"], all_uf_select.value, ALL_OPTION)
    if all_mun_select.value is not None:
        _mun, _uf = all_mun_select.value
        _mask &= (estacoes["Município"].astype(str) == _mun).to_numpy()
        _mask &= (estacoes["UF"].astype(str) == _uf).to_numpy()
    if all_text_search.value.strip():
        _mask &= any_column_search_mask(estacoes, all_text_search.value)

    _idx = np.flatnonzero(_mask)
    # Deterministic UF/Município order so every filtered point lands on exactly one, geographically coherent page
    _order = np.lexsort(
        (
            _idx,
            estacoes["Município"].astype(str).to_numpy()[_idx],
            estacoes["UF"].astype(str).to_numpy()[_idx],
        )
    )
    all_matched_idx = _idx[_order]

    return (all_matched_idx,)


@app.cell(hide_code=True)
def _(all_matched_idx, all_max_points):
    all_n_pages = max(1, -(-len(all_matched_idx) // int(all_max_points.value)))
    all_page = mo.ui.number(
        start=1,
        stop=all_n_pages,
        step=1,
        value=1,
        label=f"Página (de {all_n_pages:,}):",
    )

    return all_n_pages, all_page


@app.cell(hide_code=True)
def _(
    ISSUE_META,
    all_cluster_toggle,
    all_freq_max,
    all_freq_min,
    all_matched_idx,
    all_max_points,
    all_mun_select,
    all_n_pages,
    all_page,
    all_serv_select,
    all_text_search,
    all_uf_select,
    dirty_eng_detail,
    dist_ibge_km,
    estacoes,
    ibge_mun_name_arr,
    lat_num,
    log_templates,
    lon_num,
    poly_cd_arr,
    poly_nm_arr,
    poly_uf_arr,
    primary_issue,
):
    _size = int(all_max_points.value)
    _page = min(max(int(all_page.value or 1), 1), all_n_pages)
    _start = (_page - 1) * _size
    _page_idx = np.sort(all_matched_idx[_start : _start + _size])
    _total_matched = len(all_matched_idx)

    _points = build_issue_points(
        estacoes,
        _page_idx,
        lat_num,
        lon_num,
        primary_issue,
        poly_cd_arr,
        poly_nm_arr,
        poly_uf_arr,
        ibge_mun_name_arr,
        dist_ibge_km,
        dirty_eng_detail,
    )
    _codes = set(_points["Diagnostico"])
    _fmap = build_issue_map(
        _points,
        ISSUE_META,
        _codes,
        all_cluster_toggle.value,
        dict(zip(log_templates["Log"], log_templates["Resumo_Log"])),
    )
    _n_ok = int((_points["Diagnostico"] == "00_OK_Consistente").sum())
    _ufs = estacoes["UF"].astype(str).to_numpy()[_page_idx]
    _uf_span = f" (UF {_ufs.min()}–{_ufs.max()})" if len(_ufs) else ""

    mo.vstack(
        [
            mo.hstack(
                [all_serv_select, all_uf_select, all_mun_select], widths=[2, 1, 1]
            ),
            mo.hstack(
                [
                    all_freq_min,
                    all_freq_max,
                    all_max_points,
                    all_page,
                    all_cluster_toggle,
                ]
            ),
            all_text_search,
            mo.callout(
                mo.md(
                    f"**{_total_matched:,}** registros atendem aos filtros, em **{all_n_pages:,}** página(s) ordenadas por UF/Município. "
                    f"Página **{_page:,}**: registros **{min(_start + 1, _total_matched):,}–{_start + len(_page_idx):,}**{_uf_span} — "
                    f"{_n_ok:,} consistentes e {len(_points) - _n_ok:,} com algum diagnóstico. "
                    f"Nenhum ponto é omitido: avance as páginas (ou aumente *Pontos por página*) para percorrer todo o resultado."
                ),
                kind="info",
            ),
            mo.ui.tabs(
                {
                    "🗺️ Mapa (página atual)": folium_iframe(_fmap, "620px"),
                    "🧾 Tabela (página atual)": mo.ui.table(
                        issue_points_table(_points), pagination=True, page_size=15
                    ),
                }
            ),
        ]
    )

    return


if __name__ == "__main__":
    app.run()
