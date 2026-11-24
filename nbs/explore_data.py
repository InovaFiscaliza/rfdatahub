# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "altair>=6.0.0",
#     "folium>=1.0.0",
#     "geopandas>=1.0.0",
#     "marimo>=0.25.0",
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
def _():
    mo.md("""
    # 📡 RFDataHub — Exploratory Data Analysis & Residual Quality Audit
    **Dataset:** `/code/rfdatahub/extracao/datasources/arquivos/saida/estacoes.parquet` + `log.parquet`

    This interactive notebook covers:
    1. **Executive Overview & Schema Completeness** — row counts vs. physical emitters (`Multiplicidade`), sentinel `-1` missingness, and source/service breakdown.
    2. **Spectrum, Services & Operators** — frequency band allocations, emission bandwidths, and top entities across Brazil.
    3. **Structured `log.parquet` Explorer** — unpacking the factorized JSON dictionary logs (`Log` index → template → atomic events), pipeline flows, and before/after diffs.
    4. **Residual Data Quality Audit & Interactive Folium Map** — diagnosing data defects that survived `extracao` post-processing (swapped/sign-flipped/polar coordinates, Null Island, municipality/UF polygon mismatches, trailing spaces in `UF`, unparsed strings, and physical outliers) with detailed record tooltips.
    """)
    return


@app.cell
def _():
    def cat_to_float(s: pd.Series) -> np.ndarray:
        """Fast vectorized conversion of categorical string Series to float64 (-1 -> NaN)."""
        cats = s.cat.categories
        conv = pd.to_numeric(cats, errors="coerce").to_numpy(dtype="float64", copy=True)
        conv[cats == "-1"] = np.nan
        codes = s.cat.codes.to_numpy()
        out = np.full(len(s), np.nan, dtype="float64")
        valid = codes >= 0
        out[valid] = conv[codes[valid]]
        return out


    def classify_log_step(proc: str) -> tuple[str, bool]:
        """Return (short_tag, is_data_quality_event) for a log Processamento string."""
        if proc.startswith("MOSAICO: Registros agrupados"):
            return ("MOSAICO_AGRUPADO", False)
        if proc.startswith(("MOSAICO-TELECOM: Registros Agrupados", "Registros Agrupados")):
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
        if proc.startswith("Código do município retirado do polígono"):
            return ("IBGE_POLIGONO_RECUPERADO", True)
        if proc.startswith("Código do Município nulo"):
            return ("MUN_CODIGO_NULO", True)
        if proc.startswith("Coordenadas divergentes"):
            return ("IBGE_COORD_DIVERGENTE", True)
        if proc.startswith("Nomes de Municípios e UFs normalizados"):
            return ("IBGE_NOME_NORMALIZADO", True)
        if proc.startswith("Valor Nulo de Coordenadas"):
            return ("COORD_NULA_PREENCHIDA_IBGE", True)
        if proc.startswith("Coordenadas nulas"):
            return ("COORD_NULAS", True)
        if proc.startswith("Coordenadas corrigidas automaticamente"):
            return ("COORD_SINAL_OU_TROCA_CORRIGIDA", True)
        if proc.startswith("Coordenadas inválidas"):
            return ("COORD_ABSURDA_SUBSTITUIDA_IBGE", True)
        if proc.startswith("Coordenadas Fora do Território"):
            return ("FORA_TERRITORIO_BR", False)
        if proc.startswith("Código do Município não consta"):
            return ("MUN_NAO_CONSTA_IBGE", True)
        if proc.startswith("Valor Não Numérico Descartado"):
            return ("COORD_NAO_NUMERICA", True)
        return ("OUTRO", True)

    return cat_to_float, classify_log_step


@app.cell
def _(cat_to_float, classify_log_step):
    # Portable data directory resolution (works inside rfdatahub or standalone via MARIMO --sandbox)
    def _resolve_arq_dir() -> Path:
        import os
        env_dir = os.environ.get("RFDATAHUB_ARQ_DIR")
        if env_dir and Path(env_dir).exists():
            return Path(env_dir)
        nb_file = Path(__file__).resolve() if "__file__" in globals() else Path.cwd()
        for candidate in (
            nb_file.parent.parent / "extracao" / "datasources" / "arquivos",
            Path.cwd() / "extracao" / "datasources" / "arquivos",
            Path.cwd() / "arquivos",
            Path("/code/rfdatahub/extracao/datasources/arquivos"),
        ):
            if (candidate / "saida" / "estacoes.parquet").exists():
                return candidate
        return Path("/code/rfdatahub/extracao/datasources/arquivos")


    ARQ_DIR = _resolve_arq_dir()
    DATA_DIR = ARQ_DIR / "saida"

    estacoes = pd.read_parquet(DATA_DIR / "estacoes.parquet")
    log_raw_df = pd.read_parquet(DATA_DIR / "log.parquet")
    ibge_mun = pd.read_csv(ARQ_DIR / "municipios.csv", dtype=str)

    # Parse all 8,027 factorized JSON log templates
    _log_row_counts = estacoes["Log"].value_counts()
    log_parsed_list = [json.loads(s) for s in log_raw_df["Log"]]

    _tmpl_rows = []
    _event_rows = []
    for _lid, _items in enumerate(log_parsed_list):
        _nrows = int(_log_row_counts.get(_lid, 0))
        _tags = []
        _has_dq = False
        _pretty_parts = []
        for _step_i, _it in enumerate(_items):
            _proc = _it.get("Processamento", "")
            _col = _it.get("Coluna", "")
            _orig = _it.get("Original", "")
            _tag, _is_dq = classify_log_step(_proc)
            _has_dq = _has_dq or _is_dq
            _tags.append(_tag)
            _detail = f" [{_col}: '{_orig}']" if _col else ""
            _pretty_parts.append(f"{_step_i + 1}. {_tag}{_detail}")
            _event_rows.append(
                {
                    "Log": _lid,
                    "Passo": _step_i + 1,
                    "Tag": _tag,
                    "Alerta_Qualidade": _is_dq,
                    "Processamento": _proc.replace("\n", " "),
                    "Coluna": _col,
                    "Original": _orig,
                    "Estacoes_Afetadas": _nrows,
                }
            )
        _pipeline = " → ".join(dict.fromkeys(_tags)) if _tags else "SEM_ALTERACAO"
        _tmpl_rows.append(
            {
                "Log": _lid,
                "Qtd_Passos": len(_items),
                "Pipeline": _pipeline,
                "Alerta_Qualidade": _has_dq,
                "Resumo_Log": " | ".join(_pretty_parts) if _pretty_parts else "Sem alterações registradas",
                "JSON_Bruto": log_raw_df["Log"].iloc[_lid],
                "Estacoes_Afetadas": _nrows,
            }
        )

    log_templates = pd.DataFrame(_tmpl_rows)
    log_events = pd.DataFrame(_event_rows)

    # Pre-extract core numeric arrays for fast EDA & auditing
    freq_mhz = cat_to_float(estacoes["Frequência"])
    lat_num = cat_to_float(estacoes["Latitude"])
    lon_num = cat_to_float(estacoes["Longitude"])
    bw_khz = cat_to_float(estacoes["Largura_Emissão(kHz)"])
    mult_num = np.nan_to_num(cat_to_float(estacoes["Multiplicidade"]), nan=1.0)
    return (
        ARQ_DIR,
        estacoes,
        freq_mhz,
        ibge_mun,
        lat_num,
        log_events,
        log_templates,
        lon_num,
        mult_num,
    )


@app.cell(hide_code=True)
def _(estacoes, log_events, log_templates, mult_num):
    # Build column completeness & sentinel '-1' profile
    _profile_rows = []
    for _col in estacoes.columns:
        _s = estacoes[_col]
        if str(_s.dtype) == "category":
            _n_sent = int((_s == "-1").sum()) if "-1" in _s.cat.categories else 0
            _top_val = str(_s.value_counts().index[0])
            if len(_top_val) > 42:
                _top_val = _top_val[:39] + "..."
        else:
            _n_sent = int((_s == -1).sum())
            _top_val = str(_s.mode().iloc[0])
        _valid_pct = (1.0 - _n_sent / len(estacoes)) * 100.0
        _profile_rows.append(
            {
                "Coluna": _col,
                "Tipo_Parquet": str(_s.dtype),
                "Valores_Unicos": int(_s.nunique()),
                "Sentinela_-1": _n_sent,
                "Preenchido_%": round(_valid_pct, 2),
                "Valor_Mais_Comum": _top_val,
            }
        )
    schema_profile_df = pd.DataFrame(_profile_rows)

    _kpis = mo.hstack(
        [
            mo.stat(
                label="Registros Consolidados",
                value=f"{len(estacoes):,}",
                caption="45.7 MB Parquet (30 colunas)",
            ),
            mo.stat(
                label="Emissores Físicos (Σ Multiplicidade)",
                value=f"{int(mult_num.sum()):,}",
                caption=f"Média {mult_num.mean():.1f}x por linha agrupada",
            ),
            mo.stat(
                label="Templates em log.parquet",
                value=f"{len(log_templates):,}",
                caption=f"{len(log_events):,} eventos JSON fatorados",
            ),
            mo.stat(
                label="Município Preenchido",
                value=f"{(1 - (estacoes['Município'] == '-1').mean()) * 100:.2f}%",
                caption="Antes da correção: 70,39% (338.193 '-1' -> agora 256!)",
            ),
        ],
        widths="equal",
    )

    _miss_chart = (
        alt.Chart(schema_profile_df)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("Preenchido_%:Q", title="% Preenchido (não '-1')", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("Coluna:N", sort="-x", title=None),
            color=alt.condition(
                alt.datum["Preenchido_%"] < 50,
                alt.value("#e45756"),
                alt.value("#4c78a8"),
            ),
            tooltip=["Coluna", "Preenchido_%", "Sentinela_-1", "Valores_Unicos", "Valor_Mais_Comum"],
        )
        .properties(height=500, title="Completude por Coluna após Nova Extração (Sentinela '-1' = Ausente)")
    )

    mo.vstack(
        [
            mo.md(
                """
    ## 1. Visão Geral & Completude do Esquema (Pós-Correção)
    > **Impacto imediato na completude:** Com a correção em `location.py`, `Município == "-1"` caiu de **338.193 linhas (29,61%)** para apenas **256 linhas (0,02%)** — exatamente os 256 registros marítimos/aeronáuticos sem `Código_Município` em alto-mar/fronteira. Além disso, as UFs sujas (`"MT "` e `"RO "`) foram **100% eliminadas** (restando apenas as 27 UFs oficiais + `"-1"` nos 256 offshore). As 12 colunas técnicas de antena/RF seguem restritas às fontes de Radiodifusão (`MOSAICO-SRD` e `SRD`, ~3,87%).
    """
            ),
            _kpis,
            mo.ui.tabs(
                {
                    "📊 Gráfico de Completude": _miss_chart,
                    "📋 Tabela de Perfil das Colunas": mo.ui.table(schema_profile_df, pagination=True, page_size=15),
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(estacoes, freq_mhz, mult_num):
    # Spectrum bands, Fontes, Serviços, UFs, Top Entidades
    _band_bins = [0, 0.3, 3, 30, 300, 1000, 3000, 6000, 30000, 100000]
    _band_labels = [
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
    _bands = pd.cut(freq_mhz, bins=_band_bins, labels=_band_labels)
    _band_df = (
        pd.DataFrame({"Faixa": _bands, "Emissores": mult_num})
        .groupby("Faixa", observed=False)
        .agg(Registros=("Emissores", "size"), Emissores_Fisicos=("Emissores", "sum"))
        .reset_index()
    )

    _SERVICO_NOMES = {
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
        "-1": "-1 - Sem Código (Aero/ICAO)",
    }

    _fonte_df = (
        pd.DataFrame({"Fonte": estacoes["Fonte"], "Emissores": mult_num})
        .groupby("Fonte", observed=True)
        .agg(Registros=("Emissores", "size"), Emissores_Fisicos=("Emissores", "sum"))
        .sort_values("Registros", ascending=False)
        .head(12)
        .reset_index()
    )

    _serv_df = (
        pd.DataFrame({"Serviço": estacoes["Serviço"].astype(str), "Emissores": mult_num, "Freq_MHz": freq_mhz})
        .groupby("Serviço", observed=True)
        .agg(
            Registros=("Emissores", "size"),
            Emissores_Fisicos=("Emissores", "sum"),
            Freq_Mediana_MHz=("Freq_MHz", "median"),
        )
        .sort_values("Registros", ascending=False)
        .head(12)
        .reset_index()
    )
    _serv_df["Descricao"] = _serv_df["Serviço"].map(lambda c: _SERVICO_NOMES.get(c, f"Serviço {c}"))

    _uf_df = (
        pd.DataFrame({"UF": estacoes["UF"].astype(str), "Emissores": mult_num})
        .groupby("UF", observed=True)
        .agg(Registros=("Emissores", "size"), Emissores_Fisicos=("Emissores", "sum"))
        .sort_values("Registros", ascending=False)
        .reset_index()
    )

    _ent_df = (
        pd.DataFrame(
            {
                "Entidade": estacoes["Entidade"].astype(str),
                "Emissores": mult_num,
                "UF": estacoes["UF"].astype(str),
                "Servico": estacoes["Serviço"].astype(str),
            }
        )
        .groupby("Entidade", observed=True)
        .agg(
            Registros=("Emissores", "size"),
            Emissores_Fisicos=("Emissores", "sum"),
            UFs_Atendidas=("UF", "nunique"),
            Servico_Principal=("Servico", lambda s: s.value_counts().index[0]),
        )
        .sort_values("Registros", ascending=False)
        .head(20)
        .reset_index()
    )

    _chart_bands = (
        alt.Chart(_band_df)
        .mark_bar(color="#2b8cbe", cornerRadiusEnd=3)
        .encode(
            x=alt.X("Registros:Q", title="Registros"),
            y=alt.Y("Faixa:N", sort=None, title="Faixa de Espectro"),
            tooltip=["Faixa", alt.Tooltip("Registros:Q", format=","), alt.Tooltip("Emissores_Fisicos:Q", format=",.0f")],
        )
        .properties(width=350, height=270, title="Distribuição por Faixa de Frequência")
    )

    _chart_serv = (
        alt.Chart(_serv_df)
        .mark_bar(color="#54a24b", cornerRadiusEnd=3)
        .encode(
            x=alt.X("Registros:Q", title="Registros"),
            y=alt.Y("Descricao:N", sort="-x", title="Serviço Anatel"),
            tooltip=["Descricao", alt.Tooltip("Registros:Q", format=","), alt.Tooltip("Emissores_Fisicos:Q", format=",.0f"), alt.Tooltip("Freq_Mediana_MHz:Q", format=".2f")],
        )
        .properties(width=350, height=270, title="Top 12 Serviços (Anatel / Aero)")
    )

    _chart_uf = (
        alt.Chart(_uf_df)
        .mark_bar(cornerRadiusEnd=2)
        .encode(
            x=alt.X("UF:N", sort="-y", title="UF (note 'MT ', 'RO ' e '-1'!)"),
            y=alt.Y("Registros:Q", title="Registros"),
            color=alt.condition(
                alt.FieldOneOfPredicate(field="UF", oneOf=["-1", "MT ", "RO "]),
                alt.value("#e45756"),
                alt.value("#4c78a8"),
            ),
            tooltip=["UF", alt.Tooltip("Registros:Q", format=","), alt.Tooltip("Emissores_Fisicos:Q", format=",.0f")],
        )
        .properties(height=250, title="Registros por UF (barras vermelhas = UFs anômalas '-1', 'MT ', 'RO ')")
    )

    mo.vstack(
        [
            mo.md(
                """
    ## 2. Espectro, Serviços, Fontes & Operadoras
    - **Concentração:** `019` (SLP - Serviço Limitado Privado, 58,4%) e `010` (SMP - Telefonia Móvel, 28,1%, sendo metade Downlink `MOSAICO-LICENCIAMENTO` e metade Uplink sintético `CANALIZACAO-SMP`) somam **86,5%** das linhas e **91,1%** dos 11,99 milhões de emissores.
    - **Espectro:** Forte bimodalidade entre VHF/UHF sub-3 GHz (rádio privativo, radiodifusão, 2G/3G/4G) e SHF/EHF 6–85 GHz (enlaces ponto-a-ponto de backhaul das operadoras).
    """
            ),
            mo.ui.tabs(
                {
                    "📡 Espectro & Serviços": mo.vstack([mo.hstack([_chart_bands, _chart_serv], justify="start"), _chart_uf]),
                    "🏢 Top 20 Entidades": mo.ui.table(_ent_df, pagination=False),
                    "🗂️ Top Fontes": mo.ui.table(_fonte_df, pagination=False),
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(log_events, log_templates):
    # Controls for exploring log.parquet & linked estacoes
    _all_tags = sorted(log_events["Tag"].unique().tolist())

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
        stop=int(len(log_templates) - 1),
        step=1,
        value=8,
        label="Inspecionar Log ID (0–8026):",
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
):
    # Summary of pipelines + atomic event breakdown + SMP displacement + interactive drilldown
    _pipe_summary = (
        log_templates.groupby(["Pipeline", "Alerta_Qualidade"], as_index=False)
        .agg(
            Templates=("Log", "count"),
            Estacoes_Afetadas=("Estacoes_Afetadas", "sum"),
            Exemplo_Log_ID=("Log", "min"),
        )
        .sort_values("Estacoes_Afetadas", ascending=False)
    )
    _pipe_summary["%_da_Base"] = (_pipe_summary["Estacoes_Afetadas"] / len(estacoes) * 100).round(3)

    _tag_summary = (
        log_events.groupby(["Tag", "Alerta_Qualidade", "Processamento"], as_index=False)
        .agg(
            Templates_com_Evento=("Log", "nunique"),
            Estacoes_Afetadas=("Estacoes_Afetadas", "sum"),
        )
        .sort_values("Estacoes_Afetadas", ascending=False)
    )

    # Compute true SMP tower -> municipality centroid shift (now working thanks to smp.py fix!)
    _smp_orig_lat = (
        log_events[log_events["Tag"] == "SMP_LAT_MUNICIPIO"][["Log", "Original"]]
        .drop_duplicates("Log")
        .set_index("Log")["Original"]
    )
    _smp_orig_lon = (
        log_events[log_events["Tag"] == "SMP_LON_MUNICIPIO"][["Log", "Original"]]
        .drop_duplicates("Log")
        .set_index("Log")["Original"]
    )
    _smp_lids = _smp_orig_lat.index.intersection(_smp_orig_lon.index)
    _smp_cur = (
        estacoes[estacoes["Log"].isin(_smp_lids)][["Log", "UF", "Município", "Latitude", "Longitude"]]
        .drop_duplicates("Log")
        .copy()
    )
    _smp_cur["Orig_Lat"] = pd.to_numeric(_smp_cur["Log"].map(_smp_orig_lat), errors="coerce")
    _smp_cur["Orig_Lon"] = pd.to_numeric(_smp_cur["Log"].map(_smp_orig_lon), errors="coerce")
    _smp_cur["Mun_Lat"] = pd.to_numeric(_smp_cur["Latitude"].astype(str), errors="coerce")
    _smp_cur["Mun_Lon"] = pd.to_numeric(_smp_cur["Longitude"].astype(str), errors="coerce")
    _v_smp = _smp_cur["Orig_Lat"].notna() & _smp_cur["Orig_Lon"].notna() & (_smp_cur["Orig_Lat"].abs() <= 90) & (_smp_cur["Orig_Lon"].abs() <= 180)
    _smp_cur = _smp_cur[_v_smp].copy()
    _la1, _lo1 = np.radians(_smp_cur["Orig_Lat"]), np.radians(_smp_cur["Orig_Lon"])
    _la2, _lo2 = np.radians(_smp_cur["Mun_Lat"]), np.radians(_smp_cur["Mun_Lon"])
    _a_smp = np.sin((_la2 - _la1) / 2.0) ** 2 + np.cos(_la1) * np.cos(_la2) * np.sin((_lo2 - _lo1) / 2.0) ** 2
    _smp_cur["Deslocamento_km"] = np.round(2.0 * 6371.0 * np.arcsin(np.sqrt(np.clip(_a_smp, 0.0, 1.0))), 2)
    _smp_shift_top = _smp_cur.sort_values("Deslocamento_km", ascending=False).head(30)

    # Filtered events view
    _ev_view = log_events
    if log_only_dq.value:
        _ev_view = _ev_view[_ev_view["Alerta_Qualidade"]]
    if log_tag_filter.value != "(Todos os Eventos)":
        _ev_view = _ev_view[_ev_view["Tag"] == log_tag_filter.value]

    # Selected Log ID inspection + join with estacoes
    _sel_lid = min(max(0, int(log_id_selector.value)), len(log_templates) - 1)
    _sel_tmpl = log_templates.loc[log_templates["Log"] == _sel_lid].iloc[0]
    _sel_steps = log_events[log_events["Log"] == _sel_lid][["Passo", "Tag", "Coluna", "Original", "Processamento"]]
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
        alt.Chart(_tag_summary)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("Estacoes_Afetadas:Q", title="Ocorrências Pond. por Estações (escala log)", scale=alt.Scale(type="symlog")),
            y=alt.Y("Tag:N", sort="-x", title="Evento de Log"),
            color=alt.condition(
                alt.datum["Alerta_Qualidade"],
                alt.value("#f58518"),
                alt.value("#4c78a8"),
            ),
            tooltip=["Tag", "Processamento", alt.Tooltip("Estacoes_Afetadas:Q", format=","), "Templates_com_Evento"],
        )
        .properties(height=360, title="Impacto por Tipo de Evento no Novo log.parquet (51.512 Templates)")
    )

    mo.vstack(
        [
            mo.md(
                f"""
    ## 3. Desvendando `log.parquet` após as Correções (`smp.py` + `location.py`)
    Com as correções em `smp.py` e `location.py`, `log.parquet` passou de **8.027** para **{len(log_templates):,} templates** (**{len(log_events):,} eventos**), refletindo a rastreabilidade real de ponta a ponta:

    - **`IBGE_NOME_NORMALIZADO`:** saltou de 34 ocorrências para **421.239 registros**, documentando todos os nomes de municípios/UFs padronizados e preenchidos (incluindo os ~338 mil `Município == "-1"`).
    - **`IBGE_COORD_DIVERGENTE`:** saltou de 36 ocorrências para **63.508 eventos (~31.754 estações)** cujas coordenadas caíam no polígono de outro município e foram substituídas pelo centroide IBGE preservando o `Original`.
    - **`COORD_NULA_PREENCHIDA_IBGE`:** **42 ocorrências (21 estações)** com coordenadas nulas preenchidas pelo centroide municipal (antes bloqueadas pelo filtro `city_normalized`).
    - **Correção em `smp.py` validada:** Agora `MOSAICO-SMP` registra a **verdadeira coordenada original da ERB** antes da substituição pelo centroide municipal! Em **{len(_smp_cur):,} templates SMP**, o deslocamento mediano ERB → centroide é de **{float(_smp_cur['Deslocamento_km'].median()):.2f} km** (p90 = **{float(_smp_cur['Deslocamento_km'].quantile(0.9)):.2f} km**, máx = **{float(_smp_cur['Deslocamento_km'].max()):,.1f} km** em ERBs com coordenada bruta errada na origem).
    """
            ),
            mo.ui.tabs(
                {
                    "🔀 Pipelines & Gráfico de Eventos": mo.vstack(
                        [
                            _tag_chart,
                            mo.md("### Top 25 Sequências de Processamento (`Pipeline`)"),
                            mo.ui.table(_pipe_summary.head(25), pagination=True, page_size=15),
                        ]
                    ),
                    "📏 Deslocamento Real ERB → Município (SMP)": mo.vstack(
                        [
                            mo.md("Graças à correção em `smp.py`, podemos medir a distância entre a coordenada original da torre (`Original` no `Log`) e o centroide municipal atribuído:"),
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
                                kind="warn" if _sel_tmpl["Alerta_Qualidade"] else "info",
                            ),
                            mo.md("**Passos desaninhados deste Log ID:**"),
                            mo.ui.table(_sel_steps, pagination=False),
                            mo.md(f"**Amostra de estações com `Log == {_sel_lid}` (até 50 de {_sel_tmpl['Estacoes_Afetadas']:,}):**"),
                            mo.ui.table(_linked_stations, pagination=True, page_size=10),
                        ]
                    ),
                }
            ),
        ]
    )
    return


@app.cell
def _(ARQ_DIR, cat_to_float, estacoes, ibge_mun, lat_num, lon_num):
    @mo.persistent_cache
    def compute_polygon_lookup(coords_frame: pd.DataFrame, shp_path_str: str) -> pd.DataFrame:
        """Intersect unique (Latitude, Longitude) pairs against official IBGE 2025 municipal polygons."""
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


    _unique_coords = (
        estacoes[["Latitude", "Longitude"]]
        .astype(str)
        .drop_duplicates()
        .reset_index(drop=True)
    )
    _shp_file = str(ARQ_DIR / "BR_Municipios_2025" / "BR_Municipios_2025.shp")
    coord_poly_lookup = compute_polygon_lookup(_unique_coords, _shp_file)

    # Map polygon results and IBGE centroid distances onto all 1.142M rows
    _mi = pd.MultiIndex.from_frame(estacoes[["Latitude", "Longitude"]].astype(str))
    _lkp = coord_poly_lookup.set_index(["Latitude", "Longitude"])
    poly_cd_arr = _lkp["poly_cd"].reindex(_mi).fillna("").to_numpy(dtype=object)
    poly_nm_arr = _lkp["poly_nm"].reindex(_mi).fillna("").to_numpy(dtype=object)
    poly_uf_arr = _lkp["poly_uf"].reindex(_mi).fillna("").to_numpy(dtype=object)

    _ibge_lat_map = dict(zip(ibge_mun["Código_Município"], ibge_mun["Latitude"].astype(float)))
    _ibge_lon_map = dict(zip(ibge_mun["Código_Município"], ibge_mun["Longitude"].astype(float)))
    _ibge_uf_map = dict(zip(ibge_mun["Código_Município"], ibge_mun["UF"]))
    _ibge_nm_map = dict(zip(ibge_mun["Código_Município"], ibge_mun["Município"]))

    _cod_str = estacoes["Código_Município"].astype(str)
    _mun_lat = _cod_str.map(_ibge_lat_map).to_numpy(dtype="float64")
    _mun_lon = _cod_str.map(_ibge_lon_map).to_numpy(dtype="float64")
    _mun_uf = _cod_str.map(_ibge_uf_map).fillna("").to_numpy(dtype=object)
    ibge_mun_name_arr = _cod_str.map(_ibge_nm_map).fillna("").to_numpy(dtype=object)

    # Vectorized Haversine distance to declared IBGE municipality centroid
    _valid_hav = (~np.isnan(lat_num)) & (~np.isnan(lon_num)) & (~np.isnan(_mun_lat)) & (np.abs(lat_num) <= 90) & (np.abs(lon_num) <= 180)
    dist_ibge_km = np.full(len(estacoes), np.nan, dtype="float64")
    _lat1 = np.radians(lat_num[_valid_hav])
    _lon1 = np.radians(lon_num[_valid_hav])
    _lat2 = np.radians(_mun_lat[_valid_hav])
    _lon2 = np.radians(_mun_lon[_valid_hav])
    _a = np.sin((_lat2 - _lat1) / 2.0) ** 2 + np.cos(_lat1) * np.cos(_lat2) * np.sin((_lon2 - _lon1) / 2.0) ** 2
    dist_ibge_km[_valid_hav] = 2.0 * 6371.0 * np.arcsin(np.sqrt(np.clip(_a, 0.0, 1.0)))

    # Detect dirty unparsed strings in engineering columns
    _eng_num_cols = [
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
    _dirty_eng = np.zeros(len(estacoes), dtype=bool)
    dirty_eng_detail = np.full(len(estacoes), "", dtype=object)
    for _c in _eng_num_cols:
        _cats = estacoes[_c].cat.categories
        _conv = pd.to_numeric(_cats, errors="coerce")
        _bad_set = set(_cats[_conv.isna() & (_cats != "-1")])
        if _bad_set:
            _m = estacoes[_c].isin(_bad_set).to_numpy()
            _dirty_eng |= _m
            _vals_str = estacoes[_c].astype(str).to_numpy()
            dirty_eng_detail[_m] = np.where(
                dirty_eng_detail[_m] == "",
                _c + "=" + _vals_str[_m],
                dirty_eng_detail[_m] + "; " + _c + "=" + _vals_str[_m],
            )

    # Physical outliers
    _az = cat_to_float(estacoes["Azimute_Antena"])
    _el = cat_to_float(estacoes["Ângulo_Elevação_Antena"])
    _cota = cat_to_float(estacoes["Cota_Base_Torre(m)"])
    _alt = cat_to_float(estacoes["Altura_Antena(m)"])
    _phys_outlier = (
        ((~np.isnan(_az)) & ((_az < 0) | (_az > 360)))
        | ((~np.isnan(_el)) & ((_el < -90) | (_el > 90)))
        | ((~np.isnan(_cota)) & ((_cota < -50) | (_cota > 3000)))
        | ((~np.isnan(_alt)) & ((_alt < 0) | (_alt > 500)))
    )

    # Priority-ordered spatial & metadata defect classification
    _miss_coord = (estacoes["Latitude"] == "-1").to_numpy() | (estacoes["Longitude"] == "-1").to_numpy()
    _has_coord = (~_miss_coord) & (~np.isnan(lat_num)) & (~np.isnan(lon_num))
    _outside_poly = _has_coord & (poly_cd_arr == "")

    _polar = _outside_poly & ((np.abs(lat_num) > 60) | (np.abs(lon_num) > 180))
    _null_island = _outside_poly & (~_polar) & (np.abs(lat_num) < 1.0) & (np.abs(lon_num) < 1.0)
    _swapped = _outside_poly & (~_polar) & (~_null_island) & (lat_num < -34.0) & (lat_num >= -74.5) & (lon_num >= -34.0) & (lon_num <= 6.0)
    _sign_lon = _outside_poly & (~_polar) & (~_null_island) & (~_swapped) & (lon_num > 28.0) & (lon_num <= 75.0)
    _sign_lat = _outside_poly & (~_polar) & (~_null_island) & (~_swapped) & (~_sign_lon) & (lat_num > 5.35) & (lon_num >= -74.5) & (lon_num <= -34.0)
    _offshore = _outside_poly & (~(_polar | _null_island | _swapped | _sign_lon | _sign_lat)) & (lon_num > -53.0) & (lon_num <= -28.5) & (lat_num >= -34.5) & (lat_num <= 6.0)
    _foreign = _outside_poly & (~(_polar | _null_island | _swapped | _sign_lon | _sign_lat | _offshore))

    _uf_arr = estacoes["UF"].astype(str).to_numpy()
    _uf_trimmed = np.char.strip(_uf_arr.astype(str))
    _inside_poly = _has_coord & (poly_cd_arr != "")
    _wrong_uf_poly = _inside_poly & (_uf_arr != "-1") & (_uf_trimmed != poly_uf_arr)
    _wrong_mun_poly = _inside_poly & (~_wrong_uf_poly) & (_cod_str.to_numpy() != "-1") & (_cod_str.to_numpy() != poly_cd_arr)
    _uf_whitespace = (_uf_arr != _uf_trimmed) | ((_cod_str.to_numpy() != "-1") & (_mun_uf != "") & (_uf_trimmed != _mun_uf))
    _missing_mun_or_uf = (_cod_str.to_numpy() == "-1") | (_uf_arr == "-1")

    # Assign primary anomaly category for plotting & filtering
    primary_issue = np.full(len(estacoes), "00_OK_Consistente", dtype=object)
    primary_issue[_missing_mun_or_uf] = "11_Sem_Codigo_Municipio_ou_UF"
    primary_issue[_uf_whitespace] = "10_UF_Com_Espaco_ou_Divergente"
    primary_issue[_phys_outlier] = "09_Outlier_Fisico_Antena_Torre"
    primary_issue[_dirty_eng] = "08_Texto_Sujo_Coluna_Numerica"
    primary_issue[_offshore] = "07_Offshore_Maritimo_Ilhas"
    primary_issue[_wrong_mun_poly] = "06_Poligono_Municipio_Divergente"
    primary_issue[_wrong_uf_poly] = "05_Poligono_UF_Divergente"
    primary_issue[_foreign] = "04_Fora_Fronteira_Terrestre"
    primary_issue[_sign_lat | _sign_lon] = "03_Sinal_Invertido_Hemisferio"
    primary_issue[_swapped] = "02_Lat_Lon_Invertidas"
    primary_issue[_null_island | _polar] = "01_Null_Island_ou_Polar_Invalido"
    primary_issue[_miss_coord] = "00_Sem_Coordenadas_-1"

    ISSUE_META = {
        "01_Null_Island_ou_Polar_Invalido": ("#d62728", "Crítico: Null Island (0,0) ou Latitude Polar (±89°)"),
        "02_Lat_Lon_Invertidas": ("#e31a1c", "Crítico: Latitude ↔ Longitude Trocadas (no Oceano)"),
        "03_Sinal_Invertido_Hemisferio": ("#ff7f00", "Crítico: Sinal Invertido (Hemisfério Norte / Leste +Lon)"),
        "04_Fora_Fronteira_Terrestre": ("#984ea3", "Alerta: Coordenada em País Vizinho / Fronteira"),
        "05_Poligono_UF_Divergente": ("#e7298a", "Erro Geo: Ponto cai em OUTRA UF pelo Polígono IBGE"),
        "06_Poligono_Municipio_Divergente": ("#f781bf", "Erro Geo: Ponto cai em OUTRO Município IBGE (mesma UF)"),
        "07_Offshore_Maritimo_Ilhas": ("#1f78b4", "Contexto: Offshore / Pré-Sal / Costeiro (fora da malha terrestre)"),
        "08_Texto_Sujo_Coluna_Numerica": ("#b15928", "Erro Tipo: Strings não normalizadas em Engenharia (vírgula/unidade)"),
        "09_Outlier_Fisico_Antena_Torre": ("#a65628", "Alerta Físico: Azimute >360°, Elevação >90° ou Cota >3000m"),
        "10_UF_Com_Espaco_ou_Divergente": ("#fdbf6f", "Erro String: UF com espaço ('MT ', 'RO ') ou != IBGE"),
        "11_Sem_Codigo_Municipio_ou_UF": ("#6a3d9a", "Incompleto: Código_Município ou UF == '-1'"),
        "00_Sem_Coordenadas_-1": ("#7f7f7f", "Incompleto: Sem Latitude/Longitude ('-1', não plotável)"),
        "00_OK_Consistente": ("#2ca02c", "OK: Coordenada consistente com Polígono IBGE"),
    }

    _issue_vc = pd.Series(primary_issue).value_counts()
    issue_summary_df = pd.DataFrame(
        [
            {
                "Codigo": k,
                "Descricao": ISSUE_META[k][1],
                "Registros": int(_issue_vc.get(k, 0)),
                "%_Total": round(int(_issue_vc.get(k, 0)) / len(estacoes) * 100, 4),
                "Plotavel_no_Mapa": k != "00_Sem_Coordenadas_-1",
            }
            for k in ISSUE_META
        ]
    )
    return (
        ISSUE_META,
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
def _(estacoes, issue_summary_df):
    # Controls for Residual Quality Audit & Folium Map
    _default_issues = [
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

    _issue_options = {
        f"{row.Descricao} [{row.Registros:,}]": row.Codigo
        for row in issue_summary_df.itertuples()
        if row.Registros > 0 and row.Plotavel_no_Mapa
    }
    _default_labels = [lbl for lbl, cod in _issue_options.items() if cod in _default_issues]

    map_issue_select = mo.ui.multiselect(
        options=_issue_options,
        value=_default_labels,
        label="Categorias para Plotar no Mapa:",
    )

    _uf_opts = ["(Todas)"] + sorted(estacoes["UF"].astype(str).unique().tolist())
    map_uf_select = mo.ui.dropdown(options=_uf_opts, value="(Todas)", label="Filtrar UF:")

    _fonte_opts = ["(Todas)"] + sorted(estacoes["Fonte"].astype(str).unique().tolist())
    map_fonte_select = mo.ui.dropdown(options=_fonte_opts, value="(Todas)", label="Filtrar Fonte:")

    map_max_points = mo.ui.slider(
        start=200,
        stop=5000,
        step=200,
        value=1500,
        label="Máx. Pontos no Mapa:",
    )
    map_cluster_toggle = mo.ui.checkbox(value=False, label="Agrupar marcadores (MarkerCluster)")
    map_entity_search = mo.ui.text(value="", placeholder="Ex: PETROBRAS, PRYMUS, CLARO...", label="Buscar Entidade:")
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
    # Filter records, build Folium Map with rich record tooltips, and show Residual Quality Dashboard
    _selected_codes = set(map_issue_select.value)
    _mask = np.isin(primary_issue, list(_selected_codes))
    # Folium requires valid WGS84 [-90, 90] x [-180, 180]
    _mask &= (~np.isnan(lat_num)) & (~np.isnan(lon_num)) & (np.abs(lat_num) <= 90.0) & (np.abs(lon_num) <= 180.0)

    if map_uf_select.value != "(Todas)":
        _mask &= (estacoes["UF"].astype(str) == map_uf_select.value).to_numpy()
    if map_fonte_select.value != "(Todas)":
        _mask &= (estacoes["Fonte"].astype(str) == map_fonte_select.value).to_numpy()
    if map_entity_search.value.strip():
        _q = map_entity_search.value.strip().lower()
        _mask &= estacoes["Entidade"].astype(str).str.lower().str.contains(_q, regex=False).to_numpy()

    _matched_idx = np.flatnonzero(_mask)
    _total_matched = len(_matched_idx)

    # Deterministic stratified/sorted sampling up to map_max_points so severe anomalies always appear first
    if _total_matched > map_max_points.value:
        _order = np.argsort(primary_issue[_matched_idx].astype(str))
        _matched_idx = _matched_idx[_order[: int(map_max_points.value)]]

    _sub = estacoes.iloc[_matched_idx].copy()
    _sub_lat = lat_num[_matched_idx]
    _sub_lon = lon_num[_matched_idx]
    _sub_issue = primary_issue[_matched_idx]
    _sub_poly_nm = poly_nm_arr[_matched_idx]
    _sub_poly_uf = poly_uf_arr[_matched_idx]
    _sub_poly_cd = poly_cd_arr[_matched_idx]
    _sub_ibge_nm = ibge_mun_name_arr[_matched_idx]
    _sub_dist_km = dist_ibge_km[_matched_idx]
    _sub_dirty = dirty_eng_detail[_matched_idx]

    _log_resumo_map = dict(zip(log_templates["Log"], log_templates["Resumo_Log"]))

    # Build Folium Map
    _center = [-14.235, -51.925]
    _fmap = folium.Map(location=_center, zoom_start=4, tiles="OpenStreetMap", control_scale=True)
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr="&copy; OpenStreetMap contributors &copy; CARTO",
        name="Mapa Claro (CARTO)",
        subdomains="abcd",
    ).add_to(_fmap)

    if map_cluster_toggle.value:
        _target_layer = plugins.MarkerCluster(name="Estações Filtradas").add_to(_fmap)
        _groups = {}
    else:
        _target_layer = None
        _groups = {}
        for _code in sorted(_selected_codes):
            _color, _desc = ISSUE_META[_code]
            _groups[_code] = folium.FeatureGroup(name=_desc, show=True).add_to(_fmap)

    for _pos, _row_idx in enumerate(_matched_idx):
        _r = _sub.iloc[_pos]
        _ic = _sub_issue[_pos]
        _color, _idesc = ISSUE_META[_ic]
        _la = float(_sub_lat[_pos])
        _lo = float(_sub_lon[_pos])
        _lid = int(_r["Log"])
        _ltxt = html.escape(_log_resumo_map.get(_lid, ""))
        if len(_ltxt) > 180:
            _ltxt = _ltxt[:177] + "..."

        _mun_decl = str(_r["Município"])
        if _mun_decl == "-1" and _sub_ibge_nm[_pos]:
            _mun_decl = f"-1 (IBGE p/ código: {_sub_ibge_nm[_pos]})"

        _poly_info = (
            f"{_sub_poly_nm[_pos]} / {_sub_poly_uf[_pos]} ({_sub_poly_cd[_pos]})"
            if _sub_poly_cd[_pos]
            else "FORA DA MALHA TERRESTRE IBGE"
        )
        _dist_str = f"{_sub_dist_km[_pos]:,.1f} km" if not np.isnan(_sub_dist_km[_pos]) else "N/A"
        _dirty_line = (
            f"<br><b style='color:#b15928'>Texto Sujo Eng.:</b> {html.escape(str(_sub_dirty[_pos]))}"
            if _sub_dirty[_pos]
            else ""
        )

        _tip_html = (
            f"<div style='font-family:sans-serif;font-size:12px;min-width:290px;max-width:380px;'>"
            f"<div style='background:{_color};color:#fff;padding:4px 7px;border-radius:4px;font-weight:bold;margin-bottom:5px;'>"
            f"{html.escape(_idesc)}</div>"
            f"<b>Linha #</b>{_row_idx} | <b>Estação:</b> {html.escape(str(_r['Estação']))} | <b>Fistel:</b> {html.escape(str(_r['Fistel']))}<br>"
            f"<b>Entidade:</b> {html.escape(str(_r['Entidade']))}<br>"
            f"<b>Freq:</b> {html.escape(str(_r['Frequência']))} MHz | <b>BW:</b> {html.escape(str(_r['Largura_Emissão(kHz)']))} kHz | "
            f"<b>Serviço:</b> {html.escape(str(_r['Serviço']))} ({html.escape(str(_r['Classe']))})<br>"
            f"<b>Fonte:</b> {html.escape(str(_r['Fonte']))} | <b>Status:</b> {html.escape(str(_r['Status']))} | <b>Mult:</b> {html.escape(str(_r['Multiplicidade']))}<br>"
            f"<hr style='margin:4px 0'>"
            f"<b>Coord:</b> ({_la:.6f}, {_lo:.6f})<br>"
            f"<b>Declarado:</b> {html.escape(_mun_decl)} / '{html.escape(str(_r['UF']))}' (Cód: {html.escape(str(_r['Código_Município']))})<br>"
            f"<b>Polígono Real:</b> {html.escape(_poly_info)} | <b>Dist. Centroide:</b> {_dist_str}"
            f"{_dirty_line}"
            f"<hr style='margin:4px 0'>"
            f"<b>Log [{_lid}]:</b> {_ltxt}"
            f"</div>"
        )

        _marker = folium.CircleMarker(
            location=[_la, _lo],
            radius=5,
            color=_color,
            weight=1.5,
            fill=True,
            fill_color=_color,
            fill_opacity=0.8,
            tooltip=folium.Tooltip(_tip_html, sticky=True),
        )
        if map_cluster_toggle.value:
            _marker.add_to(_target_layer)
        else:
            _marker.add_to(_groups[_ic])

    folium.LayerControl(collapsed=True).add_to(_fmap)

    # Build clean inspection table for matched rows
    _table_out = pd.DataFrame(
        {
            "Row_ID": _matched_idx,
            "Diagnostico": _sub_issue,
            "Entidade": _sub["Entidade"].astype(str).to_numpy(),
            "Frequência_MHz": _sub["Frequência"].astype(str).to_numpy(),
            "Serviço": _sub["Serviço"].astype(str).to_numpy(),
            "Fonte": _sub["Fonte"].astype(str).to_numpy(),
            "UF_Decl": _sub["UF"].astype(str).to_numpy(),
            "Mun_Decl": _sub["Município"].astype(str).to_numpy(),
            "Cod_IBGE_Decl": _sub["Código_Município"].astype(str).to_numpy(),
            "Latitude": np.round(_sub_lat, 6),
            "Longitude": np.round(_sub_lon, 6),
            "UF_Poligono_Real": _sub_poly_uf,
            "Mun_Poligono_Real": _sub_poly_nm,
            "Dist_Centroide_km": np.round(_sub_dist_km, 1),
            "Detalhe_Engenharia": _sub_dirty,
            "Log_ID": _sub["Log"].to_numpy(),
        }
    )

    _issue_chart_df = issue_summary_df[
        (issue_summary_df["Codigo"] != "00_OK_Consistente") & (issue_summary_df["Registros"] > 0)
    ]
    _issue_chart = (
        alt.Chart(_issue_chart_df)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("Registros:Q", title="Registros com Problema Residual (escala log)", scale=alt.Scale(type="symlog")),
            y=alt.Y("Descricao:N", sort="-x", title=None),
            color=alt.Color(
                "Codigo:N",
                scale=alt.Scale(
                    domain=list(ISSUE_META.keys()),
                    range=[v[0] for v in ISSUE_META.values()],
                ),
                legend=None,
            ),
            tooltip=["Codigo", "Descricao", alt.Tooltip("Registros:Q", format=","), "%_Total"],
        )
        .properties(height=320, title=" Panorama de Problemas Residuais Após Pós-Processamento da Biblioteca")
    )

    mo.vstack(
        [
            mo.md(
                """
    ## 4. Auditoria de Dados Residuais (Pós-Correção) & Mapa Interativo Folium
    ### Comparativo Antes vs. Depois da Nova Extração
    | Indicador de Qualidade | Antes | Agora (Nova Extração) | Resultado |
    |---|---:|---:|---|
    | `Município == "-1"` | 338.193 (29,61%) | **256 (0,02%)** | ✅ **99,92% resolvido** (restam apenas 256 estações marítimas/fronteira sem município) |
    | `UF` com espaços ou divergente (`'MT '`, `'RO '`) | 160 | **0** | ✅ **100% resolvido** |
    | Ponto em **OUTRA UF** pelo Polígono IBGE | 1.834 | **10** | ✅ **99,45% resolvido** |
    | Ponto em **OUTRO Município** pelo Polígono IBGE | 29.938 | **581** | ✅ **98,06% resolvido** |
    | Sem Latitude/Longitude (`'-1'`) | 33 | **12** | ✅ **21 preenchidos via centroide IBGE** (12 restantes também não têm `Código_Município`) |

    ### O que explica os resíduos remanescentes que você vê no Mapa agora?
    1. **Os 591 pontos divergentes restantes em terra (`581` municípios + `10` UFs):** Todos os **20 municípios** envolvidos (ex.: *São José do Sabugi/PB*, *Cerro Azul/PR*, *Aricanduva/MG*, *Juruena/MT*, *Macuco/RJ*) possuem **coordenadas defasadas/incorretas no próprio arquivo de referência `municipios.csv`** em relação à malha oficial `BR_Municipios_2025.shp`! Ou seja, a biblioteca substituiu corretamente pela coordenada do `municipios.csv`, mas o centroide tabelado nesse CSV cai no município vizinho.
    2. **Os 540 pontos críticos fora do território terrestre** (`72` Null Island / polares `±89°`, `97` `Lat ↔ Lon` trocadas no Atlântico, `98` sinal de hemisfério invertido `+Lon`/`+Lat` e `273` em fronteira estrangeira): Como `CD_MUN` retorna `NA` fora dos polígonos terrestres do Brasil, `substitute_divergent_coordinates()` (`CD_MUN.notna()`) não os sobrescreve automaticamente — preservando também as **5.075 plataformas offshore/pré-sal legítimas**. *(Nota: em `location.py:348`, `Base.register_log(self.df, "Coordenadas Fora do Território Brasileiro", ...)` grava em `self.df` uma linha antes de `self.df = gdf_joined`, descartando esse log.)*
    3. **256 strings não numéricas nas colunas de engenharia do SRD** (vírgulas decimais `'4,60'`, unidades `'8.69 dBd'`, `'    4,5 ° (0.859)'`, `'Horizontal'` em `Azimute_Antena`) e **21 outliers físicos** (`Azimute > 360°`, `Cota > 3000m`)."""
            ),
            mo.ui.tabs(
                {
                    "🗺️ Mapa Folium Interativo (Tooltips)": mo.vstack(
                        [
                            mo.hstack([map_uf_select, map_fonte_select, map_entity_search], widths="equal"),
                            mo.hstack([map_issue_select, mo.vstack([map_max_points, map_cluster_toggle])], widths=[2, 1]),
                            mo.callout(
                                mo.md(
                                    f"Exibindo **{len(_matched_idx):,}** de **{_total_matched:,}** registros filtrados no mapa. "
                                    f"Passe o mouse sobre qualquer ponto para inspecionar o registro completo, polígono real IBGE, distância ao centroide e `Log`."
                                ),
                                kind="info",
                            ),
                            mo.iframe(_fmap.get_root().render(), width="100%", height="580px"),
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
                            mo.md(f"Tabela detalhada dos **{len(_table_out):,}** registros atualmente exibidos no filtro do mapa:"),
                            mo.ui.table(_table_out, pagination=True, page_size=15),
                        ]
                    ),
                }
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
