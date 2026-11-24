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
    mo.md(
        f"""
    # 📡 RFDataHub — Análise Exploratória, Rastreabilidade & Auditoria de Qualidade
    **Diretório ativo:** `{DATA_DIR}` | **Última atualização (`estacoes.parquet`):** `{estacoes_mtime_str}` ({estacoes_size_mb:.1f} MB) | **`log.parquet`:** {log_size_mb:.2f} MB

    Este aplicativo interativo atualiza automaticamente todos os indicadores, tabelas, gráficos e mapas a cada nova execução do pipeline:
    1. **Visão Geral & Completude do Esquema** — volume consolidado vs. emissores físicos (`Multiplicidade`), preenchimento por coluna e sentinelas `"-1"`.
    2. **Espectro, Serviços, Fontes & Operadoras** — distribuição por faixas de frequência, serviços Anatel/Aero, UFs e principais entidades licenciadas.
    3. **Explorador Estruturado de `log.parquet`** — desaninhamento dos templates JSON fatorados, fluxos de processamento (`Pipeline`), deslocamento ERB → município no SMP e inspeção antes/depois.
    4. **Investigação Especial — Coordenadas Divergentes (`IBGE_COORD_DIVERGENTE`)** — diagnóstico geoespacial das estações cujas coordenadas originais caíam em outro polígono municipal do IBGE (amplificação por múltiplos canais, efeito de fronteira/topo de serra vs. erros cadastrais e mapa antes → depois).
    5. **Auditoria de Qualidade Residual & Mapa Interativo Folium** — anomalias remanescentes pós-processamento com inspeção individual de estações.
    """
    )

    return


@app.cell
def _():
    OFFICIAL_UFS = frozenset(
        {
            "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO",
            "MA", "MG", "MS", "MT", "PA", "PB", "PE", "PI", "PR",
            "RJ", "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO",
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
        s_num[s.astype(str).to_numpy() == "-1"] = np.nan
        return s_num


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
        if proc.startswith("Coordenadas e Código do Município nulos"):
            return ("MUN_E_COORD_NULOS", True)
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


    return OFFICIAL_UFS, SERVICO_NOMES, cat_to_float, classify_log_step


@app.cell
def _(cat_to_float, classify_log_step):
    # Portable data directory resolution (works inside rfdatahub or standalone via marimo run/edit)
    def _resolve_arq_dir() -> Path:
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

    _shp_candidates = sorted(ARQ_DIR.glob("BR_Municipios_*/*.shp"))
    SHP_PATH = str(_shp_candidates[-1]) if _shp_candidates else str(ARQ_DIR / "BR_Municipios_2025" / "BR_Municipios_2025.shp")

    estacoes_parquet_path = DATA_DIR / "estacoes.parquet"
    log_parquet_path = DATA_DIR / "log.parquet"
    estacoes_size_mb = estacoes_parquet_path.stat().st_size / (1024**2)
    log_size_mb = log_parquet_path.stat().st_size / (1024**2)
    estacoes_mtime_str = datetime.fromtimestamp(estacoes_parquet_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")

    estacoes = pd.read_parquet(estacoes_parquet_path)
    log_raw_df = pd.read_parquet(log_parquet_path)
    ibge_mun = pd.read_csv(ARQ_DIR / "municipios.csv", dtype=str)

    # Parse factorized JSON log templates dynamically
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


@app.cell(hide_code=True)
def _(
    OFFICIAL_UFS,
    estacoes,
    estacoes_size_mb,
    log_events,
    log_templates,
    mult_num,
):
    # Build column completeness & sentinel '-1' profile dynamically
    _n_rows = max(len(estacoes), 1)
    _n_cols = len(estacoes.columns)
    _profile_rows = []
    for _col in estacoes.columns:
        _s = estacoes[_col]
        if str(_s.dtype) == "category":
            _n_sent = int((_s == "-1").sum()) if "-1" in _s.cat.categories else 0
            _vc = _s.value_counts()
            _top_val = str(_vc.index[0]) if len(_vc) > 0 else ""
        else:
            _n_sent = int((_s == -1).sum())
            _modes = _s.mode()
            _top_val = str(_modes.iloc[0]) if len(_modes) > 0 else ""
        if len(_top_val) > 42:
            _top_val = _top_val[:39] + "..."
        _valid_pct = (1.0 - _n_sent / _n_rows) * 100.0
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

    _mun_missing = int((estacoes["Município"].astype(str) == "-1").sum())
    _mun_filled_pct = (1.0 - _mun_missing / _n_rows) * 100.0
    _uf_vals = set(estacoes["UF"].astype(str).unique())
    _valid_ufs_count = len(_uf_vals & OFFICIAL_UFS)
    _dirty_ufs = sorted(_uf_vals - OFFICIAL_UFS - {"-1"})
    _uf_missing = int((estacoes["UF"].astype(str) == "-1").sum())
    _low_fill_count = int((schema_profile_df["Preenchido_%"] < 50.0).sum())
    _srd_share_pct = float(estacoes["Fonte"].astype(str).isin(["MOSAICO-SRD", "SRD"]).mean() * 100.0)

    _uf_status_txt = (
        f"nenhuma UF fora do padrão IBGE ({_valid_ufs_count} UFs oficiais presentes e **{_uf_missing:,}** registros offshore/fronteira com `UF == '-1'`)"
        if not _dirty_ufs
        else f"**{len(_dirty_ufs)}** UFs anômalas detectadas (`{_dirty_ufs}`) além de {_valid_ufs_count} UFs oficiais e **{_uf_missing:,}** com `'-1'`"
    )

    _kpis = mo.hstack(
        [
            mo.stat(
                label="Registros Consolidados",
                value=f"{len(estacoes):,}",
                caption=f"{estacoes_size_mb:.1f} MB Parquet ({_n_cols} colunas)",
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
                value=f"{_mun_filled_pct:.2f}%",
                caption=f"{_mun_missing:,} registros com sentinela '-1' ({100.0 - _mun_filled_pct:.2f}%)",
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
            tooltip=["Coluna", "Preenchido_%", alt.Tooltip("Sentinela_-1:Q", format=","), alt.Tooltip("Valores_Unicos:Q", format=","), "Valor_Mais_Comum"],
        )
        .properties(height=500, title=f"Completude por Coluna ({len(estacoes):,} registros | Sentinela '-1' = Ausente)")
    )

    mo.vstack(
        [
            mo.md(
                f"""
    ## 1. Visão Geral & Completude do Esquema
    > **Síntese da extração atual:** A base consolidada soma **{len(estacoes):,} registros** (**{int(mult_num.sum()):,} emissores físicos** considerando `Multiplicidade`). O campo `Município` está preenchido em **{_mun_filled_pct:.2f}%** da base (restando **{_mun_missing:,}** registros com `"-1"`), com {_uf_status_txt}. Atualmente **{_low_fill_count} de {_n_cols} colunas** apresentam preenchimento inferior a 50% — concentradas nos atributos técnicos de engenharia exclusivos das fontes de Radiodifusão (`MOSAICO-SRD` e `SRD`, que representam **{_srd_share_pct:.2f}%** dos registros).
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
def _(OFFICIAL_UFS, SERVICO_NOMES, estacoes, freq_mhz, mult_num):
    # Spectrum bands, Fontes, Serviços, UFs, Top Entidades
    _n_total = max(len(estacoes), 1)
    _emit_total = max(float(mult_num.sum()), 1.0)

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

    _fonte_df = (
        pd.DataFrame({"Fonte": estacoes["Fonte"].astype(str), "Emissores": mult_num})
        .groupby("Fonte", observed=True)
        .agg(Registros=("Emissores", "size"), Emissores_Fisicos=("Emissores", "sum"))
        .sort_values("Registros", ascending=False)
        .head(12)
        .reset_index()
    )
    _fonte_df["%_Registros"] = (_fonte_df["Registros"] / _n_total * 100.0).round(2)

    _serv_all_df = (
        pd.DataFrame({"Serviço": estacoes["Serviço"].astype(str), "Emissores": mult_num, "Freq_MHz": freq_mhz})
        .groupby("Serviço", observed=True)
        .agg(
            Registros=("Emissores", "size"),
            Emissores_Fisicos=("Emissores", "sum"),
            Freq_Mediana_MHz=("Freq_MHz", "median"),
        )
        .sort_values("Registros", ascending=False)
        .reset_index()
    )
    _serv_all_df["Descricao"] = _serv_all_df["Serviço"].map(lambda c: SERVICO_NOMES.get(c, f"Serviço {c}"))
    _serv_df = _serv_all_df.head(12).copy()

    _top1_s = _serv_df.iloc[0]
    _top2_s = _serv_df.iloc[1] if len(_serv_df) > 1 else _top1_s
    _top1_pct = float(_top1_s["Registros"] / _n_total * 100.0)
    _top2_pct = float(_top2_s["Registros"] / _n_total * 100.0) if len(_serv_df) > 1 else 0.0
    _top2_sum_pct = _top1_pct + _top2_pct
    _top2_emit_pct = float((_serv_df["Emissores_Fisicos"].iloc[:2].sum()) / _emit_total * 100.0)

    _sub3_pct = float(np.nanmean((freq_mhz > 0) & (freq_mhz <= 3000.0)) * 100.0)
    _shf_pct = float(np.nanmean(freq_mhz >= 6000.0) * 100.0)

    _uf_df = (
        pd.DataFrame({"UF": estacoes["UF"].astype(str), "Emissores": mult_num})
        .groupby("UF", observed=True)
        .agg(Registros=("Emissores", "size"), Emissores_Fisicos=("Emissores", "sum"))
        .sort_values("Registros", ascending=False)
        .reset_index()
    )
    _uf_df["Anomala"] = ~_uf_df["UF"].isin(OFFICIAL_UFS)

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
            x=alt.X("UF:N", sort="-y", title="Unidade da Federação"),
            y=alt.Y("Registros:Q", title="Registros"),
            color=alt.condition(
                alt.datum["Anomala"],
                alt.value("#e45756"),
                alt.value("#4c78a8"),
            ),
            tooltip=["UF", alt.Tooltip("Registros:Q", format=","), alt.Tooltip("Emissores_Fisicos:Q", format=",.0f"), "Anomala"],
        )
        .properties(height=250, title="Registros por UF (barras vermelhas = fora das 27 UFs oficiais)")
    )

    mo.vstack(
        [
            mo.md(
                f"""
    ## 2. Espectro, Serviços, Fontes & Operadoras
    - **Concentração por Serviço:** `{_top1_s['Descricao']}` (**{_top1_pct:.1f}%**, {int(_top1_s['Registros']):,} linhas) e `{_top2_s['Descricao']}` (**{_top2_pct:.1f}%**, {int(_top2_s['Registros']):,} linhas) concentram juntos **{_top2_sum_pct:.1f}%** dos registros e **{_top2_emit_pct:.1f}%** dos **{int(mult_num.sum()):,}** emissores físicos.
    - **Perfil Espectral:** **{_sub3_pct:.1f}%** das frequências válidas operam abaixo de 3 GHz (rádio privativo VHF/UHF, radiodifusão e cobertura móvel 2G/3G/4G) e **{_shf_pct:.1f}%** operam acima de 6 GHz (enlaces ponto-a-ponto SHF/EHF de *backhaul*).
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
):
    # Summary of pipelines + atomic event breakdown (deduplicated per station!) + SMP displacement
    pipe_summary_df = (
        log_templates.groupby(["Pipeline", "Alerta_Qualidade"], as_index=False)
        .agg(
            Templates=("Log", "count"),
            Estacoes_Afetadas=("Estacoes_Afetadas", "sum"),
            Exemplo_Log_ID=("Log", "min"),
        )
        .sort_values("Estacoes_Afetadas", ascending=False)
    )
    pipe_summary_df["%_da_Base"] = (pipe_summary_df["Estacoes_Afetadas"] / max(len(estacoes), 1) * 100).round(3)

    # Deduplicate (Log, Tag) before summing Estacoes_Afetadas so multi-column logs (#Latitude + #Longitude) are not double-counted!
    _tmpl_station_map = log_templates.set_index("Log")["Estacoes_Afetadas"]
    _dedup_tag_log = (
        log_events.groupby(["Tag", "Alerta_Qualidade", "Log"], as_index=False)
        .agg(
            Colunas_Afetadas=("Coluna", "count"),
            Exemplo_Processamento=("Processamento", "first"),
        )
    )
    _dedup_tag_log["Estacoes_Unicas"] = _dedup_tag_log["Log"].map(_tmpl_station_map).fillna(0).astype(int)
    _dedup_tag_log["Eventos_Coluna"] = _dedup_tag_log["Estacoes_Unicas"] * _dedup_tag_log["Colunas_Afetadas"]

    tag_summary_df = (
        _dedup_tag_log.groupby(["Tag", "Alerta_Qualidade"], as_index=False)
        .agg(
            Templates_com_Evento=("Log", "nunique"),
            Estacoes_Unicas=("Estacoes_Unicas", "sum"),
            Eventos_Coluna=("Eventos_Coluna", "sum"),
            Processamento=("Exemplo_Processamento", "first"),
        )
        .sort_values("Estacoes_Unicas", ascending=False)
    )

    _tag_stations = dict(zip(tag_summary_df["Tag"], tag_summary_df["Estacoes_Unicas"]))
    _tag_col_events = dict(zip(tag_summary_df["Tag"], tag_summary_df["Eventos_Coluna"]))

    # Compute true SMP tower -> municipality centroid shift dynamically
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
    _v_smp = (
        _smp_cur["Orig_Lat"].notna()
        & _smp_cur["Orig_Lon"].notna()
        & (_smp_cur["Orig_Lat"].abs() <= 90)
        & (_smp_cur["Orig_Lon"].abs() <= 180)
    )
    _smp_cur = _smp_cur[_v_smp].copy()
    if len(_smp_cur) > 0:
        _la1, _lo1 = np.radians(_smp_cur["Orig_Lat"]), np.radians(_smp_cur["Orig_Lon"])
        _la2, _lo2 = np.radians(_smp_cur["Mun_Lat"]), np.radians(_smp_cur["Mun_Lon"])
        _a_smp = np.sin((_la2 - _la1) / 2.0) ** 2 + np.cos(_la1) * np.cos(_la2) * np.sin((_lo2 - _lo1) / 2.0) ** 2
        _smp_cur["Deslocamento_km"] = np.round(2.0 * 6371.0 * np.arcsin(np.sqrt(np.clip(_a_smp, 0.0, 1.0))), 2)
        _smp_med_km = float(_smp_cur["Deslocamento_km"].median())
        _smp_p90_km = float(_smp_cur["Deslocamento_km"].quantile(0.9))
        _smp_max_km = float(_smp_cur["Deslocamento_km"].max())
    else:
        _smp_cur["Deslocamento_km"] = pd.Series(dtype="float64")
        _smp_med_km = _smp_p90_km = _smp_max_km = 0.0
    _smp_shift_top = _smp_cur.sort_values("Deslocamento_km", ascending=False).head(30)

    # Filtered events view
    _ev_view = log_events
    if log_only_dq.value:
        _ev_view = _ev_view[_ev_view["Alerta_Qualidade"]]
    if log_tag_filter.value != "(Todos os Eventos)":
        _ev_view = _ev_view[_ev_view["Tag"] == log_tag_filter.value]

    # Selected Log ID inspection + join with estacoes
    _sel_lid = min(max(0, int(log_id_selector.value)), max(0, len(log_templates) - 1))
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
        alt.Chart(tag_summary_df)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("Estacoes_Unicas:Q", title="Estações Únicas Afetadas (escala symlog)", scale=alt.Scale(type="symlog")),
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
        .properties(height=360, title=f"Estações Únicas por Tipo de Evento em log.parquet ({len(log_templates):,} Templates)")
    )

    _n_norm = int(_tag_stations.get("IBGE_NOME_NORMALIZADO", 0))
    _n_div_st = int(_tag_stations.get("IBGE_COORD_DIVERGENTE", 0))
    _n_div_ev = int(_tag_col_events.get("IBGE_COORD_DIVERGENTE", 0))
    _n_poly_rec = int(_tag_stations.get("IBGE_POLIGONO_RECUPERADO", 0))
    _n_auto_fix = int(_tag_stations.get("COORD_SINAL_OU_TROCA_CORRIGIDA", 0)) + int(_tag_stations.get("COORD_ABSURDA_SUBSTITUIDA_IBGE", 0))
    _n_null_fill = int(_tag_stations.get("COORD_NULA_PREENCHIDA_IBGE", 0))

    mo.vstack(
        [
            mo.md(
                f"""
    ## 3. Explorador Estruturado de `log.parquet` (Rastreabilidade Antes → Depois)
    O dicionário fatorado `log.parquet` contém **{len(log_templates):,} templates distintos** (**{len(log_events):,} eventos atômicos**), permitindo auditar cada transformação aplicada às **{len(estacoes):,}** estações:

    - **`IBGE_NOME_NORMALIZADO`:** **{_n_norm:,} estações** tiveram nome de `Município` e/ou `UF` padronizados ou preenchidos pela tabela oficial do IBGE.
    - **`IBGE_COORD_DIVERGENTE`:** **{_n_div_st:,} estações únicas** ({_n_div_ev:,} substituições de `#Latitude`/`#Longitude`) apresentaram coordenada em polígono municipal distinto do `Código_Município` declarado — analisadas em detalhe na **Seção 4** abaixo.
    - **Recuperação Geoespacial Automática:** **{_n_poly_rec:,} estações** sem código IBGE recuperaram `Código_Município`/`Município`/`UF` via interseção com o polígono territorial (`IBGE_POLIGONO_RECUPERADO`), **{_n_auto_fix:,} estações** tiveram sinal/troca de coordenadas ou Null Island corrigidos automaticamente, e **{_n_null_fill:,} estações** sem coordenadas receberam o centroide municipal.
    - **Deslocamento ERB → Centroide no `MOSAICO-SMP`:** Em **{len(_smp_cur):,} templates SMP** agrupados por município, o deslocamento mediano entre a coordenada original da ERB e o centroide municipal é de **{_smp_med_km:.2f} km** (p90 = **{_smp_p90_km:.2f} km**, máx = **{_smp_max_km:,.1f} km**).
    """
            ),
            mo.ui.tabs(
                {
                    "🔀 Pipelines & Gráfico de Eventos": mo.vstack(
                        [
                            _tag_chart,
                            mo.md("### Resumo por Tag (Estações Únicas vs. Eventos de Coluna)"),
                            mo.ui.table(tag_summary_df, pagination=False),
                            mo.md("### Top 25 Sequências de Processamento (`Pipeline`)"),
                            mo.ui.table(pipe_summary_df.head(25), pagination=True, page_size=15),
                        ]
                    ),
                    "📏 Deslocamento Real ERB → Município (SMP)": mo.vstack(
                        [
                            mo.md("Distância geodésica entre a coordenada original da torre (`Original` preservado no `Log`) e o centroide municipal atribuído no agrupamento SMP:"),
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
def _(SHP_PATH, cat_to_float, estacoes, log_events):
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
        "5_Nao_Vizinho_Mesma_UF": (
            "#d73027",
            "5. Município Não-Vizinho na Mesma UF",
        ),
        "6_Outra_UF_Nao_Vizinho": (
            "#762a83",
            "6. Outra UF Não-Limítrofe (Erro Cadastral / Digitação)",
        ),
    }


    @mo.persistent_cache
    def compute_divergent_forensics(unique_pairs: pd.DataFrame, shp_path_str: str) -> pd.DataFrame:
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

        regions = gpd.read_file(shp_path_str, columns=["CD_MUN", "NM_MUN", "SIGLA_UF", "geometry"])
        regions["CD_MUN"] = regions["CD_MUN"].astype(str)

        pts_wgs = shapely.points(unique_pairs["Orig_Lon"].to_numpy(), unique_pairs["Orig_Lat"].to_numpy())
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
            dist_km[valid_g] = shapely.distance(pts_m[valid_g], decl_geoms[valid_g]) / 1000.0
            vizinho[valid_g] = shapely.intersects(decl_geoms[valid_g], real_geoms[valid_g])

        res["Dist_Borda_km"] = dist_km
        res["Vizinho"] = vizinho
        return res


    _div_ev = log_events[log_events["Tag"] == "IBGE_COORD_DIVERGENTE"]
    _orig_lat_map = _div_ev[_div_ev["Coluna"] == "#Latitude"].drop_duplicates("Log").set_index("Log")["Original"]
    _orig_lon_map = _div_ev[_div_ev["Coluna"] == "#Longitude"].drop_duplicates("Log").set_index("Log")["Original"]
    _div_lids = _orig_lat_map.index.union(_orig_lon_map.index)

    _div_base = estacoes[estacoes["Log"].isin(_div_lids)].copy()
    _div_base["Cod_Decl"] = _div_base["Código_Município"].astype(str)
    _div_base["UF_Decl"] = _div_base["UF"].astype(str)
    _div_base["Mun_Decl"] = _div_base["Município"].astype(str)
    _div_base["New_Lat"] = pd.to_numeric(_div_base["Latitude"].astype(str), errors="coerce")
    _div_base["New_Lon"] = pd.to_numeric(_div_base["Longitude"].astype(str), errors="coerce")
    _div_base["Orig_Lat"] = pd.to_numeric(_div_base["Log"].map(_orig_lat_map), errors="coerce").fillna(_div_base["New_Lat"])
    _div_base["Orig_Lon"] = pd.to_numeric(_div_base["Log"].map(_orig_lon_map), errors="coerce").fillna(_div_base["New_Lon"])
    _div_base["Mult_Num"] = np.nan_to_num(cat_to_float(_div_base["Multiplicidade"]), nan=1.0)

    _u_div_pairs = (
        _div_base[["Cod_Decl", "Orig_Lat", "Orig_Lon"]]
        .dropna()
        .drop_duplicates()
        .reset_index(drop=True)
    )
    _forensic_lkp = compute_divergent_forensics(_u_div_pairs, SHP_PATH)

    div_records_df = _div_base.merge(_forensic_lkp, on=["Cod_Decl", "Orig_Lat", "Orig_Lon"], how="left")
    if len(div_records_df) > 0:
        _dla1, _dlo1 = np.radians(div_records_df["Orig_Lat"]), np.radians(div_records_df["Orig_Lon"])
        _dla2, _dlo2 = np.radians(div_records_df["New_Lat"]), np.radians(div_records_df["New_Lon"])
        _da = np.sin((_dla2 - _dla1) / 2.0) ** 2 + np.cos(_dla1) * np.cos(_dla2) * np.sin((_dlo2 - _dlo1) / 2.0) ** 2
        div_records_df["Shift_Centroid_km"] = 2.0 * 6371.0 * np.arcsin(np.sqrt(np.clip(_da, 0.0, 1.0)))
        div_records_df["Mesma_UF"] = div_records_df["UF_Decl"] == div_records_df["Real_UF"].astype(str)
        div_records_df["Vizinho"] = div_records_df["Vizinho"].fillna(False).astype(bool)

        _r_conds = [
            div_records_df["Dist_Borda_km"] <= 0.1,
            div_records_df["Dist_Borda_km"] <= 1.0,
            div_records_df["Vizinho"] & (div_records_df["Dist_Borda_km"] <= 10.0),
            div_records_df["Vizinho"] & (div_records_df["Dist_Borda_km"] > 10.0),
            (~div_records_df["Vizinho"]) & div_records_df["Mesma_UF"],
        ]
        _r_codes = [
            "1_Micro_Fronteira_100m",
            "2_Topo_Serra_Borda_1km",
            "3_Limitrofe_1_10km",
            "4_Limitrofe_Maior_10km",
            "5_Nao_Vizinho_Mesma_UF",
        ]
        div_records_df["Regime_Code"] = np.select(_r_conds, _r_codes, default="6_Outra_UF_Nao_Vizinho")
        div_records_df["Regime_Desc"] = div_records_df["Regime_Code"].map(lambda k: DIVERGENT_REGIME_META[k][1])

        div_sites_df = (
            div_records_df.groupby(
                [
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
                ],
                observed=True,
                as_index=False,
            )
            .agg(
                Canais_Afetados=("Log", "size"),
                Emissores_Fisicos=("Mult_Num", "sum"),
                Estacoes_IDs=("Estação", lambda s: int(s.astype(str).nunique())),
                Dist_Borda_km=("Dist_Borda_km", "first"),
                Shift_Centroid_km=("Shift_Centroid_km", "first"),
                Entidade_Principal=("Entidade", lambda s: str(s.astype(str).value_counts().index[0])),
                Fonte_Principal=("Fonte", lambda s: str(s.astype(str).value_counts().index[0])),
                Servico_Principal=("Serviço", lambda s: str(s.astype(str).value_counts().index[0])),
                Exemplo_Estacao=("Estação", lambda s: str(s.astype(str).iloc[0])),
                Exemplo_Log_ID=("Log", "min"),
            )
            .sort_values("Canais_Afetados", ascending=False)
            .reset_index(drop=True)
        )
    else:
        div_records_df["Shift_Centroid_km"] = pd.Series(dtype="float64")
        div_records_df["Mesma_UF"] = pd.Series(dtype=bool)
        div_records_df["Regime_Code"] = pd.Series(dtype=str)
        div_records_df["Regime_Desc"] = pd.Series(dtype=str)
        div_sites_df = pd.DataFrame(
            columns=[
                "Cod_Decl", "UF_Decl", "Mun_Decl", "Orig_Lat", "Orig_Lon", "New_Lat", "New_Lon",
                "Real_CD_MUN", "Real_NM_MUN", "Real_UF", "Vizinho", "Mesma_UF", "Regime_Code",
                "Regime_Desc", "Canais_Afetados", "Emissores_Fisicos", "Estacoes_IDs",
                "Dist_Borda_km", "Shift_Centroid_km", "Entidade_Principal", "Fonte_Principal",
                "Servico_Principal", "Exemplo_Estacao", "Exemplo_Log_ID",
            ]
        )

    _reg_rows = []
    for _rk, (_rc, _rdesc) in DIVERGENT_REGIME_META.items():
        _sub_r = div_records_df[div_records_df["Regime_Code"] == _rk]
        _sub_s = div_sites_df[div_sites_df["Regime_Code"] == _rk]
        _nr = len(_sub_r)
        _reg_rows.append(
            {
                "Regime_Code": _rk,
                "Regime": _rdesc,
                "Registros": _nr,
                "%_Divergentes": round(_nr / max(len(div_records_df), 1) * 100.0, 2),
                "Sites_Fisicos_Unicos": len(_sub_s),
                "Dist_Divisa_Mediana_km": round(float(_sub_r["Dist_Borda_km"].median()), 3) if _nr > 0 else 0.0,
                "Shift_Centroide_Mediano_km": round(float(_sub_r["Shift_Centroid_km"].median()), 2) if _nr > 0 else 0.0,
            }
        )
    div_regime_summary_df = pd.DataFrame(_reg_rows)

    return (
        DIVERGENT_REGIME_META,
        div_records_df,
        div_regime_summary_df,
        div_sites_df,
    )


@app.cell(hide_code=True)
def _(div_regime_summary_df, div_sites_df):
    # Interactive filters for Section 4 (IBGE_COORD_DIVERGENTE investigation)
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

    _div_uf_list = ["(Todas)"] + sorted(div_sites_df["UF_Decl"].astype(str).unique().tolist())
    div_uf_select = mo.ui.dropdown(options=_div_uf_list, value="(Todas)", label="UF Declarada:")

    _div_fonte_list = ["(Todas)"] + sorted(div_sites_df["Fonte_Principal"].astype(str).unique().tolist())
    div_fonte_select = mo.ui.dropdown(options=_div_fonte_list, value="(Todas)", label="Fonte Principal:")

    div_entity_search = mo.ui.text(
        value="",
        placeholder="Ex: POLICIA MILITAR, TELEFONICA, VALE, Nova Lima...",
        label="Buscar Entidade ou Município:",
    )

    div_max_sites = mo.ui.slider(
        start=100,
        stop=2000,
        step=100,
        value=600,
        label="Máx. Sites Físicos no Mapa:",
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
    DIVERGENT_REGIME_META,
    SERVICO_NOMES,
    div_entity_search,
    div_fonte_select,
    div_max_sites,
    div_records_df,
    div_regime_select,
    div_regime_summary_df,
    div_show_vectors,
    div_sites_df,
    div_uf_select,
    estacoes,
):
    # Section 4: Interactive Forensics Dashboard & Before -> After Vector Map for IBGE_COORD_DIVERGENTE
    _n_div = len(div_records_df)
    _n_sites = len(div_sites_df)
    _div_pct_base = _n_div / max(len(estacoes), 1) * 100.0
    _div_emit = int(div_records_df["Mult_Num"].sum()) if _n_div > 0 else 0
    _avg_ch_per_site = _n_div / max(_n_sites, 1)
    _max_ch_site = int(div_sites_df["Canais_Afetados"].max()) if _n_sites > 0 else 0

    _n_100m = int((div_records_df["Dist_Borda_km"] <= 0.1).sum()) if _n_div > 0 else 0
    _n_1km = int((div_records_df["Dist_Borda_km"] <= 1.0).sum()) if _n_div > 0 else 0
    _pct_1km = _n_1km / max(_n_div, 1) * 100.0
    _n_viz = int(div_records_df["Vizinho"].sum()) if _n_div > 0 else 0
    _pct_viz = _n_viz / max(_n_div, 1) * 100.0
    _n_non_viz = _n_div - _n_viz
    _pct_non_viz = _n_non_viz / max(_n_div, 1) * 100.0

    _med_borda = float(div_records_df["Dist_Borda_km"].median()) if _n_div > 0 else 0.0
    _med_shift = float(div_records_df["Shift_Centroid_km"].median()) if _n_div > 0 else 0.0
    _pct_shift_worse = (
        float((div_records_df["Shift_Centroid_km"] > div_records_df["Dist_Borda_km"]).mean() * 100.0)
        if _n_div > 0
        else 0.0
    )

    # Dynamic concentration metrics (Heavy multi-channel sites, top Fonte, Serviço, UF, Entidade)
    _heavy_sites = div_sites_df[div_sites_df["Canais_Afetados"] >= 50]
    _n_heavy_sites = len(_heavy_sites)
    _n_heavy_rows = int(_heavy_sites["Canais_Afetados"].sum()) if _n_heavy_sites > 0 else 0
    _pct_heavy_rows = _n_heavy_rows / max(_n_div, 1) * 100.0

    if _n_div > 0:
        _vc_fonte = div_records_df["Fonte"].astype(str).value_counts()
        _top_div_fonte, _top_div_fonte_n = str(_vc_fonte.index[0]), int(_vc_fonte.iloc[0])
        _top_div_fonte_pct = _top_div_fonte_n / _n_div * 100.0

        _vc_serv = div_records_df["Serviço"].astype(str).value_counts()
        _top_div_serv_code = str(_vc_serv.index[0])
        _top_div_serv_lbl = SERVICO_NOMES.get(_top_div_serv_code, f"Serviço {_top_div_serv_code}")
        _top_div_serv_pct = int(_vc_serv.iloc[0]) / _n_div * 100.0

        _vc_uf = div_records_df["UF_Decl"].value_counts()
        _top_div_uf, _top_div_uf_n = str(_vc_uf.index[0]), int(_vc_uf.iloc[0])
        _top_div_uf_pct = _top_div_uf_n / _n_div * 100.0

        _vc_ent = div_records_df["Entidade"].astype(str).value_counts()
        _top_div_ent, _top_div_ent_n = str(_vc_ent.index[0]), int(_vc_ent.iloc[0])
        _top_div_ent_sites = int(
            div_records_df.loc[div_records_df["Entidade"].astype(str) == _top_div_ent, ["Orig_Lat", "Orig_Lon"]]
            .drop_duplicates()
            .shape[0]
        )
    else:
        _top_div_fonte, _top_div_fonte_n, _top_div_fonte_pct = "N/A", 0, 0.0
        _top_div_serv_lbl, _top_div_serv_pct = "N/A", 0.0
        _top_div_uf, _top_div_uf_n, _top_div_uf_pct = "N/A", 0, 0.0
        _top_div_ent, _top_div_ent_n, _top_div_ent_sites = "N/A", 0, 0

    _div_kpis = mo.hstack(
        [
            mo.stat(
                label="Estações c/ Coord. Divergente",
                value=f"{_n_div:,}",
                caption=f"{_div_pct_base:.2f}% da base ({_div_emit:,} emissores)",
            ),
            mo.stat(
                label="Sites Físicos Únicos (Coord.)",
                value=f"{_n_sites:,}",
                caption=f"Média {_avg_ch_per_site:.1f} canais/site (máx {_max_ch_site:,})",
            ),
            mo.stat(
                label="Na Divisa / Topo de Serra (≤ 1 km)",
                value=f"{_n_1km:,} ({_pct_1km:.1f}%)",
                caption=f"{_n_100m:,} registros a ≤ 100 m da fronteira",
            ),
            mo.stat(
                label="Em Município Limítrofe",
                value=f"{_n_viz:,} ({_pct_viz:.1f}%)",
                caption=f"Mediana {_med_borda:.2f} km da divisa → {_med_shift:.2f} km p/ centroide",
            ),
        ],
        widths="equal",
    )

    # Charts for Section 4
    _chart_regimes = (
        alt.Chart(div_regime_summary_df)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("Registros:Q", title="Estações Afetadas"),
            y=alt.Y("Regime:N", sort=list(DIVERGENT_REGIME_META[k][1] for k in DIVERGENT_REGIME_META), title=None),
            color=alt.Color(
                "Regime_Code:N",
                scale=alt.Scale(
                    domain=list(DIVERGENT_REGIME_META.keys()),
                    range=[v[0] for v in DIVERGENT_REGIME_META.values()],
                ),
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

    _div_ent_top = (
        div_records_df.groupby("Entidade", observed=True)
        .agg(
            Registros=("Log", "size"),
            Sites_Unicos=("Orig_Lat", lambda s: int(div_records_df.loc[s.index, ["Orig_Lat", "Orig_Lon"]].drop_duplicates().shape[0])),
            Dist_Divisa_Med_km=("Dist_Borda_km", "median"),
        )
        .sort_values("Registros", ascending=False)
        .head(12)
        .reset_index()
        if _n_div > 0
        else pd.DataFrame(columns=["Entidade", "Registros", "Sites_Unicos", "Dist_Divisa_Med_km"])
    )
    _div_ent_top["Entidade_Curta"] = _div_ent_top["Entidade"].astype(str).str.slice(0, 38)

    _chart_div_ent = (
        alt.Chart(_div_ent_top)
        .mark_bar(color="#e6550d", cornerRadiusEnd=3)
        .encode(
            x=alt.X("Registros:Q", title="Registros Divergentes"),
            y=alt.Y("Entidade_Curta:N", sort="-x", title="Top 12 Entidades"),
            tooltip=[
                "Entidade",
                alt.Tooltip("Registros:Q", format=","),
                alt.Tooltip("Sites_Unicos:Q", format=",", title="Sites Físicos Únicos"),
                alt.Tooltip("Dist_Divisa_Med_km:Q", format=".3f", title="Dist. Mediana Divisa (km)"),
            ],
        )
        .properties(width=340, height=250, title="Concentração por Entidade (Canais vs. Sites)")
    )

    _div_pairs_top = (
        div_records_df.groupby(["UF_Decl", "Mun_Decl", "Real_UF", "Real_NM_MUN", "Vizinho"], observed=True, as_index=False)
        .agg(
            Registros=("Log", "size"),
            Sites_Unicos=("Orig_Lat", "nunique"),
            Dist_Divisa_Med_km=("Dist_Borda_km", "median"),
            Shift_Centroide_Med_km=("Shift_Centroid_km", "median"),
            Entidade_Principal=("Entidade", lambda s: str(s.astype(str).value_counts().index[0])),
        )
        .sort_values("Registros", ascending=False)
        .head(40)
        if _n_div > 0
        else pd.DataFrame()
    )
    if len(_div_pairs_top) > 0:
        _div_pairs_top["Dist_Divisa_Med_km"] = _div_pairs_top["Dist_Divisa_Med_km"].round(3)
        _div_pairs_top["Shift_Centroide_Med_km"] = _div_pairs_top["Shift_Centroide_Med_km"].round(2)

    # Filter physical sites for the interactive Before -> After Folium Map
    _sel_reg_codes = set(div_regime_select.value)
    _s_mask = div_sites_df["Regime_Code"].isin(_sel_reg_codes)
    _s_mask &= (
        div_sites_df["Orig_Lat"].notna()
        & div_sites_df["Orig_Lon"].notna()
        & (div_sites_df["Orig_Lat"].abs() <= 90.0)
        & (div_sites_df["Orig_Lon"].abs() <= 180.0)
    )
    if div_uf_select.value != "(Todas)":
        _s_mask &= div_sites_df["UF_Decl"] == div_uf_select.value
    if div_fonte_select.value != "(Todas)":
        _s_mask &= div_sites_df["Fonte_Principal"] == div_fonte_select.value
    if div_entity_search.value.strip():
        _dq = div_entity_search.value.strip().lower()
        _s_mask &= (
            div_sites_df["Entidade_Principal"].str.lower().str.contains(_dq, regex=False)
            | div_sites_df["Mun_Decl"].str.lower().str.contains(_dq, regex=False)
            | div_sites_df["Real_NM_MUN"].str.lower().str.contains(_dq, regex=False)
        )

    _filtered_sites = div_sites_df[_s_mask].copy()
    _total_filtered_sites = len(_filtered_sites)
    _total_filtered_rows = int(_filtered_sites["Canais_Afetados"].sum()) if _total_filtered_sites > 0 else 0
    _plot_sites = _filtered_sites.head(int(div_max_sites.value))

    _div_map = folium.Map(location=[-16.5, -47.5], zoom_start=5, tiles="OpenStreetMap", control_scale=True)
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr="&copy; OpenStreetMap contributors &copy; CARTO",
        name="Mapa Claro (CARTO)",
        subdomains="abcd",
    ).add_to(_div_map)

    _reg_layers = {}
    for _rk in sorted(_sel_reg_codes):
        _rc, _rdesc = DIVERGENT_REGIME_META[_rk]
        _reg_layers[_rk] = folium.FeatureGroup(name=_rdesc, show=True).add_to(_div_map)

    for _sr in _plot_sites.itertuples():
        _rc, _rdesc = DIVERGENT_REGIME_META[_sr.Regime_Code]
        _ola, _olo = float(_sr.Orig_Lat), float(_sr.Orig_Lon)
        _nla, _nlo = float(_sr.New_Lat), float(_sr.New_Lon)
        _db_str = (
            f"{_sr.Dist_Borda_km * 1000.0:,.0f} m"
            if pd.notna(_sr.Dist_Borda_km) and _sr.Dist_Borda_km < 1.0
            else (f"{_sr.Dist_Borda_km:,.2f} km" if pd.notna(_sr.Dist_Borda_km) else "N/A")
        )
        _sh_str = f"{_sr.Shift_Centroid_km:,.2f} km" if pd.notna(_sr.Shift_Centroid_km) else "N/A"
        _radius = float(min(12.0, max(4.5, 3.5 + np.log1p(_sr.Canais_Afetados) * 1.3)))

        _tip = (
            f"<div style='font-family:sans-serif;font-size:12px;min-width:290px;max-width:380px;'>"
            f"<div style='background:{_rc};color:#fff;padding:4px 7px;border-radius:4px;font-weight:bold;margin-bottom:5px;'>"
            f"{html.escape(_rdesc)}</div>"
            f"<b>Canais neste Site:</b> {_sr.Canais_Afetados:,} (Σ Mult: {int(_sr.Emissores_Fisicos):,}) | "
            f"<b>Estação ex.:</b> {html.escape(str(_sr.Exemplo_Estacao))}<br>"
            f"<b>Entidade:</b> {html.escape(str(_sr.Entidade_Principal))}<br>"
            f"<b>Fonte:</b> {html.escape(str(_sr.Fonte_Principal))} | <b>Serviço:</b> {html.escape(str(_sr.Servico_Principal))} | "
            f"<b>Log ID:</b> {_sr.Exemplo_Log_ID}<br>"
            f"<hr style='margin:4px 0'>"
            f"<b>Município Declarado:</b> {html.escape(str(_sr.Mun_Decl))}/{html.escape(str(_sr.UF_Decl))} ({html.escape(str(_sr.Cod_Decl))})<br>"
            f"<b>Polígono da Coord. Original:</b> {html.escape(str(_sr.Real_NM_MUN))}/{html.escape(str(_sr.Real_UF))} ({html.escape(str(_sr.Real_CD_MUN))})<br>"
            f"<b>Coord. Original:</b> ({_ola:.6f}, {_olo:.6f})<br>"
            f"<b>Centroide IBGE Substituído:</b> ({_nla:.6f}, {_nlo:.6f})<br>"
            f"<b>Distância até Divisa Declarada:</b> <span style='color:#d95f02;font-weight:bold'>{_db_str}</span> | "
            f"<b>Deslocamento p/ Centroide:</b> {_sh_str}"
            f"</div>"
        )

        _grp = _reg_layers[_sr.Regime_Code]
        if div_show_vectors.value and pd.notna(_nla) and pd.notna(_nlo) and (abs(_nla) <= 90) and (abs(_nlo) <= 180):
            folium.PolyLine(
                locations=[[_ola, _olo], [_nla, _nlo]],
                color=_rc,
                weight=1.5,
                opacity=0.45,
                dash_array="4,4",
            ).add_to(_grp)

        folium.CircleMarker(
            location=[_ola, _olo],
            radius=_radius,
            color=_rc,
            weight=1.5,
            fill=True,
            fill_color=_rc,
            fill_opacity=0.82,
            tooltip=folium.Tooltip(_tip, sticky=True),
        ).add_to(_grp)

    folium.LayerControl(collapsed=True).add_to(_div_map)

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
    ].copy()
    _sites_display["Dist_Borda_km"] = _sites_display["Dist_Borda_km"].round(3)
    _sites_display["Shift_Centroid_km"] = _sites_display["Shift_Centroid_km"].round(2)
    _sites_display["Orig_Lat"] = _sites_display["Orig_Lat"].round(6)
    _sites_display["Orig_Lon"] = _sites_display["Orig_Lon"].round(6)

    mo.vstack(
        [
            mo.md(
                f"""
    ## 4. Investigação Especial — Por que há `{_n_div:,}` estações com `IBGE_COORD_DIVERGENTE`?
    A mensagem `"Coordenadas divergentes (Polígono do IBGE retorna município distinto). Coordenadas do Município do IBGE."` é disparada em `Geography.substitute_divergent_coordinates()` sempre que o ponto `(Latitude, Longitude)` cai em um polígono municipal do IBGE diferente do `Código_Município` declarado. A auditoria geoespacial das coordenadas originais (`#Latitude`, `#Longitude` preservadas no `Log`) revela **quatro fatores estruturais**:

    1. **Amplificação por multicanalização (`{_n_sites:,}` sites físicos → `{_n_div:,}` linhas):** Como o agrupamento preserva uma linha por `Frequência`, estações repetidoras multicanal multiplicam o evento. `{_top_div_fonte}` concentra **{_top_div_fonte_pct:.1f}%** dos casos ({_top_div_fonte_n:,} linhas) e `{_top_div_serv_lbl}` responde por **{_top_div_serv_pct:.1f}%**. Apenas **{_n_heavy_sites:,} sites físicos** com ≥50 frequências somam **{_n_heavy_rows:,} linhas ({_pct_heavy_rows:.1f}%)** — liderados em `{_top_div_uf}` (**{_top_div_uf_pct:.1f}%**, {_top_div_uf_n:,} linhas) por `{_top_div_ent}` (**{_top_div_ent_n:,} linhas** em apenas **{_top_div_ent_sites:,} coordenadas únicas**).
    2. **Efeito de Fronteira, Crista de Serra e Datum (`{_n_1km:,}` linhas = `{_pct_1km:.1f}%` a ≤ 1 km da divisa):** **{_n_100m:,} linhas** estão a **≤ 100 m** do limite municipal declarado (margem típica de deslocamento SAD69→SIRGAS2000 ~65 m, arredondamento de segundos DMS ~30 m e generalização da malha IBGE), e outras **{_n_1km - _n_100m:,}** entre **100 m e 1 km** (parques de antenas em topos de serra divisórios).
    3. **Conurbação e Plantas Limítrofes (`{_n_viz:,}` linhas = `{_pct_viz:.1f}%` em município vizinho):** A grande maioria ocorre entre municípios com fronteira comum (redes metropolitanas, mineradoras, usinas e fazendas intermunicipais). Como a regra atual não aplica *buffer* de tolerância na divisa, em **{_pct_shift_worse:.1f}%** dos registros o deslocamento imposto até o centroide municipal (mediana **{_med_shift:.2f} km**) é maior que a distância original até a divisa (mediana **{_med_borda:.2f} km**).
    4. **Erros Cadastrais Genuínos à Distância (`{_n_non_viz:,}` linhas = `{_pct_non_viz:.1f}%` em municípios não-vizinhos / outra UF):** Coordenadas de sede copiadas para licenças do interior ou erros de digitação em graus/minutos — casos em que substituir pelo centroide IBGE corrige distorções de dezenas a centenas de quilômetros.
    """
            ),
            _div_kpis,
            mo.ui.tabs(
                {
                    "📊 Regimes, Entidades & Pares": mo.vstack(
                        [
                            mo.hstack([_chart_regimes, _chart_div_ent], justify="start"),
                            mo.md("### Síntese Quantitativa por Regime Geoespacial"),
                            mo.ui.table(div_regime_summary_df, pagination=False),
                            mo.md("### Top Pares (Município Declarado → Polígono Real da Coordenada Original)"),
                            mo.ui.table(_div_pairs_top, pagination=True, page_size=12),
                        ]
                    ),
                    "🗺️ Mapa Antes → Depois (Vetores)": mo.vstack(
                        [
                            mo.hstack([div_uf_select, div_fonte_select, div_entity_search], widths="equal"),
                            mo.hstack([div_regime_select, mo.vstack([div_max_sites, div_show_vectors])], widths=[2, 1]),
                            mo.callout(
                                mo.md(
                                    f"Exibindo **{len(_plot_sites):,}** de **{_total_filtered_sites:,} sites físicos** filtrados "
                                    f"(representando **{_total_filtered_rows:,}** estações/canais). "
                                    f"Os círculos marcam a **coordenada original da torre** e as linhas tracejadas indicam o deslocamento até o **centroide IBGE**."
                                ),
                                kind="info",
                            ),
                            mo.iframe(_div_map.get_root().render(), width="100%", height="560px"),
                        ]
                    ),
                    "🗼 Tabela de Sites Físicos Filtrados": mo.vstack(
                        [
                            mo.md(f"Detalhamento dos **{_total_filtered_sites:,} sites físicos** (**{_total_filtered_rows:,} canais**) nos filtros atuais:"),
                            mo.ui.table(_sites_display, pagination=True, page_size=15),
                        ]
                    ),
                }
            ),
        ]
    )

    return


@app.cell
def _(SHP_PATH, cat_to_float, estacoes, ibge_mun, lat_num, lon_num):
    @mo.persistent_cache
    def compute_polygon_lookup(coords_frame: pd.DataFrame, shp_path_str: str) -> pd.DataFrame:
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


    _unique_coords = (
        estacoes[["Latitude", "Longitude"]]
        .astype(str)
        .drop_duplicates()
        .reset_index(drop=True)
    )
    coord_poly_lookup = compute_polygon_lookup(_unique_coords, SHP_PATH)

    # Map polygon results and IBGE centroid distances onto all consolidated rows
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
        "10_UF_Com_Espaco_ou_Divergente": ("#fdbf6f", "Erro String: UF com espaço ou divergente do IBGE"),
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

    _issue_counts = dict(zip(issue_summary_df["Codigo"], issue_summary_df["Registros"]))
    _n_ok = int(_issue_counts.get("00_OK_Consistente", 0))
    _pct_ok = _n_ok / max(len(estacoes), 1) * 100.0
    _n_offshore = int(_issue_counts.get("07_Offshore_Maritimo_Ilhas", 0))
    _n_poly_div_res = int(_issue_counts.get("05_Poligono_UF_Divergente", 0)) + int(_issue_counts.get("06_Poligono_Municipio_Divergente", 0))
    _n_crit_coord = (
        int(_issue_counts.get("01_Null_Island_ou_Polar_Invalido", 0))
        + int(_issue_counts.get("02_Lat_Lon_Invertidas", 0))
        + int(_issue_counts.get("03_Sinal_Invertido_Hemisferio", 0))
    )
    _n_foreign = int(_issue_counts.get("04_Fora_Fronteira_Terrestre", 0))
    _n_dirty_eng = int(_issue_counts.get("08_Texto_Sujo_Coluna_Numerica", 0))
    _n_phys_out = int(_issue_counts.get("09_Outlier_Fisico_Antena_Torre", 0))
    _n_uf_dirty = int(_issue_counts.get("10_UF_Com_Espaco_ou_Divergente", 0)) + int(_issue_counts.get("11_Sem_Codigo_Municipio_ou_UF", 0))
    _n_no_coord = int(_issue_counts.get("00_Sem_Coordenadas_-1", 0))

    _res_kpis = mo.hstack(
        [
            mo.stat(
                label="Consistentes c/ Malha IBGE",
                value=f"{_n_ok:,}",
                caption=f"{_pct_ok:.2f}% do total consolidado",
            ),
            mo.stat(
                label="Offshore / Costeiro",
                value=f"{_n_offshore:,}",
                caption="Plataformas marítimas, pré-sal e ilhas",
            ),
            mo.stat(
                label="Fronteira / Críticos no Mar",
                value=f"{_n_foreign + _n_crit_coord:,}",
                caption=f"{_n_foreign:,} fronteira/exterior + {_n_crit_coord:,} inversões/nulos",
            ),
            mo.stat(
                label="Resíduos Eng. / Sem Coord.",
                value=f"{_n_dirty_eng + _n_phys_out + _n_no_coord:,}",
                caption=f"{_n_dirty_eng:,} texto sujo | {_n_phys_out:,} outliers | {_n_no_coord:,} sem coord.",
            ),
        ],
        widths="equal",
    )

    _issue_chart_df = issue_summary_df[
        (issue_summary_df["Codigo"] != "00_OK_Consistente") & (issue_summary_df["Registros"] > 0)
    ]
    _issue_chart = (
        alt.Chart(_issue_chart_df)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("Registros:Q", title="Registros Residualmente Sinalizados (escala symlog)", scale=alt.Scale(type="symlog")),
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
        .properties(height=300, title="Panorama Atual de Registros Fora da Malha Terrestre ou com Alertas de Engenharia")
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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
