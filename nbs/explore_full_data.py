# /// script
# dependencies = [
#     "altair>=6.0.0",
#     "folium>=1.0.0",
#     "marimo",
#     "polars>=2.2.0",
#     "pyarrow>=18.0.0",
# ]
# requires-python = ">=3.13"
# ///

import marimo

__generated_with = "0.26.0"
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
    import marimo as mo
    import polars as pl


@app.cell(hide_code=True)
def header(DATA_DIR, full_mtime_str, full_size_mb, log_templates, n_rows):
    mo.md(f"""
    # 📡 RFDataHub — EDA da Base Completa (`estacoes_full.parquet`), Rastreabilidade do `Log` & Descartes
    **Diretório ativo:** `{DATA_DIR}` | **Última atualização:** `{full_mtime_str}` | **Tamanho:** {full_size_mb:.1f} MB | **Registros:** {n_rows:,} (um por emissor, sem agrupamento)

    Diferente de `estacoes.parquet` (agrupado + `log.parquet` fatorado), a base completa traz o histórico de processamento **embutido na coluna `Log`** de cada linha.
    Toda a análise é feita com **Polars em modo *lazy*** (`scan_parquet` + *projection/predicate pushdown*): nenhuma seção materializa a base inteira —
    os ~{len(log_templates):,} *templates* distintos de `Log` são decodificados uma única vez e as linhas são filtradas por `Log.is_in(...)`.

    1. **Visão Geral & Completude do Esquema** — preenchimento por coluna, sentinelas `"-1"` e perfil numérico das colunas convertidas.
    2. **Espectro, Serviços, Fontes & Operadoras** — faixas de frequência, serviços, UFs, fontes, entidades e validade das licenças.
    3. **Explorador do `Log` (Processamento)** — todas as mensagens/etapas, *pipelines*, colunas alteradas e **filtro por evento** com paginação sobre a base completa (valor *Original* × valor *Atual*).
    4. **Explorador Geral & Mapa Folium Paginado** — filtros combinados (incluindo eventos de `Log`) sobre os {n_rows:,} registros, sem omitir nenhum ponto.
    5. **Registros Descartados** — união de todos os `*_discarded.parquet` por fonte e o **motivo do descarte** registrado no `Log`.
    """)


@app.cell
def constants():
    # Sentinel used by every "(Todas)" dropdown filter
    ALL_OPTION = "(Todas)"

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
        "248": "248 - TV Digital",
        "728": "728 - Serviço Auxiliar de Radiodifusão",
        "-1": "-1 - Sem Código (Aero/ICAO)",
    }

    # Upper edges (MHz) of each band; pl.cut produces len(breaks) + 1 labelled bins
    SPECTRUM_BAND_BREAKS = [0.3, 3, 30, 300, 1000, 3000, 6000, 30000]
    SPECTRUM_BAND_LABELS = [
        "1. LF (<300 kHz)",
        "2. MF/OM (0.3–3 MHz)",
        "3. HF (3–30 MHz)",
        "4. VHF (30–300 MHz)",
        "5. UHF Baixo (300–1000 MHz)",
        "6. UHF/L/S (1–3 GHz)",
        "7. Banda C/5G (3–6 GHz)",
        "8. SHF Micro-ondas (6–30 GHz)",
        "9. EHF mmWave (>30 GHz)",
    ]

    # Columns stored as Categorical strings that are really numeric -> name of derived Float64 column
    NUMERIC_COLS = {
        "Frequência": "Freq_MHz",
        "Latitude": "Lat",
        "Longitude": "Lon",
        "Largura_Emissão(kHz)": "Largura_kHz",
        "Potência_Transmissor(W)": "Potencia_W",
        "Altura_Antena(m)": "Altura_m",
        "Cota_Base_Torre(m)": "Cota_m",
        "Ganho_Antena(dBd)": "Ganho_dBd",
        "Ângulo_Elevação_Antena": "Elevacao_graus",
        "Azimute_Antena": "Azimute_graus",
        "Atenuação_Linha(db/100m)": "Atenuacao_dB_100m",
        "Perdas_Acessórias_Linha(db)": "Perdas_dB",
        "Comprimento_Linha(m)": "Comprimento_m",
    }

    # Generic categorical palette (Tableau 20) for map colouring
    PALETTE = [
        "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1",
        "#ff9da7", "#9c755f", "#bab0ac", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]


    def band_expr(freq_col: str = "Freq_MHz") -> pl.Expr:
        """Spectrum band label (String) for a numeric MHz column."""
        return (
            pl.col(freq_col)
            .bin_intervals(SPECTRUM_BAND_BREAKS, labels=SPECTRUM_BAND_LABELS, right_closed=True)
            .cast(pl.String)
        )


    return ALL_OPTION, NUMERIC_COLS, PALETTE, SERVICO_NOMES, band_expr


@app.cell
def log_model():
    # Category -> colour. Order = priority used to pick a template's primary category.
    LOG_CATEGORY_META = {
        "Descarte": "#b279a2",
        "Correção de Coordenadas": "#f58518",
        "Validação IBGE": "#e45756",
        "Território": "#72b7b2",
        "Classificação RADCOM": "#54a24b",
        "Agrupamento": "#9d755d",
        "Enriquecimento SMP": "#4c78a8",
        "Outro": "#7f7f7f",
        "Sem alteração": "#bab0ac",
    }


    def classify_log_step(msg: str) -> tuple[str, str]:
        """Return (Tag, Categoria) for a raw log message (JSON key "Log")."""
        low = (msg or "").lower()
        if "valor não numérico descartado" in low:
            return ("COORD_NAO_NUMERICA", "Correção de Coordenadas")
        if "descartad" in low or "não considerado" in low:
            if low.startswith("srd:"):
                return ("DESCARTE_SRD_STATUS", "Descarte")
            if "canalização uplink" in low:
                return ("DESCARTE_SMP_CANAL_UPLINK", "Descarte")
            if "canalização inválida" in low:
                return ("DESCARTE_SMP_CANAL_INVALIDA", "Descarte")
            return ("DESCARTE_OUTRO", "Descarte")
        if "colunas criadas à partir da canalização" in low:
            return ("SMP_CANALIZACAO", "Enriquecimento SMP")
        if "registro adicional com frequência de recepção" in low:
            return ("SMP_RX_ESPELHADO", "Enriquecimento SMP")
        if "do município inserida" in low:
            return ("SMP_COORD_MUNICIPIO", "Enriquecimento SMP")
        if "agrupamento" in low or "agrupados" in low:
            return ("AGRUPAMENTO", "Agrupamento")
        if low.startswith("radcom:"):
            return ("RADCOM_FASE_SITUACAO", "Classificação RADCOM")
        if "município distinto" in low or low.startswith("coordenadas divergentes"):
            return ("IBGE_COORD_DIVERGENTE", "Validação IBGE")
        if "coordenadas inválidas" in low:
            return ("COORD_INVALIDA_SUBST_IBGE", "Validação IBGE")
        if "coordenadas nulas" in low or "valor nulo de coordenadas" in low:
            return ("COORD_NULA_PREENCHIDA_IBGE", "Validação IBGE")
        if "dado derivado das coordenadas" in low or "retirado do polígono" in low:
            return ("IBGE_POLIGONO_RECUPERADO", "Validação IBGE")
        if "nomes de municípios" in low:
            return ("IBGE_NOME_NORMALIZADO", "Validação IBGE")
        if "código do município" in low:
            return ("MUN_CODIGO_INCONSISTENTE", "Validação IBGE")
        if "fora do território" in low:
            return ("FORA_TERRITORIO_BR", "Território")
        if "trocadas" in low:
            return ("COORD_LAT_LON_TROCADAS", "Correção de Coordenadas")
        if "invertido" in low:
            return ("COORD_SINAL_INVERTIDO", "Correção de Coordenadas")
        if "por 10" in low:
            return ("COORD_ESCALA_10", "Correção de Coordenadas")
        if low.startswith("coordenadas corrigidas"):
            return ("COORD_CORRIGIDA_OUTRA", "Correção de Coordenadas")
        return ("OUTRO", "Outro")


    def primary_category(categories: list[str]) -> str:
        """Highest-priority category (per LOG_CATEGORY_META order) among a template's steps."""
        for cat in LOG_CATEGORY_META:
            if cat in categories:
                return cat
        return "Sem alteração"


    def parse_log_templates(log_counts: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Unnest distinct JSON `Log` templates into (templates, events).

        `log_counts` has one row per distinct raw `Log` string and its row count (`Linhas`).
        Events are one row per (template, step, affected column).
        """
        tmpl_rows, event_rows = [], []
        for raw, n_rows in log_counts.select("Log", "Linhas").iter_rows():
            items = json.loads(raw) if raw else []
            tags, cats, parts = [], [], []
            for step, it in enumerate(items, 1):
                msg = " ".join(str(it.get("Log", "")).split())
                tag, cat = classify_log_step(msg)
                tags.append(tag)
                cats.append(cat)
                # A step may touch several columns at once: Coluna/Original are then parallel lists
                cols, vals = it.get("Coluna"), it.get("Original")
                if not isinstance(cols, list):
                    cols = [cols]
                if not isinstance(vals, list):
                    vals = [vals] * len(cols)
                detail = ", ".join(f"{c}: '{v}'" for c, v in zip(cols, vals) if c)
                parts.append(f"{step}. {tag}" + (f" [{detail}]" if detail else ""))
                for col, val in zip(cols, vals):
                    event_rows.append(
                        {
                            "Log": raw,
                            "Passo": step,
                            "Tag": tag,
                            "Categoria": cat,
                            "Processamento": msg,
                            # Columns are logged as "#Latitude"; normalise to the dataset column name
                            "Coluna": "" if col is None else str(col).lstrip("#"),
                            "Coluna_Log": "" if col is None else str(col),
                            "Original": "" if val is None else str(val),
                            "Linhas": int(n_rows),
                        }
                    )
            uniq_cats = list(dict.fromkeys(cats))
            tmpl_rows.append(
                {
                    "Log": raw,
                    "Qtd_Passos": len(items),
                    "Pipeline": " → ".join(dict.fromkeys(tags)) or "SEM_ALTERACAO",
                    "Categoria_Principal": primary_category(uniq_cats),
                    "Categorias": " | ".join(uniq_cats) or "Sem alteração",
                    "Resumo_Log": " | ".join(parts) or "Sem alterações registradas",
                    "Linhas": int(n_rows),
                }
            )
        event_schema = {
            "Log": pl.String, "Passo": pl.Int64, "Tag": pl.String, "Categoria": pl.String,
            "Processamento": pl.String, "Coluna": pl.String, "Coluna_Log": pl.String, "Original": pl.String, "Linhas": pl.Int64,
        }
        templates = pl.DataFrame(tmpl_rows) if tmpl_rows else pl.DataFrame(
            schema={"Log": pl.String, "Qtd_Passos": pl.Int64, "Pipeline": pl.String,
                    "Categoria_Principal": pl.String, "Categorias": pl.String,
                    "Resumo_Log": pl.String, "Linhas": pl.Int64}
        )
        return templates, pl.DataFrame(event_rows, schema=event_schema)


    return LOG_CATEGORY_META, parse_log_templates


@app.cell
def data(NUMERIC_COLS, parse_log_templates):
    def _resolve_arq_dir() -> Path:
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
            if (candidate / "saida" / "estacoes_full.parquet").exists():
                return candidate
        return fallback


    def num_expr(col: str) -> pl.Expr:
        """Parse a Categorical/String column to Float64 ('-1' sentinel and garbage -> null, ',' -> '.')."""
        s = pl.col(col).cast(pl.String).str.strip_chars()
        return (
            pl.when(s == "-1")
            .then(None)
            .otherwise(s.str.replace(",", ".", literal=True).cast(pl.Float64, strict=False))
        )


    DATA_DIR = _resolve_arq_dir() / "saida"
    full_path = DATA_DIR / "estacoes_full.parquet"
    full_size_mb = full_path.stat().st_size / (1024**2)
    full_mtime_str = datetime.fromtimestamp(full_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")

    # Lazy scan: nothing is materialised here; every section pushes its own projection/filters down
    full_raw_lf = pl.scan_parquet(full_path, row_index_name="Linha")
    FULL_COLUMNS = [c for c in full_raw_lf.collect_schema().names() if c != "Linha"]

    # Typed view: numeric columns parsed, validity date parsed, Log kept Categorical for cheap filtering
    full_lf = full_raw_lf.with_columns(
        *[num_expr(src).alias(dst) for src, dst in NUMERIC_COLS.items() if src in FULL_COLUMNS],
        pl.col("Validade_RF").cast(pl.String).str.to_date("%Y-%m-%d", strict=False).alias("Validade_Data"),
    )

    # Only ~3k distinct Log templates exist for millions of rows: parse them once
    log_counts = (
        full_raw_lf.group_by("Log")
        .agg(pl.len().cast(pl.Int64).alias("Linhas"))
        .with_columns(pl.col("Log").cast(pl.String))
        .collect()
    )
    n_rows = int(log_counts["Linhas"].sum())
    log_templates, log_events = parse_log_templates(log_counts)
    return (
        DATA_DIR,
        FULL_COLUMNS,
        full_lf,
        full_mtime_str,
        full_raw_lf,
        full_size_mb,
        log_counts,
        log_events,
        log_templates,
        n_rows,
        num_expr,
    )


@app.cell
def helpers_ui(PALETTE):
    def kpi_row(items: list[tuple[str, str, str]]):
        """Row of equally sized `mo.stat` cards from (label, value, caption) tuples."""
        return mo.hstack(
            [mo.stat(label=label, value=value, caption=caption) for label, value, caption in items],
            widths="equal",
        )


    def folium_iframe(fmap: folium.Map, height: str):
        """Embed a Folium map as a self-contained marimo iframe."""
        return mo.iframe(fmap.get_root().render(), width="100%", height=height)


    def color_mapping(values: list[str], fixed: dict[str, str] | None = None) -> dict[str, str]:
        """Stable value -> colour map: `fixed` colours first, then the generic palette in order."""
        fixed = fixed or {}
        out, i = {}, 0
        for v in values:
            if v in fixed:
                out[v] = fixed[v]
            else:
                out[v] = PALETTE[i % len(PALETTE)]
                i += 1
        return out


    def build_points_map(
        points: pl.DataFrame,
        color_col: str,
        colors: dict[str, str],
        tooltip_cols: list[str],
        *,
        cluster: bool = False,
    ) -> folium.Map:
        """Folium map with one toggleable layer per `color_col` value; `points` must have float `Lat`/`Lon`."""
        pts = points.filter(
            pl.col("Lat").is_not_null() & pl.col("Lon").is_not_null()
            & pl.col("Lat").is_between(-90, 90) & pl.col("Lon").is_between(-180, 180)
        )
        center = [pts["Lat"].mean(), pts["Lon"].mean()] if pts.height else [-14.2, -51.9]
        fmap = folium.Map(location=center, zoom_start=4 if pts.height > 50 else 7, tiles="OpenStreetMap",
                          control_scale=True, prefer_canvas=True)
        counts = pts.group_by(color_col).len().sort("len", descending=True)
        for value, n in counts.iter_rows():
            layer = folium.FeatureGroup(name=f"{html.escape(str(value))} ({n:,})", show=True).add_to(fmap)
            target = plugins.MarkerCluster(disableClusteringAtZoom=13).add_to(layer) if cluster else layer
            color = colors.get(value, "#333333")
            sub = pts.filter(pl.col(color_col) == value).select("Lat", "Lon", *tooltip_cols)
            for row in sub.iter_rows(named=True):
                tip = "".join(
                    f"<tr><td style='padding-right:6px'><b>{html.escape(c)}</b></td><td>{html.escape(str(row[c])[:300])}</td></tr>"
                    for c in tooltip_cols
                )
                folium.CircleMarker(
                    location=[row["Lat"], row["Lon"]], radius=4, color=color, weight=1,
                    fill=True, fill_color=color, fill_opacity=0.75,
                    tooltip=folium.Tooltip(f"<table style='font-size:11px;max-width:480px'>{tip}</table>", sticky=True),
                ).add_to(target)
        if pts.height:
            fmap.fit_bounds([[pts["Lat"].min(), pts["Lon"].min()], [pts["Lat"].max(), pts["Lon"].max()]])
        folium.LayerControl(collapsed=False).add_to(fmap)
        return fmap


    return build_points_map, color_mapping, folium_iframe, kpi_row


@app.cell
def overview_calc(FULL_COLUMNS, NUMERIC_COLS, full_lf, full_raw_lf, n_rows):
    _stat_exprs = []
    for _c in FULL_COLUMNS:
        _s = pl.col(_c).cast(pl.String)
        _stat_exprs += [
            pl.col(_c).n_unique().alias(f"{_c}|nu"),
            (_s == "-1").sum().alias(f"{_c}|sent"),
            pl.col(_c).null_count().alias(f"{_c}|null"),
        ]
    _stat_exprs += [
        (pl.col("Log").cast(pl.String) != "[]").sum().alias("rows_with_log"),
        pl.col("Fistel").n_unique().alias("n_fistel"),
    ]
    _top_lfs = [
        full_raw_lf.group_by(_c).agg(pl.len().alias("n")).sort("n", descending=True).head(1)
        .select(pl.col(_c).cast(pl.String).alias("v"), "n")
        for _c in FULL_COLUMNS
    ]
    _num_cols = [dst for src, dst in NUMERIC_COLS.items() if src in FULL_COLUMNS]
    _num_exprs = []
    for _c in _num_cols:
        _num_exprs += [
            pl.col(_c).count().alias(f"{_c}|n"),
            pl.col(_c).min().alias(f"{_c}|min"),
            pl.col(_c).quantile(0.01).alias(f"{_c}|p01"),
            pl.col(_c).median().alias(f"{_c}|p50"),
            pl.col(_c).mean().alias(f"{_c}|mean"),
            pl.col(_c).quantile(0.99).alias(f"{_c}|p99"),
            pl.col(_c).max().alias(f"{_c}|max"),
        ]
    _stats_df, _num_df, *_tops = pl.collect_all(
        [full_raw_lf.select(_stat_exprs), full_lf.select(_num_exprs), *_top_lfs]
    )
    overview_stats = _stats_df.row(0, named=True)
    _num = _num_df.row(0, named=True)

    schema_profile_df = pl.DataFrame(
        [
            {
                "Coluna": _c,
                "Valores_Unicos": overview_stats[f"{_c}|nu"],
                "Sentinela_-1": overview_stats[f"{_c}|sent"],
                "Nulos": overview_stats[f"{_c}|null"],
                "Preenchido_%": round(
                    (1.0 - (overview_stats[f"{_c}|sent"] + overview_stats[f"{_c}|null"]) / max(n_rows, 1)) * 100.0, 2
                ),
                "Valor_Mais_Comum": (_t["v"][0] or "")[:60] if _t.height else "",
                "Freq_Valor_Mais_Comum_%": round(_t["n"][0] / max(n_rows, 1) * 100.0, 2) if _t.height else 0.0,
            }
            for _c, _t in zip(FULL_COLUMNS, _tops)
        ]
    )

    _src_of = {dst: src for src, dst in NUMERIC_COLS.items()}
    numeric_profile_df = pl.DataFrame(
        [
            {
                "Coluna_Derivada": _c,
                "Coluna_Original": _src_of[_c],
                "Valores_Numericos": _num[f"{_c}|n"],
                "Numerico_%": round(_num[f"{_c}|n"] / max(n_rows, 1) * 100.0, 2),
                "Nao_Numericos_(exc. -1)": n_rows - _num[f"{_c}|n"] - overview_stats[f"{_src_of[_c]}|sent"],
                **{k: _num[f"{_c}|{k}"] for k in ("min", "p01", "p50", "mean", "p99", "max")},
            }
            for _c in _num_cols
        ]
    )
    return numeric_profile_df, overview_stats, schema_profile_df


@app.cell(hide_code=True)
def overview_view(
    FULL_COLUMNS,
    full_size_mb,
    kpi_row,
    log_events,
    log_templates,
    n_rows,
    numeric_profile_df,
    overview_stats,
    schema_profile_df,
):
    _rows_log = overview_stats["rows_with_log"]
    _low_fill = schema_profile_df.filter(pl.col("Preenchido_%") < 50.0).height
    _dirty_num = numeric_profile_df.filter(pl.col("Nao_Numericos_(exc. -1)") > 0)

    _miss_chart = (
        alt.Chart(schema_profile_df)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("Preenchido_%:Q", title="% Preenchido (não '-1' / não nulo)", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("Coluna:N", sort="-x", title=None),
            color=alt.condition(alt.datum["Preenchido_%"] < 50, alt.value("#e45756"), alt.value("#4c78a8")),
            tooltip=[
                "Coluna",
                "Preenchido_%",
                alt.Tooltip("Sentinela_-1:Q", format=","),
                alt.Tooltip("Valores_Unicos:Q", format=","),
                "Valor_Mais_Comum",
                "Freq_Valor_Mais_Comum_%",
            ],
        )
        .properties(height=520, title=f"Completude por Coluna ({n_rows:,} registros)")
    )

    mo.vstack(
        [
            mo.md(
                f"""
    ## 1. Visão Geral & Completude do Esquema
    > A base completa tem **{n_rows:,} registros** em **{len(FULL_COLUMNS)} colunas**, cobrindo **{overview_stats['Estação|nu']:,} estações**, **{overview_stats['Entidade|nu']:,} entidades** e **{overview_stats['n_fistel']:,} Fistéis**.
    > **{_rows_log:,} registros ({_rows_log / n_rows * 100:.2f}%)** passaram por pelo menos uma etapa registrada no `Log`. **{_low_fill} de {len(FULL_COLUMNS)} colunas** têm preenchimento < 50% (atributos de engenharia exclusivos de radiodifusão).
    > Todas as colunas são `Categorical` apenas por armazenamento: abaixo, as colunas numéricas foram convertidas para `Float64` (sentinela `"-1"` → nulo, vírgula decimal → ponto) — **{_dirty_num.height}** delas ainda contêm valores não numéricos.
    """
            ),
            kpi_row(
                [
                    ("Registros (emissores)", f"{n_rows:,}", f"{full_size_mb:.1f} MB Parquet"),
                    ("Estações Distintas", f"{overview_stats['Estação|nu']:,}", f"{n_rows / max(overview_stats['Estação|nu'], 1):.1f} registros / estação"),
                    ("Templates de Log", f"{len(log_templates):,}", f"{log_events['Processamento'].n_unique()} mensagens distintas"),
                    ("Registros com Log", f"{_rows_log:,}", f"{_rows_log / n_rows * 100:.2f}% da base"),
                ]
            ),
            mo.ui.tabs(
                {
                    "📊 Completude": _miss_chart,
                    "📋 Perfil das Colunas": mo.ui.table(schema_profile_df, pagination=True, page_size=15),
                    "🔢 Perfil Numérico (colunas convertidas)": mo.ui.table(numeric_profile_df, pagination=False),
                }
            ),
        ]
    )


@app.cell
def dist_calc(SERVICO_NOMES, band_expr, full_lf, full_raw_lf):
    _band = (
        full_lf.select(
            band_expr().alias("Faixa")
        )
        .group_by("Faixa").agg(pl.len().alias("Registros")).sort("Faixa")
    )
    _serv = (
        full_lf.group_by(pl.col("Serviço").cast(pl.String))
        .agg(pl.len().alias("Registros"), pl.col("Freq_MHz").median().alias("Freq_Mediana_MHz"),
             pl.col("Estação").n_unique().alias("Estacoes"))
        .sort("Registros", descending=True)
    )
    _fonte = (
        full_raw_lf.group_by(pl.col("Fonte").cast(pl.String))
        .agg(pl.len().alias("Registros"), pl.col("Estação").n_unique().alias("Estacoes"),
             (pl.col("Log").cast(pl.String) != "[]").mean().mul(100).round(2).alias("%_Com_Log"))
        .sort("Registros", descending=True)
    )
    _uf = full_raw_lf.group_by(pl.col("UF").cast(pl.String)).agg(pl.len().alias("Registros")).sort("Registros", descending=True)
    _ent = (
        full_raw_lf.group_by(pl.col("Entidade").cast(pl.String))
        .agg(pl.len().alias("Registros"), pl.col("Estação").n_unique().alias("Estacoes"),
             pl.col("UF").n_unique().alias("UFs"),
             pl.col("Serviço").cast(pl.String).mode().first().alias("Servico_Principal"),
             pl.col("Fonte").cast(pl.String).mode().first().alias("Fonte_Principal"))
        .sort("Registros", descending=True).head(25)
    )
    _valid = (
        full_lf.select(pl.col("Validade_Data").dt.year().alias("Ano"))
        .group_by("Ano").agg(pl.len().alias("Registros")).sort("Ano")
    )
    _emis = (
        full_raw_lf.group_by(pl.col("Classe_Emissão").cast(pl.String))
        .agg(pl.len().alias("Registros")).sort("Registros", descending=True).head(20)
    )
    _status = full_raw_lf.group_by(pl.col("Status").cast(pl.String)).agg(pl.len().alias("Registros")).sort("Registros", descending=True)
    _fonte_serv = (
        full_raw_lf.group_by(pl.col("Fonte").cast(pl.String), pl.col("Serviço").cast(pl.String))
        .agg(pl.len().alias("Registros"))
    )
    _freq_hist = (
        full_lf.filter(pl.col("Freq_MHz") > 0)
        .select(pl.col("Freq_MHz").log10().floor().add(pl.col("Freq_MHz").log10().mod(1).mul(10).floor().truediv(10)).alias("log10_bin"))
        .group_by("log10_bin").agg(pl.len().alias("Registros")).sort("log10_bin")
    )

    (
        band_df, serv_df, fonte_df, uf_df, ent_df, validade_df, emissao_df, status_df, fonte_serv_df, freq_hist_df
    ) = pl.collect_all([_band, _serv, _fonte, _uf, _ent, _valid, _emis, _status, _fonte_serv, _freq_hist])
    serv_df = serv_df.with_columns(
        pl.col("Serviço").replace_strict(SERVICO_NOMES, default=pl.lit("Serviço ") + pl.col("Serviço")).alias("Descricao")
    )
    return (
        band_df,
        emissao_df,
        ent_df,
        fonte_df,
        fonte_serv_df,
        freq_hist_df,
        serv_df,
        status_df,
        uf_df,
        validade_df,
    )


@app.cell(hide_code=True)
def dist_view(
    SERVICO_NOMES,
    band_df,
    emissao_df,
    ent_df,
    fonte_df,
    fonte_serv_df,
    freq_hist_df,
    n_rows,
    serv_df,
    status_df,
    uf_df,
    validade_df,
):
    _total = max(n_rows, 1)
    _top = serv_df.head(2).to_dicts()
    _sub3 = band_df.filter(pl.col("Faixa") <= "6.").get_column("Registros").sum() / _total * 100
    _shf = band_df.filter(pl.col("Faixa") >= "8.").get_column("Registros").sum() / _total * 100
    _today_year = datetime.now().year
    _expired = validade_df.filter(pl.col("Ano") < _today_year)["Registros"].sum()

    _bar = lambda df, x, y, color, title, sort="-x", h=300: (
        alt.Chart(df).mark_bar(color=color, cornerRadiusEnd=3)
        .encode(x=alt.X(f"{x}:Q", title=x), y=alt.Y(f"{y}:N", sort=sort, title=None),
                tooltip=[c for c in df.columns])
        .properties(height=h, title=title)
    )

    _freq_chart = (
        alt.Chart(freq_hist_df.with_columns((10 ** pl.col("log10_bin")).alias("MHz")))
        .mark_bar(color="#2b8cbe")
        .encode(
            x=alt.X("MHz:Q", scale=alt.Scale(type="log"), title="Frequência (MHz, escala log; bins de 0.1 década)"),
            y=alt.Y("Registros:Q", scale=alt.Scale(type="symlog")),
            tooltip=[alt.Tooltip("MHz:Q", format=",.3f"), alt.Tooltip("Registros:Q", format=",")],
        )
        .properties(height=260, width=700, title="Histograma de Frequências (todas as linhas)")
    )

    _heat = (
        alt.Chart(fonte_serv_df.with_columns(pl.col("Serviço").replace_strict(SERVICO_NOMES, default=pl.col("Serviço")).alias("Servico")))
        .mark_rect()
        .encode(
            x=alt.X("Servico:N", title="Serviço"),
            y=alt.Y("Fonte:N", title="Fonte"),
            color=alt.Color("Registros:Q", scale=alt.Scale(type="log", scheme="blues")),
            tooltip=["Fonte", "Servico", alt.Tooltip("Registros:Q", format=",")],
        )
        .properties(height=420, title="Matriz Fonte × Serviço (escala log)")
    )

    _valid_chart = (
        alt.Chart(validade_df.drop_nulls("Ano"))
        .mark_bar()
        .encode(
            x=alt.X("Ano:O", title="Ano de Validade_RF"),
            y=alt.Y("Registros:Q"),
            color=alt.condition(alt.datum.Ano < _today_year, alt.value("#e45756"), alt.value("#4c78a8")),
            tooltip=["Ano", alt.Tooltip("Registros:Q", format=",")],
        )
        .properties(height=250, title="Validade das Licenças (vermelho = ano já vencido)")
    )

    mo.vstack(
        [
            mo.md(
                f"""
    ## 2. Espectro, Serviços, Fontes & Operadoras
    - **Concentração por Serviço:** `{_top[0]['Descricao']}` (**{_top[0]['Registros'] / _total * 100:.1f}%**) e `{_top[1]['Descricao']}` (**{_top[1]['Registros'] / _total * 100:.1f}%**) somam **{(_top[0]['Registros'] + _top[1]['Registros']) / _total * 100:.1f}%** das linhas.
    - **Perfil Espectral:** **{_sub3:.1f}%** dos registros estão abaixo de 6 GHz e **{_shf:.1f}%** acima de 6 GHz (enlaces SHF/EHF).
    - **Validade:** **{_expired:,}** registros têm `Validade_RF` em ano anterior a {_today_year}; **{validade_df.filter(pl.col('Ano').is_null())['Registros'].sum():,}** não têm data válida.
    """
            ),
            mo.ui.tabs(
                {
                    "📡 Espectro & Serviços": mo.vstack(
                        [
                            mo.hstack(
                                [
                                    _bar(band_df, "Registros", "Faixa", "#2b8cbe", "Registros por Faixa", sort=None),
                                    _bar(serv_df.head(15), "Registros", "Descricao", "#54a24b", "Top 15 Serviços"),
                                ],
                                justify="start",
                            ),
                            _freq_chart,
                        ]
                    ),
                    "🗂️ Fontes & UFs": mo.vstack(
                        [
                            mo.hstack(
                                [
                                    _bar(fonte_df, "Registros", "Fonte", "#f58518", "Registros por Fonte", h=420),
                                    _bar(uf_df, "Registros", "UF", "#4c78a8", "Registros por UF", h=420),
                                ],
                                justify="start",
                            ),
                            mo.ui.table(fonte_df, pagination=False),
                        ]
                    ),
                    "🔥 Fonte × Serviço": _heat,
                    "🏢 Top 25 Entidades": mo.ui.table(ent_df, pagination=False),
                    "📅 Validade, Status & Emissão": mo.vstack(
                        [
                            _valid_chart,
                            mo.hstack(
                                [
                                    _bar(status_df, "Registros", "Status", "#b279a2", "Status", h=380),
                                    _bar(emissao_df, "Registros", "Classe_Emissão", "#9d755d", "Top 20 Classes de Emissão", h=380),
                                ],
                                justify="start",
                            ),
                        ]
                    ),
                }
            ),
        ]
    )


@app.cell
def helpers_pl():
    def haversine_expr(lat1: pl.Expr, lon1: pl.Expr, lat2: pl.Expr, lon2: pl.Expr) -> pl.Expr:
        """Great-circle distance in km between two coordinate expressions (nulls propagate)."""
        rad = 3.141592653589793 / 180.0
        dlat = (lat2 - lat1) * rad
        dlon = (lon2 - lon1) * rad
        a = (dlat / 2).sin() ** 2 + (lat1 * rad).cos() * (lat2 * rad).cos() * (dlon / 2).sin() ** 2
        return 2.0 * 6371.0 * a.clip(0.0, 1.0).sqrt().arcsin()


    def templates_expr(templates) -> pl.Expr:
        """Row filter keeping rows whose raw `Log` JSON is one of `templates`."""
        return pl.col("Log").cast(pl.String).is_in(list(templates))


    def search_expr(query: str, uniques: dict[str, pl.Series]) -> pl.Expr | None:
        """Case-insensitive search over every column: each whitespace-separated term must match some column.

        Matching is done on each column's distinct values (`uniques`), then mapped back with `is_in`,
        so the full table is never cast/scanned per term.
        """
        terms = query.lower().split()
        if not terms:
            return None
        out = pl.lit(True)
        for term in terms:
            term_expr = pl.lit(False)
            for col, values in uniques.items():
                hits = values.filter(values.str.to_lowercase().str.contains(term, literal=True))
                if hits.len():
                    term_expr = term_expr | pl.col(col).cast(pl.String).is_in(hits.implode())
            out = out & term_expr
        return out


    def current_value_expr(columns: list[str]) -> pl.Expr:
        """Current (post-processing) value of the column named in `Coluna`, as String."""
        expr = pl.lit(None, dtype=pl.String)
        for c in columns:
            expr = pl.when(pl.col("Coluna") == c).then(pl.col(c).cast(pl.String)).otherwise(expr)
        return expr


    return current_value_expr, haversine_expr, search_expr, templates_expr


@app.cell
def log_calc(
    full_lf,
    full_raw_lf,
    haversine_expr,
    log_events,
    log_templates,
    n_rows,
    templates_expr,
):
    # One small scan gives every Fonte × template combination; all row counts below are template joins
    fonte_log_counts = (
        full_raw_lf.group_by(pl.col("Fonte").cast(pl.String), pl.col("Log").cast(pl.String))
        .agg(pl.len().cast(pl.Int64).alias("Linhas"))
        .collect()
    )

    _ev_msg = log_events.unique(["Log", "Processamento"])
    msg_summary_df = (
        _ev_msg.group_by("Categoria", "Tag", "Processamento")
        .agg(pl.col("Log").n_unique().alias("Templates"), pl.col("Linhas").sum())
        .join(
            log_events.group_by("Processamento").agg(
                pl.col("Coluna").filter(pl.col("Coluna") != "").unique().sort().str.join(", ").alias("Colunas_Afetadas"),
                pl.col("Original").filter(pl.col("Original") != "").n_unique().alias("Valores_Originais_Distintos"),
            ),
            on="Processamento",
        )
        .with_columns((pl.col("Linhas") / n_rows * 100).round(3).alias("%_Base"))
        .sort("Linhas", descending=True)
    )

    pipeline_df = (
        log_templates.group_by("Pipeline", "Categoria_Principal", "Qtd_Passos")
        .agg(pl.len().alias("Templates"), pl.col("Linhas").sum(), pl.col("Resumo_Log").first().alias("Exemplo"))
        .with_columns((pl.col("Linhas") / n_rows * 100).round(3).alias("%_Base"))
        .sort("Linhas", descending=True)
    )

    tag_fonte_df = (
        fonte_log_counts.join(_ev_msg.unique(["Log", "Tag"]).select("Log", "Tag", "Categoria"), on="Log")
        .group_by("Fonte", "Tag", "Categoria")
        .agg(pl.col("Linhas").sum())
    )

    coluna_df = (
        log_events.filter(pl.col("Coluna") != "")
        .unique(["Log", "Coluna", "Tag"])
        .group_by("Coluna", "Tag")
        .agg(pl.col("Linhas").sum(), pl.col("Log").n_unique().alias("Templates"))
        .sort("Linhas", descending=True)
    )

    # Coordinate shift: rawest logged Latitude/Longitude (first step) vs. the final value in the row
    _coord_orig = (
        log_events.filter(pl.col("Coluna").is_in(["Latitude", "Longitude"]))
        .sort("Passo")
        .group_by("Log", "Coluna")
        .agg(pl.col("Original").first(), pl.col("Tag").first())
        .with_columns(pl.col("Original").str.replace(",", ".", literal=True).cast(pl.Float64, strict=False))
        .pivot(on="Coluna", index="Log", values=["Original", "Tag"])
    )
    for _c in ("Original_Latitude", "Original_Longitude", "Tag_Latitude", "Tag_Longitude"):
        if _c not in _coord_orig.columns:
            _coord_orig = _coord_orig.with_columns(pl.lit(None, dtype=pl.Float64 if _c.startswith("Orig") else pl.String).alias(_c))
    if _coord_orig.height:
        coord_shift_df = (
            full_lf.filter(templates_expr(_coord_orig["Log"]))
            .select("Linha", "Fonte", "UF", "Município", "Entidade", "Estação", "Lat", "Lon",
                    pl.col("Log").cast(pl.String))
            .collect()
            .join(_coord_orig, on="Log")
            .with_columns(
                pl.coalesce("Tag_Latitude", "Tag_Longitude").alias("Tag"),
                # A step may touch a single axis: the other one keeps its current value
                pl.when(pl.col("Tag_Latitude").is_null()).then(pl.col("Lat")).otherwise(pl.col("Original_Latitude")).alias("Original_Latitude"),
                pl.when(pl.col("Tag_Longitude").is_null()).then(pl.col("Lon")).otherwise(pl.col("Original_Longitude")).alias("Original_Longitude"),
            )
            .with_columns(
                haversine_expr(pl.col("Original_Latitude"), pl.col("Original_Longitude"), pl.col("Lat"), pl.col("Lon"))
                .alias("Deslocamento_km"),
            )
            .drop("Tag_Latitude", "Tag_Longitude")
        )
    else:
        coord_shift_df = pl.DataFrame()
    return (
        coluna_df,
        coord_shift_df,
        fonte_log_counts,
        msg_summary_df,
        pipeline_df,
        tag_fonte_df,
    )


@app.cell(hide_code=True)
def log_overview_view(
    LOG_CATEGORY_META,
    coluna_df,
    coord_shift_df,
    kpi_row,
    log_templates,
    msg_summary_df,
    n_rows,
    pipeline_df,
    tag_fonte_df,
):
    _rows_log = int(log_templates.filter(pl.col("Log") != "[]")["Linhas"].sum())
    _cat_df = log_templates.group_by("Categoria_Principal").agg(pl.col("Linhas").sum()).sort("Linhas", descending=True)
    _cat_scale = alt.Scale(domain=list(LOG_CATEGORY_META), range=list(LOG_CATEGORY_META.values()))

    _msg_chart = (
        alt.Chart(msg_summary_df)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("Linhas:Q", scale=alt.Scale(type="symlog"), title="Registros afetados (symlog)"),
            y=alt.Y("Processamento:N", sort="-x", title=None, axis=alt.Axis(labelLimit=520)),
            color=alt.Color("Categoria:N", scale=_cat_scale),
            tooltip=["Categoria", "Tag", "Processamento", alt.Tooltip("Linhas:Q", format=","), "Templates",
                     "Colunas_Afetadas", "%_Base"],
        )
        .properties(height=max(220, 24 * msg_summary_df.height), title="Todas as Mensagens de Processamento do Log")
    )

    _heat = (
        alt.Chart(tag_fonte_df)
        .mark_rect()
        .encode(
            x=alt.X("Tag:N", title=None, axis=alt.Axis(labelAngle=-40)),
            y=alt.Y("Fonte:N", title=None),
            color=alt.Color("Linhas:Q", scale=alt.Scale(type="log", scheme="orangered")),
            tooltip=["Fonte", "Tag", "Categoria", alt.Tooltip("Linhas:Q", format=",")],
        )
        .properties(height=380, title="Eventos de Log por Fonte (registros, escala log)")
    )

    _steps = (
        alt.Chart(log_templates.group_by("Qtd_Passos").agg(pl.col("Linhas").sum()))
        .mark_bar(color="#4c78a8")
        .encode(x=alt.X("Qtd_Passos:O", title="Nº de passos no Log"), y=alt.Y("Linhas:Q", scale=alt.Scale(type="symlog")),
                tooltip=["Qtd_Passos", alt.Tooltip("Linhas:Q", format=",")])
        .properties(height=220, width=360, title="Profundidade do Log por registro")
    )

    if coord_shift_df.height:
        _shift = coord_shift_df.with_columns(
            pl.col("Deslocamento_km").clip(0.001, None).log10().round(1).alias("log10_km")
        )
        _shift_summary = (
            coord_shift_df.group_by("Tag")
            .agg(pl.len().alias("Registros"), pl.col("Deslocamento_km").median().alias("Mediana_km"),
                 pl.col("Deslocamento_km").quantile(0.9).alias("P90_km"), pl.col("Deslocamento_km").max().alias("Max_km"))
            .sort("Registros", descending=True)
        )
        _shift_chart = (
            alt.Chart(_shift.group_by("Tag", "log10_km").len())
            .mark_bar()
            .encode(
                x=alt.X("log10_km:Q", title="log10(deslocamento km)  [−1 = 100 m, 0 = 1 km, 2 = 100 km]"),
                y=alt.Y("len:Q", title="Registros", stack=True),
                color=alt.Color("Tag:N"),
                tooltip=["Tag", "log10_km", alt.Tooltip("len:Q", format=",")],
            )
            .properties(height=260, width=700, title="Deslocamento entre a Coordenada Original (Log) e a Final")
        )
        _shift_tab = mo.vstack(
            [
                mo.md(f"**{coord_shift_df.height:,}** registros tiveram `Latitude`/`Longitude` alteradas; distância entre o valor original registrado no `Log` e o valor final:"),
                _shift_chart,
                mo.ui.table(_shift_summary, pagination=False),
                mo.ui.table(coord_shift_df.sort("Deslocamento_km", descending=True).drop("Log"), pagination=True, page_size=10),
            ]
        )
    else:
        _shift_tab = mo.md("Nenhuma alteração de coordenadas registrada.")

    mo.vstack(
        [
            mo.md(
                f"""
    ## 3. Explorador do `Log` — Todo o Processamento Registrado
    A coluna `Log` contém uma lista JSON de passos `{{"Log": mensagem, "Coluna": coluna(s), "Original": valor(es) antes do passo}}`.
    Há **{len(log_templates):,} templates distintos** ( **{msg_summary_df.height} mensagens** classificadas em **{len(msg_summary_df['Tag'].unique())} tags** ) cobrindo **{_rows_log:,} registros ({_rows_log / n_rows * 100:.2f}%)**.
    """
            ),
            kpi_row(
                [(row["Categoria_Principal"], f"{row['Linhas']:,}", f"{row['Linhas'] / n_rows * 100:.2f}% dos registros")
                 for row in _cat_df.head(5).iter_rows(named=True)]
            ),
            mo.ui.tabs(
                {
                    "📊 Mensagens": mo.vstack([_msg_chart, mo.ui.table(msg_summary_df, pagination=False)]),
                    "🔁 Pipelines": mo.vstack([_steps, mo.ui.table(pipeline_df, pagination=True, page_size=15)]),
                    "🗂️ Fonte × Tag": _heat,
                    "🧬 Colunas Alteradas": mo.ui.table(coluna_df, pagination=True, page_size=15),
                    "📍 Deslocamento de Coordenadas": _shift_tab,
                }
            ),
        ]
    )


@app.cell(hide_code=True)
def log_controls(ALL_OPTION, fonte_df, log_events, msg_summary_df):
    log_msg_select = mo.ui.multiselect(
        options={
            f"[{r['Categoria']}] {r['Processamento'][:110]} — {r['Linhas']:,}": r["Processamento"]
            for r in msg_summary_df.iter_rows(named=True)
        },
        value=[],
        label="Mensagens do Log (vazio = qualquer):",
        full_width=True,
    )
    log_match_mode = mo.ui.radio(
        options={"Qualquer selecionada (OU)": "any", "Todas selecionadas (E)": "all"},
        value="Qualquer selecionada (OU)",
        label="Combinação:",
        inline=True,
    )
    log_coluna_select = mo.ui.dropdown(
        options=[ALL_OPTION] + sorted(c for c in log_events["Coluna"].unique().to_list() if c),
        value=ALL_OPTION,
        label="Coluna alterada:",
    )
    log_original_search = mo.ui.text(value="", placeholder="Ex: TV-CE-01, 0.0, -22.9...", label="Valor Original contém:")
    log_fonte_select = mo.ui.dropdown(
        options=[ALL_OPTION] + fonte_df["Fonte"].to_list(), value=ALL_OPTION, label="Fonte:"
    )
    log_only_changed = mo.ui.checkbox(value=True, label="Somente registros com Log não vazio")
    log_page_size = mo.ui.dropdown(options=["25", "50", "100", "250", "500"], value="50", label="Linhas por página:")
    return (
        log_coluna_select,
        log_fonte_select,
        log_match_mode,
        log_msg_select,
        log_only_changed,
        log_original_search,
        log_page_size,
    )


@app.cell
def log_match(
    ALL_OPTION,
    fonte_log_counts,
    full_lf,
    log_coluna_select,
    log_events,
    log_fonte_select,
    log_match_mode,
    log_msg_select,
    log_only_changed,
    log_original_search,
    log_templates,
    templates_expr,
):
    _ev = log_events
    _msgs = list(log_msg_select.value)
    if _msgs:
        _ev = _ev.filter(pl.col("Processamento").is_in(_msgs))
    if log_coluna_select.value != ALL_OPTION:
        _ev = _ev.filter(pl.col("Coluna") == log_coluna_select.value)
    if log_original_search.value.strip():
        _ev = _ev.filter(pl.col("Original").str.to_lowercase().str.contains(log_original_search.value.strip().lower(), literal=True))

    _has_event_filter = bool(_msgs) or log_coluna_select.value != ALL_OPTION or bool(log_original_search.value.strip())
    _tmpl = set(_ev["Log"]) if _has_event_filter else set(log_templates["Log"])
    if _msgs and log_match_mode.value == "all":
        _all = log_events.filter(pl.col("Processamento").is_in(_msgs)).group_by("Log").agg(pl.col("Processamento").n_unique().alias("k"))
        _tmpl &= set(_all.filter(pl.col("k") == len(_msgs))["Log"])
    if log_only_changed.value:
        _tmpl.discard("[]")
    log_sel_templates = sorted(_tmpl)
    log_sel_events = _ev.filter(pl.col("Log").is_in(log_sel_templates))

    _counts = fonte_log_counts.filter(pl.col("Log").is_in(log_sel_templates))
    _row_filter = templates_expr(log_sel_templates)
    if log_fonte_select.value != ALL_OPTION:
        _counts = _counts.filter(pl.col("Fonte") == log_fonte_select.value)
        _row_filter = _row_filter & (pl.col("Fonte") == log_fonte_select.value)
    log_sel_fonte_counts = _counts.group_by("Fonte").agg(pl.col("Linhas").sum()).sort("Linhas", descending=True)
    log_sel_n = int(_counts["Linhas"].sum())
    # Lazy: nothing is read until a page / aggregation is collected
    log_sel_lf = full_lf.filter(_row_filter)
    return (
        log_sel_events,
        log_sel_fonte_counts,
        log_sel_lf,
        log_sel_n,
        log_sel_templates,
    )


@app.cell(hide_code=True)
def log_page_ctrl(log_page_size, log_sel_n):
    log_n_pages = max(1, -(-log_sel_n // int(log_page_size.value)))
    log_page = mo.ui.number(start=1, stop=log_n_pages, step=1, value=1, label=f"Página (de {log_n_pages:,}):")
    return log_n_pages, log_page


@app.cell(hide_code=True)
def log_filter_view(
    FULL_COLUMNS,
    current_value_expr,
    log_coluna_select,
    log_events,
    log_fonte_select,
    log_match_mode,
    log_msg_select,
    log_n_pages,
    log_only_changed,
    log_original_search,
    log_page,
    log_page_size,
    log_sel_events,
    log_sel_fonte_counts,
    log_sel_lf,
    log_sel_n,
    log_sel_templates,
    log_templates,
    n_rows,
):
    _size = int(log_page_size.value)
    _page_no = min(max(int(log_page.value or 1), 1), log_n_pages)
    _offset = (_page_no - 1) * _size

    _page, _uf = pl.collect_all(
        [
            log_sel_lf.slice(_offset, _size).with_columns(pl.col("Log").cast(pl.String)),
            log_sel_lf.group_by(pl.col("UF").cast(pl.String)).agg(pl.len().alias("Linhas")).sort("Linhas", descending=True),
        ]
    )
    _page = _page.join(log_templates.select("Log", "Pipeline", "Resumo_Log"), on="Log", how="left")
    _page_view = _page.select(
        "Linha", "Fonte", "Entidade", "Estação", "Serviço", "Frequência", "Município", "UF",
        "Latitude", "Longitude", "Status", "Pipeline", "Resumo_Log",
    )

    # Before -> after for each logged step of the rows in this page
    _ev_page = (log_sel_events if log_sel_events.height else log_events).select(
        "Log", "Passo", "Tag", "Processamento", "Coluna", "Original"
    )
    _cols_in_full = [c for c in _ev_page["Coluna"].unique().to_list() if c in FULL_COLUMNS]
    _before_after = (
        _page.join(_ev_page, on="Log", how="inner")
        .with_columns(current_value_expr(_cols_in_full).alias("Atual"))
        .with_columns((pl.col("Original") != pl.col("Atual")).alias("Alterado"))
        .select("Linha", "Fonte", "Estação", "Passo", "Tag", "Processamento", "Coluna", "Original", "Atual", "Alterado")
        .sort("Linha", "Passo")
    )

    _tmpl_table = (
        log_templates.filter(pl.col("Log").is_in(log_sel_templates))
        .select("Linhas", "Qtd_Passos", "Pipeline", "Resumo_Log", "Log")
        .sort("Linhas", descending=True)
    )

    mo.vstack(
        [
            mo.md("### 🔎 Filtro por Evento do Log (aplicado à base completa, com paginação *lazy*)"),
            log_msg_select,
            mo.hstack([log_match_mode, log_coluna_select, log_original_search, log_fonte_select], widths=[2, 1, 1, 1]),
            mo.hstack([log_only_changed, log_page_size, log_page], justify="start"),
            mo.callout(
                mo.md(
                    f"**{log_sel_n:,}** registros ({log_sel_n / n_rows * 100:.3f}%) em **{len(log_sel_templates):,}** templates atendem ao filtro. "
                    f"Página **{_page_no:,}/{log_n_pages:,}** → linhas {min(_offset + 1, log_sel_n):,}–{_offset + _page.height:,} "
                    "(lidas sob demanda via `slice` no *LazyFrame*)."
                ),
                kind="info",
            ),
            mo.ui.tabs(
                {
                    "🧾 Registros (página)": mo.ui.table(_page_view, pagination=False),
                    "↔️ Antes → Depois (página)": mo.ui.table(_before_after, pagination=True, page_size=20),
                    "🗂️ Por Fonte / UF": mo.hstack(
                        [mo.ui.table(log_sel_fonte_counts, pagination=False), mo.ui.table(_uf, pagination=False)]
                    ),
                    "🧩 Templates Selecionados": mo.ui.table(_tmpl_table, pagination=True, page_size=15),
                }
            ),
        ]
    )


@app.cell
def exp_calc(FULL_COLUMNS, full_raw_lf, log_counts, log_events):
    # Distinct values per column (~1.7M strings total) back the any-column search and the Município dropdown
    _uniq = pl.collect_all(
        [full_raw_lf.select(pl.col(c).cast(pl.String).unique().drop_nulls()) for c in FULL_COLUMNS if c != "Log"]
    )
    column_uniques = {df.columns[0]: df.to_series() for df in _uniq}
    mun_uf_pairs = (
        full_raw_lf.select(pl.col("Município").cast(pl.String), pl.col("UF").cast(pl.String))
        .unique()
        .sort("UF", "Município")
        .collect()
    )
    # Tag -> templates (plus a pseudo-tag for rows without any Log step)
    tag_templates = {
        tag: set(logs)
        for tag, logs in log_events.group_by("Tag").agg(pl.col("Log").unique()).iter_rows()
    }
    tag_templates["SEM_ALTERACAO"] = {"[]"}
    tag_row_counts = (
        log_events.unique(["Log", "Tag"]).group_by("Tag").agg(pl.col("Linhas").sum())
        .vstack(pl.DataFrame({"Tag": ["SEM_ALTERACAO"], "Linhas": [int(log_counts.filter(pl.col("Log") == "[]")["Linhas"].sum())]}))
        .sort("Linhas", descending=True)
    )
    return column_uniques, mun_uf_pairs, tag_row_counts, tag_templates


@app.cell(hide_code=True)
def exp_controls(ALL_OPTION, fonte_df, serv_df, tag_row_counts, uf_df):
    exp_serv_select = mo.ui.multiselect(
        options={f"{r['Descricao']} [{r['Registros']:,}]": r["Serviço"] for r in serv_df.iter_rows(named=True)},
        value=[], label="Serviço (vazio = todos):",
    )
    exp_fonte_select = mo.ui.multiselect(
        options={f"{r['Fonte']} [{r['Registros']:,}]": r["Fonte"] for r in fonte_df.iter_rows(named=True)},
        value=[], label="Fonte (vazio = todas):",
    )
    exp_uf_select = mo.ui.dropdown(options=[ALL_OPTION] + sorted(uf_df["UF"].to_list()), value=ALL_OPTION, label="UF:")
    exp_freq_min = mo.ui.number(start=0, stop=100000, step=0.001, value=0, label="Freq. mín. (MHz):")
    exp_freq_max = mo.ui.number(start=0, stop=100000, step=0.001, value=100000, label="Freq. máx. (MHz):")
    exp_log_tags = mo.ui.multiselect(
        options={f"{r['Tag']} [{r['Linhas']:,}]": r["Tag"] for r in tag_row_counts.iter_rows(named=True)},
        value=[], label="Evento do Log (Tag; vazio = todos):",
    )
    exp_text_search = mo.ui.text(
        value="", placeholder="Ex: CLARO 3500, 50450314308, PETROBRAS...",
        label="Busca livre (todas as colunas; todos os termos devem aparecer):", full_width=True,
    )
    exp_color_by = mo.ui.dropdown(
        options={"Categoria do Log": "Categoria_Principal", "Tag/Pipeline do Log": "Pipeline", "Fonte": "Fonte",
                 "Serviço": "Servico_Desc", "Faixa de Frequência": "Faixa", "UF": "UF"},
        value="Categoria do Log", label="Colorir por:",
    )
    exp_page_size = mo.ui.slider(start=100, stop=10000, step=100, value=2000, label="Pontos por página:")
    exp_cluster = mo.ui.checkbox(value=False, label="Agrupar marcadores (MarkerCluster)")
    return (
        exp_cluster,
        exp_color_by,
        exp_fonte_select,
        exp_freq_max,
        exp_freq_min,
        exp_log_tags,
        exp_page_size,
        exp_serv_select,
        exp_text_search,
        exp_uf_select,
    )


@app.cell(hide_code=True)
def exp_mun(ALL_OPTION, exp_uf_select, mun_uf_pairs):
    _pairs = mun_uf_pairs if exp_uf_select.value == ALL_OPTION else mun_uf_pairs.filter(pl.col("UF") == exp_uf_select.value)
    exp_mun_select = mo.ui.dropdown(
        options={ALL_OPTION: None} | {f"{m} / {u}": (m, u) for m, u in _pairs.iter_rows()},
        value=ALL_OPTION, searchable=True, label="Município declarado:",
    )
    return (exp_mun_select,)


@app.cell
def exp_match(
    ALL_OPTION,
    column_uniques,
    exp_fonte_select,
    exp_freq_max,
    exp_freq_min,
    exp_log_tags,
    exp_mun_select,
    exp_serv_select,
    exp_text_search,
    exp_uf_select,
    full_lf,
    search_expr,
    tag_templates,
    templates_expr,
):
    _filters = []
    if exp_serv_select.value:
        _filters.append(pl.col("Serviço").cast(pl.String).is_in(list(exp_serv_select.value)))
    if exp_fonte_select.value:
        _filters.append(pl.col("Fonte").cast(pl.String).is_in(list(exp_fonte_select.value)))
    if exp_uf_select.value != ALL_OPTION:
        _filters.append(pl.col("UF") == exp_uf_select.value)
    if exp_mun_select.value is not None:
        _filters += [pl.col("Município") == exp_mun_select.value[0], pl.col("UF") == exp_mun_select.value[1]]
    if exp_freq_min.value > 0 or exp_freq_max.value < 100000:
        _filters.append(pl.col("Freq_MHz").is_between(exp_freq_min.value, exp_freq_max.value))
    if exp_log_tags.value:
        _filters.append(templates_expr(set().union(*(tag_templates[t] for t in exp_log_tags.value))))
    _search = search_expr(exp_text_search.value, column_uniques)
    if _search is not None:
        _filters.append(_search)

    exp_filtered_lf = full_lf.filter(pl.all_horizontal(_filters)) if _filters else full_lf
    # Deterministic UF/Município order so every matched row lands on exactly one, geographically coherent page
    exp_matched = (
        exp_filtered_lf.select("Linha", pl.col("UF").cast(pl.String), pl.col("Município").cast(pl.String))
        .sort("UF", "Município", "Linha")
        .collect()
    )
    return (exp_matched,)


@app.cell(hide_code=True)
def exp_page_ctrl(exp_matched, exp_page_size):
    exp_n_pages = max(1, -(-exp_matched.height // int(exp_page_size.value)))
    exp_page = mo.ui.number(start=1, stop=exp_n_pages, step=1, value=1, label=f"Página (de {exp_n_pages:,}):")
    return exp_n_pages, exp_page


@app.cell(hide_code=True)
def exp_view(
    LOG_CATEGORY_META,
    SERVICO_NOMES,
    band_expr,
    build_points_map,
    color_mapping,
    exp_cluster,
    exp_color_by,
    exp_fonte_select,
    exp_freq_max,
    exp_freq_min,
    exp_log_tags,
    exp_matched,
    exp_mun_select,
    exp_n_pages,
    exp_page,
    exp_page_size,
    exp_serv_select,
    exp_text_search,
    exp_uf_select,
    folium_iframe,
    full_lf,
    log_templates,
    n_rows,
):
    _size = int(exp_page_size.value)
    _page_no = min(max(int(exp_page.value or 1), 1), exp_n_pages)
    _start = (_page_no - 1) * _size
    _ids = exp_matched["Linha"].slice(_start, _size)

    _points = (
        full_lf.filter(pl.col("Linha").is_in(_ids.implode()))
        .with_columns(pl.col("Log").cast(pl.String), band_expr().alias("Faixa"))
        .collect()
        .join(log_templates.select("Log", "Categoria_Principal", "Pipeline", "Resumo_Log"), on="Log", how="left")
        .with_columns(
            pl.col("Serviço").cast(pl.String).replace_strict(SERVICO_NOMES, default=pl.col("Serviço").cast(pl.String)).alias("Servico_Desc"),
            *[pl.col(c).cast(pl.String) for c in ("Fonte", "UF")],
        )
        .sort("UF", "Município", "Linha")
    )
    _color_col = exp_color_by.value
    _values = _points.group_by(_color_col).len().sort("len", descending=True)[_color_col].to_list()
    _fmap = build_points_map(
        _points,
        _color_col,
        color_mapping(_values, LOG_CATEGORY_META if _color_col == "Categoria_Principal" else None),
        ["Entidade", "Fistel", "Estação", "Servico_Desc", "Frequência", "Município", "UF", "Fonte", "Status",
         "Latitude", "Longitude", "Resumo_Log"],
        cluster=exp_cluster.value,
    )
    _n_total = exp_matched.height
    _ufs = _points["UF"]
    _uf_span = f" (UF {_ufs.min()}–{_ufs.max()})" if _points.height else ""
    _table_cols = ["Linha", "Fonte", "Entidade", "Fistel", "Estação", "Servico_Desc", "Frequência", "Faixa",
                   "Município", "UF", "Latitude", "Longitude", "Status", "Validade_RF", "Categoria_Principal", "Resumo_Log"]

    mo.vstack(
        [
            mo.md(
                f"""
    ## 4. Explorador Geral da Base Completa & Mapa Folium Paginado
    Filtre por **Serviço**, **Fonte**, **UF/Município**, **faixa de Frequência**, **evento do `Log`** e/ou **busca livre** (aplicada às *categorias distintas* de cada coluna e mapeada de volta via `is_in`).
    O resultado é ordenado por UF/Município e dividido em páginas; **nenhum ponto é descartado** — percorra as páginas para cobrir todo o conjunto.
    """
            ),
            mo.hstack([exp_serv_select, exp_fonte_select, exp_log_tags], widths="equal"),
            mo.hstack([exp_uf_select, exp_mun_select, exp_freq_min, exp_freq_max], justify="start"),
            exp_text_search,
            mo.hstack([exp_color_by, exp_page_size, exp_page, exp_cluster], justify="start"),
            mo.callout(
                mo.md(
                    f"**{_n_total:,}** registros ({_n_total / n_rows * 100:.3f}%) atendem aos filtros em **{exp_n_pages:,}** página(s). "
                    f"Página **{_page_no:,}**: registros **{min(_start + 1, _n_total):,}–{_start + _points.height:,}**{_uf_span}. "
                    f"Pontos com coordenadas idênticas (ex.: ERBs/SMP com múltiplas frequências) ficam sobrepostos — use a tabela ou o cluster."
                ),
                kind="info",
            ),
            mo.ui.tabs(
                {
                    "🗺️ Mapa (página atual)": folium_iframe(_fmap, "640px"),
                    "🧾 Tabela (página atual)": mo.ui.table(_points.select(_table_cols), pagination=True, page_size=15),
                    "📊 Composição (página atual)": mo.ui.table(
                        _points.group_by(_color_col).agg(pl.len().alias("Registros")).sort("Registros", descending=True),
                        pagination=False,
                    ),
                }
            ),
        ]
    )


@app.cell
def disc_calc(DATA_DIR, SERVICO_NOMES, full_lf, num_expr, parse_log_templates):
    _inv, _frames = [], []
    for _f in sorted(DATA_DIR.glob("*_discarded.parquet")):
        _df = pl.read_parquet(_f)
        _origem = _f.name.removesuffix("_discarded.parquet")
        _inv.append({"Arquivo": _f.name, "Origem": _origem, "Registros": _df.height, "Colunas": _df.width,
                     "Tamanho_KB": round(_f.stat().st_size / 1024, 1)})
        if _df.height:
            # Schemas differ per source (SMP/SRD keep their raw columns): normalise to String before stacking
            _frames.append(
                _df.with_columns(pl.all().cast(pl.String)).with_columns(
                    pl.lit(_f.name).alias("Arquivo"), pl.lit(_origem).alias("Origem")
                )
            )
    discard_inventory_df = pl.DataFrame(_inv).sort("Registros", descending=True)
    _all = pl.concat(_frames, how="diagonal_relaxed").with_columns(pl.col("Log").fill_null("[]"))

    # Stable record key: source id when available, otherwise a hash of the raw record
    _key_cols = [c for c in ("_id", "Id") if c in _all.columns]
    _raw_cols = [c for c in _all.columns if c not in ("Arquivo", "Origem")]
    _all = _all.with_columns(
        pl.coalesce(*[pl.col(c) for c in _key_cols], pl.concat_str(_raw_cols, separator="|", ignore_nulls=True).hash().cast(pl.String))
        .alias("Chave")
    )
    _presence = _all.group_by("Chave").agg(
        pl.col("Arquivo").unique().sort().str.join(", ").alias("Presente_em"),
        pl.col("Arquivo").n_unique().alias("Num_Arquivos"),
    )

    disc_templates, disc_events = parse_log_templates(
        _all.group_by("Log").agg(pl.len().cast(pl.Int64).alias("Linhas"))
    )
    _reason = (
        disc_events.filter(pl.col("Categoria") == "Descarte")
        .group_by("Log")
        .agg(
            pl.col("Processamento").unique().sort().str.join(" + ").alias("Motivo_Descarte"),
            pl.col("Tag").unique().sort().str.join(" + ").alias("Tag_Motivo"),
            pl.col("Coluna").first().alias("Coluna_Motivo"),
            pl.col("Original").first().alias("Valor_Motivo"),
        )
    )

    # Is the discarded station (and the same frequency) still present in estacoes_full?
    _estacoes = _all["Estação"].drop_nulls().unique() if "Estação" in _all.columns else pl.Series([], dtype=pl.String)
    _in_full = (
        full_lf.filter(pl.col("Estação").cast(pl.String).is_in(_estacoes.implode()))
        .select(pl.col("Estação").cast(pl.String), pl.col("Freq_MHz").round(4))
        .unique()
        .collect()
    )

    discarded_df = (
        _all.join(_presence, on="Chave", how="left")
        .join(disc_templates.select("Log", "Pipeline", "Resumo_Log"), on="Log", how="left")
        .join(_reason, on="Log", how="left")
        .with_columns(
            pl.col("Motivo_Descarte").fill_null("Sem motivo explícito no Log"),
            pl.col("Tag_Motivo").fill_null("SEM_MOTIVO"),
            pl.col("Valor_Motivo").fill_null(""),
            num_expr("Latitude").alias("Lat"),
            num_expr("Longitude").alias("Lon"),
            num_expr("Frequência").alias("Freq_MHz"),
            pl.col("Estação").is_in(_in_full["Estação"].implode()).alias("Estacao_no_Full"),
        )
        .join(
            _in_full.with_columns(pl.lit(True).alias("Estacao_Freq_no_Full")),
            left_on=["Estação", pl.col("Freq_MHz").round(4)], right_on=["Estação", "Freq_MHz"], how="left",
        )
        .with_columns(
            pl.col("Estacao_Freq_no_Full").fill_null(False),
            pl.col("Serviço").replace_strict(SERVICO_NOMES, default=pl.col("Serviço")).alias("Servico_Desc"),
        )
    )
    return disc_templates, discard_inventory_df, discarded_df


@app.cell(hide_code=True)
def disc_controls(ALL_OPTION, discard_inventory_df, discarded_df):
    disc_dedupe = mo.ui.checkbox(value=True, label="Remover duplicados entre arquivos (ex.: smp ⊂ smp_full)")
    disc_file_select = mo.ui.multiselect(
        options={f"{r['Arquivo']} [{r['Registros']:,}]": r["Arquivo"] for r in discard_inventory_df.filter(pl.col("Registros") > 0).iter_rows(named=True)},
        value=[], label="Arquivo (vazio = todos):",
    )
    _reasons = discarded_df.group_by("Motivo_Descarte").len().sort("len", descending=True)
    disc_reason_select = mo.ui.multiselect(
        options={f"{m} [{n:,}]": m for m, n in _reasons.iter_rows()}, value=[], label="Motivo do descarte (vazio = todos):",
    )
    disc_uf_select = mo.ui.dropdown(
        options=[ALL_OPTION] + sorted(discarded_df["UF"].drop_nulls().unique().to_list()), value=ALL_OPTION, label="UF:",
    )
    disc_search = mo.ui.text(value="", placeholder="Entidade, Estação, Fistel, Município, Status...", label="Busca:")
    disc_color_by = mo.ui.dropdown(
        options={"Motivo do descarte": "Motivo_Descarte", "Valor que motivou": "Valor_Motivo", "Origem": "Origem", "UF": "UF"},
        value="Motivo do descarte", label="Colorir por:",
    )
    disc_page_size = mo.ui.slider(start=100, stop=5000, step=100, value=1000, label="Pontos por página:")
    return (
        disc_color_by,
        disc_dedupe,
        disc_file_select,
        disc_page_size,
        disc_reason_select,
        disc_search,
        disc_uf_select,
    )


@app.cell
def disc_match(
    ALL_OPTION,
    disc_dedupe,
    disc_file_select,
    disc_reason_select,
    disc_search,
    disc_uf_select,
    discarded_df,
):
    _df = discarded_df
    if disc_dedupe.value:
        # Prefer the "_full" variant (superset) when a record appears in several files
        _df = _df.sort(pl.col("Arquivo").str.contains("_full"), descending=True).unique("Chave", keep="first", maintain_order=True)
    if disc_file_select.value:
        _df = _df.filter(pl.col("Arquivo").is_in(list(disc_file_select.value)))
    if disc_reason_select.value:
        _df = _df.filter(pl.col("Motivo_Descarte").is_in(list(disc_reason_select.value)))
    if disc_uf_select.value != ALL_OPTION:
        _df = _df.filter(pl.col("UF") == disc_uf_select.value)
    _q = disc_search.value.strip().lower()
    if _q:
        _cols = [c for c in ("Entidade", "Estação", "Fistel", "Município", "Status", "Frequência", "Valor_Motivo") if c in _df.columns]
        _df = _df.filter(pl.any_horizontal([pl.col(c).str.to_lowercase().str.contains(_q, literal=True) for c in _cols]))
    disc_filtered = _df.sort("UF", "Município", "Chave", nulls_last=True)
    return (disc_filtered,)


@app.cell(hide_code=True)
def disc_page_ctrl(disc_filtered, disc_page_size):
    disc_n_pages = max(1, -(-disc_filtered.height // int(disc_page_size.value)))
    disc_page = mo.ui.number(start=1, stop=disc_n_pages, step=1, value=1, label=f"Página (de {disc_n_pages:,}):")
    return disc_n_pages, disc_page


@app.cell(hide_code=True)
def disc_view(
    build_points_map,
    color_mapping,
    disc_color_by,
    disc_dedupe,
    disc_file_select,
    disc_filtered,
    disc_n_pages,
    disc_page,
    disc_page_size,
    disc_reason_select,
    disc_search,
    disc_templates,
    disc_uf_select,
    discard_inventory_df,
    folium_iframe,
    kpi_row,
):
    _d = disc_filtered
    _n = max(_d.height, 1)
    _raw_total = int(discard_inventory_df["Registros"].sum())
    _empty = discard_inventory_df.filter(pl.col("Registros") == 0)["Arquivo"].to_list()

    _reason_src = _d.group_by("Motivo_Descarte", "Origem").agg(pl.len().alias("Registros"))
    _reason_tbl = (
        _d.group_by("Motivo_Descarte", "Tag_Motivo", "Coluna_Motivo")
        .agg(
            pl.len().alias("Registros"),
            pl.col("Origem").unique().sort().str.join(", ").alias("Origens"),
            pl.col("Valor_Motivo").n_unique().alias("Valores_Distintos"),
            pl.col("Entidade").n_unique().alias("Entidades"),
            pl.col("Estacao_no_Full").mean().mul(100).round(1).alias("%_Estacao_no_Full"),
            pl.col("Estacao_Freq_no_Full").mean().mul(100).round(1).alias("%_Estacao+Freq_no_Full"),
        )
        .sort("Registros", descending=True)
    )
    _val = _d.group_by("Motivo_Descarte", "Coluna_Motivo", "Valor_Motivo").agg(pl.len().alias("Registros")).sort("Registros", descending=True)

    _reason_chart = (
        alt.Chart(_reason_src)
        .mark_bar()
        .encode(
            x=alt.X("Registros:Q"), y=alt.Y("Motivo_Descarte:N", sort="-x", title=None, axis=alt.Axis(labelLimit=500)),
            color=alt.Color("Origem:N"), tooltip=["Motivo_Descarte", "Origem", alt.Tooltip("Registros:Q", format=",")],
        )
        .properties(height=max(120, 40 * _reason_src["Motivo_Descarte"].n_unique()), title="Registros descartados por motivo e origem")
    )
    _val_chart = (
        alt.Chart(_val.head(30).with_columns((pl.col("Coluna_Motivo") + " = " + pl.col("Valor_Motivo")).alias("Valor")))
        .mark_bar()
        .encode(
            x=alt.X("Registros:Q", scale=alt.Scale(type="symlog")), y=alt.Y("Valor:N", sort="-x", title=None),
            color=alt.Color("Motivo_Descarte:N", legend=alt.Legend(orient="bottom", labelLimit=500)),
            tooltip=["Motivo_Descarte", "Coluna_Motivo", "Valor_Motivo", alt.Tooltip("Registros:Q", format=",")],
        )
        .properties(height=500, title="Top 30 valores que motivaram o descarte (valor 'Original' do passo de descarte)")
    )

    _by = lambda cols: (
        _d.group_by([c for c in cols if c in _d.columns]).agg(pl.len().alias("Registros")).sort("Registros", descending=True)
    )
    _specific = {}
    if "Canalização" in _d.columns and _d["Canalização"].drop_nulls().len():
        _specific["📶 SMP: Faixa × Canalização × Tecnologia"] = mo.ui.table(
            _by(["Motivo_Descarte", "Faixas", "Canalização", "Tecnologia", "Frequência"]).drop_nulls("Faixas"), pagination=True, page_size=15
        )
    if "locpb" in _d.columns and _d["locpb"].drop_nulls().len():
        _specific["📺 SRD: Status × Serviço × Classe"] = mo.ui.table(
            _by(["Status", "Servico_Desc", "Classe"]).filter(pl.col("Status").is_not_null()), pagination=True, page_size=15
        )

    _size = int(disc_page_size.value)
    _page_no = min(max(int(disc_page.value or 1), 1), disc_n_pages)
    _page = _d.slice((_page_no - 1) * _size, _size)
    _cc = disc_color_by.value
    _vals = _page.group_by(_cc).len().sort("len", descending=True)[_cc].to_list()
    _fmap = build_points_map(
        _page.with_columns(pl.col(_cc).fill_null("(nulo)")), _cc, color_mapping(_vals),
        ["Arquivo", "Motivo_Descarte", "Valor_Motivo", "Entidade", "Fistel", "Estação", "Frequência", "Município", "UF",
         "Status", "Estacao_no_Full", "Resumo_Log"],
    )
    _table_cols = [c for c in ("Arquivo", "Presente_em", "Motivo_Descarte", "Coluna_Motivo", "Valor_Motivo", "Entidade", "Fistel",
                                "Estação", "Servico_Desc", "Frequência", "Município", "UF", "Latitude", "Longitude", "Status",
                                "Canalização", "Faixas", "Tecnologia", "Estacao_no_Full", "Estacao_Freq_no_Full", "Resumo_Log")
                   if c in _d.columns]

    mo.vstack(
        [
            mo.md(
                f"""
    ## 5. Registros Descartados — Por que foram removidos?
    Os descartes são gravados **por fonte** em `*_discarded.parquet` ({len(discard_inventory_df)} arquivos, {_raw_total:,} linhas brutas; vazios: `{', '.join(_empty) or '—'}`).
    Eles foram unidos (`concat` diagonal, colunas específicas de cada fonte preservadas) e o **motivo** foi extraído do passo do `Log` classificado como *Descarte*
    (o valor `Original` desse passo indica o dado que motivou a exclusão — ex.: `Status` não monitorável no SRD, `Frequência` em canalização *uplink*/inválida no SMP).
    A coluna `Estacao_no_Full` indica se a mesma `Estação` continua presente em `estacoes_full.parquet` (ex.: a ERB segue na base com outras frequências).
    """
            ),
            mo.hstack([disc_dedupe, disc_file_select, disc_reason_select], widths=[1, 1, 2]),
            mo.hstack([disc_uf_select, disc_search, disc_color_by, disc_page_size, disc_page], justify="start"),
            kpi_row(
                [
                    ("Linhas brutas (todos os arquivos)", f"{_raw_total:,}", f"{discard_inventory_df.filter(pl.col('Registros') > 0).height} arquivos não vazios"),
                    ("Registros no filtro", f"{_d.height:,}", "deduplicados" if disc_dedupe.value else "com duplicatas entre arquivos"),
                    ("Motivos distintos", f"{_d['Motivo_Descarte'].n_unique()}", f"{_d['Entidade'].n_unique():,} entidades afetadas"),
                    ("Estação ainda no Full", f"{_d['Estacao_no_Full'].sum():,}", f"{_d['Estacao_no_Full'].sum() / _n * 100:.1f}% dos descartados ({_d['Estacao_Freq_no_Full'].sum():,} c/ mesma freq.)"),
                ]
            ),
            mo.ui.tabs(
                {
                    "❓ Motivos": mo.vstack([_reason_chart, mo.ui.table(_reason_tbl, pagination=False)]),
                    "🔬 Valores que Motivaram": mo.vstack([_val_chart, mo.ui.table(_val, pagination=True, page_size=15)]),
                    **_specific,
                    "🏢 Entidades & UFs": mo.hstack(
                        [mo.ui.table(_by(["Entidade", "Motivo_Descarte"]), pagination=True, page_size=15),
                         mo.ui.table(_by(["UF"]), pagination=True, page_size=15)],
                        widths=[3, 1],
                    ),
                    "🗺️ Mapa (página)": mo.vstack(
                        [mo.md(f"Página **{_page_no}/{disc_n_pages}** — {_page.height:,} registros."), folium_iframe(_fmap, "600px")]
                    ),
                    "🧾 Registros (página)": mo.ui.table(_page.select(_table_cols), pagination=True, page_size=15),
                    "📁 Inventário de Arquivos": mo.ui.table(discard_inventory_df, pagination=False),
                    "🧩 Templates de Log dos Descartes": mo.ui.table(
                        disc_templates.select("Linhas", "Pipeline", "Resumo_Log").sort("Linhas", descending=True), pagination=True, page_size=15
                    ),
                }
            ),
        ]
    )


if __name__ == "__main__":
    app.run()
