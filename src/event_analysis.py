from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURACOES
# ============================================================

RAIZ = Path(__file__).resolve().parents[1]

ARQUIVO_BASE = (
    RAIZ
    / "data"
    / "chuva_cota"
    / "dataset_chuva_cota_modelagem.csv"
)

PASTA_SAIDA = (
    RAIZ
    / "data"
    / "analise_eventos"
)

PASTA_SAIDA.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CARREGAMENTO
# ============================================================

def carregar_base():

    if not ARQUIVO_BASE.exists():
        raise FileNotFoundError(
            f"Base nao encontrada:\n{ARQUIVO_BASE}"
        )

    df = pd.read_csv(
        ARQUIVO_BASE,
        encoding="utf-8-sig",
    )

    df["data"] = pd.to_datetime(
        df["data"],
        errors="coerce",
    )

    df = (
        df
        .sort_values("data")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# PREPARACAO DOS EVENTOS
# ============================================================

def preparar_eventos(df):

    base = df.copy()

    # --------------------------------------------------------
    # A cota_Dmais1 ja foi criada no processamento anterior
    # respeitando o calendario diario.
    # --------------------------------------------------------

    if "cota_Dmais1" not in base.columns:
        raise ValueError(
            "A coluna cota_Dmais1 nao existe na base."
        )

    # Variacao da cota entre hoje e o dia seguinte
    base["delta_cota_Dmais1"] = (
        base["cota_Dmais1"]
        - base["cota"]
    )

    # --------------------------------------------------------
    # LIMIARES CALCULADOS NA PROPRIA SERIE
    #
    # Estes limiares sao EXPLORATORIOS.
    # Nao representam alerta ou inundacao oficial.
    # --------------------------------------------------------

    cotas_validas = (
        base["cota"]
        .dropna()
    )

    deltas_validos = (
        base["delta_cota_Dmais1"]
        .dropna()
    )

    p90_cota = (
        cotas_validas
        .quantile(0.90)
    )

    p95_cota = (
        cotas_validas
        .quantile(0.95)
    )

    p90_delta = (
        deltas_validos
        .quantile(0.90)
    )

    p95_delta = (
        deltas_validos
        .quantile(0.95)
    )

    return (
        base,
        p90_cota,
        p95_cota,
        p90_delta,
        p95_delta,
    )


# ============================================================
# IDENTIFICACAO DOS EVENTOS
# ============================================================

def identificar_eventos(
    base,
    p90_cota,
    p90_delta,
):

    eventos = base[
        (
            base["cota_Dmais1"]
            >= p90_cota
        )
        |
        (
            base["delta_cota_Dmais1"]
            >= p90_delta
        )
    ].copy()

    eventos["evento_cota_elevada"] = (
        eventos["cota_Dmais1"]
        >= p90_cota
    )

    eventos["evento_subida_rapida"] = (
        eventos["delta_cota_Dmais1"]
        >= p90_delta
    )

    eventos["tipo_evento"] = np.select(
        [
            (
                eventos["evento_cota_elevada"]
                & eventos["evento_subida_rapida"]
            ),
            eventos["evento_cota_elevada"],
            eventos["evento_subida_rapida"],
        ],
        [
            "Cota elevada + subida rapida",
            "Cota elevada",
            "Subida rapida",
        ],
        default="Outro",
    )

    eventos["data_evento"] = (
        eventos["data"]
        + pd.Timedelta(days=1)
    )

    return eventos


# ============================================================
# TABELA DETALHADA
# ============================================================

def criar_tabela_eventos(eventos):

    colunas_desejadas = [
        "data",
        "data_evento",
        "cota",
        "cota_Dmais1",
        "delta_cota_Dmais1",
        "precipitacao_mm",
        "chuva_lag_1d",
        "chuva_lag_2d",
        "chuva_lag_3d",
        "chuva_lag_4d",
        "chuva_lag_5d",
        "chuva_lag_6d",
        "chuva_lag_7d",
        "chuva_acum_2d",
        "chuva_acum_3d",
        "chuva_acum_5d",
        "chuva_acum_7d",
        "evento_cota_elevada",
        "evento_subida_rapida",
        "tipo_evento",
    ]

    colunas_existentes = [
        coluna
        for coluna in colunas_desejadas
        if coluna in eventos.columns
    ]

    tabela = eventos[
        colunas_existentes
    ].copy()

    tabela = tabela.sort_values(
        "delta_cota_Dmais1",
        ascending=False,
    )

    return tabela


# ============================================================
# RESUMO DOS TIPOS DE EVENTO
# ============================================================

def resumir_eventos(eventos):

    resumo = (
        eventos
        .groupby(
            "tipo_evento",
            dropna=False,
        )
        .agg(
            numero_eventos=(
                "tipo_evento",
                "size",
            ),
            cota_inicial_media=(
                "cota",
                "mean",
            ),
            cota_evento_media=(
                "cota_Dmais1",
                "mean",
            ),
            subida_media=(
                "delta_cota_Dmais1",
                "mean",
            ),
            subida_maxima=(
                "delta_cota_Dmais1",
                "max",
            ),
            chuva_dia_origem_media=(
                "precipitacao_mm",
                "mean",
            ),
            chuva_2d_media=(
                "chuva_acum_2d",
                "mean",
            ),
            chuva_3d_media=(
                "chuva_acum_3d",
                "mean",
            ),
            chuva_5d_media=(
                "chuva_acum_5d",
                "mean",
            ),
            chuva_7d_media=(
                "chuva_acum_7d",
                "mean",
            ),
        )
        .reset_index()
    )

    return resumo


# ============================================================
# COMPARACAO EVENTO X NAO EVENTO
# ============================================================

def comparar_eventos_nao_eventos(
    base,
    p90_delta,
):

    analise = base.dropna(
        subset=[
            "delta_cota_Dmais1",
            "precipitacao_mm",
            "chuva_acum_2d",
            "chuva_acum_3d",
            "chuva_acum_5d",
            "chuva_acum_7d",
            "cota",
        ]
    ).copy()

    analise["grupo"] = np.where(
        analise["delta_cota_Dmais1"]
        >= p90_delta,
        "Subida rapida",
        "Demais dias",
    )

    resumo = (
        analise
        .groupby("grupo")
        .agg(
            n=(
                "grupo",
                "size",
            ),
            cota_media=(
                "cota",
                "mean",
            ),
            delta_medio=(
                "delta_cota_Dmais1",
                "mean",
            ),
            chuva_dia_media=(
                "precipitacao_mm",
                "mean",
            ),
            chuva_2d_media=(
                "chuva_acum_2d",
                "mean",
            ),
            chuva_3d_media=(
                "chuva_acum_3d",
                "mean",
            ),
            chuva_5d_media=(
                "chuva_acum_5d",
                "mean",
            ),
            chuva_7d_media=(
                "chuva_acum_7d",
                "mean",
            ),
        )
        .reset_index()
    )

    return resumo


# ============================================================
# GRAFICO 1
# ============================================================

def grafico_maiores_subidas(
    tabela_eventos,
    p90_delta,
):

    maiores = (
        tabela_eventos
        .dropna(
            subset=[
                "delta_cota_Dmais1"
            ]
        )
        .sort_values(
            "delta_cota_Dmais1",
            ascending=False,
        )
        .head(25)
        .copy()
    )

    if maiores.empty:
        return

    plt.figure(
        figsize=(15, 8)
    )

    plt.bar(
        maiores[
            "data_evento"
        ].dt.strftime("%d/%m/%Y"),
        maiores[
            "delta_cota_Dmais1"
        ],
    )

    plt.axhline(
        p90_delta,
        linestyle="--",
        linewidth=1.2,
        label="P90 da variacao de cota",
    )

    plt.xlabel(
        "Data da cota em D+1"
    )

    plt.ylabel(
        "Variacao da cota"
    )

    plt.title(
        "Maiores elevacoes de cota em 1 dia"
    )

    plt.xticks(
        rotation=90
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    plt.tight_layout()

    plt.savefig(
        PASTA_SAIDA
        / "01_maiores_subidas_cota.png",
        dpi=180,
    )

    plt.close()


# ============================================================
# GRAFICO 2
# ============================================================

def grafico_chuva_delta(
    base,
    p90_delta,
):

    dados = base.dropna(
        subset=[
            "chuva_acum_3d",
            "delta_cota_Dmais1",
        ]
    ).copy()

    subida_rapida = (
        dados[
            "delta_cota_Dmais1"
        ]
        >= p90_delta
    )

    plt.figure(
        figsize=(10, 8)
    )

    plt.scatter(
        dados.loc[
            ~subida_rapida,
            "chuva_acum_3d",
        ],
        dados.loc[
            ~subida_rapida,
            "delta_cota_Dmais1",
        ],
        alpha=0.35,
        label="Demais dias",
    )

    plt.scatter(
        dados.loc[
            subida_rapida,
            "chuva_acum_3d",
        ],
        dados.loc[
            subida_rapida,
            "delta_cota_Dmais1",
        ],
        alpha=0.8,
        label="Subidas rapidas",
    )

    plt.axhline(
        p90_delta,
        linestyle="--",
        linewidth=1.2,
        label="P90 da variacao",
    )

    plt.xlabel(
        "Precipitacao acumulada em 3 dias"
    )

    plt.ylabel(
        "Variacao da cota em D+1"
    )

    plt.title(
        "Chuva antecedente x variacao futura da cota"
    )

    plt.legend()

    plt.grid(
        alpha=0.25,
    )

    plt.tight_layout()

    plt.savefig(
        PASTA_SAIDA
        / "02_chuva3d_variacao_cota.png",
        dpi=180,
    )

    plt.close()


# ============================================================
# GRAFICO 3
# ============================================================

def grafico_perfil_chuva(
    eventos,
):

    eventos_rapidos = eventos[
        eventos[
            "evento_subida_rapida"
        ]
    ].copy()

    if eventos_rapidos.empty:
        return

    variaveis = [
        "chuva_lag_7d",
        "chuva_lag_6d",
        "chuva_lag_5d",
        "chuva_lag_4d",
        "chuva_lag_3d",
        "chuva_lag_2d",
        "chuva_lag_1d",
        "precipitacao_mm",
    ]

    variaveis = [
        coluna
        for coluna in variaveis
        if coluna in eventos_rapidos.columns
    ]

    medias = (
        eventos_rapidos[
            variaveis
        ]
        .mean()
    )

    labels = [
        "D-7",
        "D-6",
        "D-5",
        "D-4",
        "D-3",
        "D-2",
        "D-1",
        "D",
    ]

    labels = labels[
        -len(medias):
    ]

    plt.figure(
        figsize=(11, 6)
    )

    plt.bar(
        labels,
        medias.values,
    )

    plt.xlabel(
        "Dia em relacao a origem da previsao"
    )

    plt.ylabel(
        "Precipitacao media"
    )

    plt.title(
        "Perfil medio de chuva antes das subidas rapidas"
    )

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    plt.tight_layout()

    plt.savefig(
        PASTA_SAIDA
        / "03_perfil_chuva_subidas_rapidas.png",
        dpi=180,
    )

    plt.close()


# ============================================================
# GRAFICO 4
# ============================================================

def grafico_cota_inicial_delta(
    base,
    p90_delta,
):

    dados = base.dropna(
        subset=[
            "cota",
            "delta_cota_Dmais1",
        ]
    ).copy()

    subida = (
        dados[
            "delta_cota_Dmais1"
        ]
        >= p90_delta
    )

    plt.figure(
        figsize=(10, 8)
    )

    plt.scatter(
        dados.loc[
            ~subida,
            "cota",
        ],
        dados.loc[
            ~subida,
            "delta_cota_Dmais1",
        ],
        alpha=0.35,
        label="Demais dias",
    )

    plt.scatter(
        dados.loc[
            subida,
            "cota",
        ],
        dados.loc[
            subida,
            "delta_cota_Dmais1",
        ],
        alpha=0.8,
        label="Subidas rapidas",
    )

    plt.axhline(
        p90_delta,
        linestyle="--",
        linewidth=1.2,
        label="P90 da variacao",
    )

    plt.xlabel(
        "Cota no dia de origem"
    )

    plt.ylabel(
        "Variacao da cota em D+1"
    )

    plt.title(
        "Estado inicial do rio x variacao em D+1"
    )

    plt.legend()

    plt.grid(
        alpha=0.25,
    )

    plt.tight_layout()

    plt.savefig(
        PASTA_SAIDA
        / "04_cota_inicial_variacao.png",
        dpi=180,
    )

    plt.close()


# ============================================================
# EXECUCAO
# ============================================================

def main():

    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
    )

    print()
    print("=" * 78)
    print(
        "SOBERBOPREV - ANALISE DE EVENTOS"
    )
    print(
        "COTAS ELEVADAS E SUBIDAS RAPIDAS"
    )
    print("=" * 78)

    df = carregar_base()

    (
        base,
        p90_cota,
        p95_cota,
        p90_delta,
        p95_delta,
    ) = preparar_eventos(df)

    print()
    print("=" * 78)
    print(
        "LIMIARES ESTATISTICOS EXPLORATORIOS"
    )
    print("=" * 78)

    print()
    print(
        f"P90 da cota: {p90_cota:.2f}"
    )

    print(
        f"P95 da cota: {p95_cota:.2f}"
    )

    print(
        f"P90 da subida diaria: "
        f"{p90_delta:.2f}"
    )

    print(
        f"P95 da subida diaria: "
        f"{p95_delta:.2f}"
    )

    print()
    print(
        "ATENCAO: estes valores sao "
        "referencias estatisticas da serie."
    )

    print(
        "Eles nao representam niveis oficiais "
        "de atencao, alerta ou inundacao."
    )

    # --------------------------------------------------------
    # EVENTOS
    # --------------------------------------------------------

    eventos = identificar_eventos(
        base,
        p90_cota,
        p90_delta,
    )

    tabela_eventos = (
        criar_tabela_eventos(
            eventos
        )
    )

    resumo_eventos = (
        resumir_eventos(
            eventos
        )
    )

    comparacao = (
        comparar_eventos_nao_eventos(
            base,
            p90_delta,
        )
    )

    print()
    print("=" * 78)
    print(
        "QUANTIDADE DE EVENTOS"
    )
    print("=" * 78)
    print()

    print(
        eventos[
            "tipo_evento"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print("=" * 78)
    print(
        "RESUMO DOS EVENTOS"
    )
    print("=" * 78)
    print()

    print(
        resumo_eventos.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.2f}",
        )
    )

    print()
    print("=" * 78)
    print(
        "SUBIDAS RAPIDAS X DEMAIS DIAS"
    )
    print("=" * 78)
    print()

    print(
        comparacao.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.2f}",
        )
    )

    # --------------------------------------------------------
    # MAIORES SUBIDAS
    # --------------------------------------------------------

    maiores_subidas = (
        tabela_eventos[
            tabela_eventos[
                "delta_cota_Dmais1"
            ]
            > 0
        ]
        .sort_values(
            "delta_cota_Dmais1",
            ascending=False,
        )
        .head(25)
        .copy()
    )

    print()
    print("=" * 78)
    print(
        "25 MAIORES SUBIDAS DE COTA"
    )
    print("=" * 78)
    print()

    colunas_print = [
        "data",
        "data_evento",
        "cota",
        "cota_Dmais1",
        "delta_cota_Dmais1",
        "precipitacao_mm",
        "chuva_lag_1d",
        "chuva_lag_2d",
        "chuva_acum_3d",
        "chuva_acum_7d",
        "tipo_evento",
    ]

    colunas_print = [
        coluna
        for coluna in colunas_print
        if coluna in maiores_subidas.columns
    ]

    print(
        maiores_subidas[
            colunas_print
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.2f}",
        )
    )

    # --------------------------------------------------------
    # SALVAR
    # --------------------------------------------------------

    tabela_eventos.to_csv(
        PASTA_SAIDA
        / "eventos_detalhados.csv",
        index=False,
    )

    resumo_eventos.to_csv(
        PASTA_SAIDA
        / "resumo_tipos_evento.csv",
        index=False,
    )

    comparacao.to_csv(
        PASTA_SAIDA
        / "comparacao_subidas_rapidas_demais_dias.csv",
        index=False,
    )

    maiores_subidas.to_csv(
        PASTA_SAIDA
        / "25_maiores_subidas.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {
                "indicador":
                    "P90_cota",
                "valor":
                    p90_cota,
            },
            {
                "indicador":
                    "P95_cota",
                "valor":
                    p95_cota,
            },
            {
                "indicador":
                    "P90_delta_cota_Dmais1",
                "valor":
                    p90_delta,
            },
            {
                "indicador":
                    "P95_delta_cota_Dmais1",
                "valor":
                    p95_delta,
            },
        ]
    ).to_csv(
        PASTA_SAIDA
        / "limiares_estatisticos_eventos.csv",
        index=False,
    )

    # --------------------------------------------------------
    # GRAFICOS
    # --------------------------------------------------------

    grafico_maiores_subidas(
        tabela_eventos,
        p90_delta,
    )

    grafico_chuva_delta(
        base,
        p90_delta,
    )

    grafico_perfil_chuva(
        eventos
    )

    grafico_cota_inicial_delta(
        base,
        p90_delta,
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print(
        "ANALISE DE EVENTOS CONCLUIDA"
    )
    print("=" * 78)

    print()
    print(
        "Resultados salvos em:"
    )

    print(
        PASTA_SAIDA
    )

    print()
    print(
        "Nenhuma observacao da base "
        "original foi modificada."
    )


if __name__ == "__main__":
    main()