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
    / "modelo_baseline"
)

PASTA_SAIDA.mkdir(
    parents=True,
    exist_ok=True,
)

PROPORCAO_TREINO = 0.80

# Variaveis conhecidas no instante t
FEATURES = [
    "cota",
    "precipitacao_mm",
    "chuva_lag_1d",
    "chuva_acum_2d",
    "chuva_acum_3d",
]

# Alvo: cota do dia seguinte
TARGET = "cota_Dmais1"


# ============================================================
# REGRESSAO LINEAR OLS
# ============================================================

class RegressaoLinearOLS:

    def __init__(self):
        self.coeficientes = None

    def fit(self, X, y):

        X = np.asarray(
            X,
            dtype=float,
        )

        y = np.asarray(
            y,
            dtype=float,
        )

        # Adiciona intercepto
        X_design = np.column_stack(
            [
                np.ones(len(X)),
                X,
            ]
        )

        # Minimos quadrados via pseudoinversa
        self.coeficientes = (
            np.linalg.pinv(X_design)
            @ y
        )

        return self

    def predict(self, X):

        X = np.asarray(
            X,
            dtype=float,
        )

        X_design = np.column_stack(
            [
                np.ones(len(X)),
                X,
            ]
        )

        return (
            X_design
            @ self.coeficientes
        )


# ============================================================
# METRICAS
# ============================================================

def mae(observado, previsto):

    observado = np.asarray(
        observado,
        dtype=float,
    )

    previsto = np.asarray(
        previsto,
        dtype=float,
    )

    return np.mean(
        np.abs(
            observado - previsto
        )
    )


def rmse(observado, previsto):

    observado = np.asarray(
        observado,
        dtype=float,
    )

    previsto = np.asarray(
        previsto,
        dtype=float,
    )

    return np.sqrt(
        np.mean(
            (observado - previsto) ** 2
        )
    )


def r2(observado, previsto):

    observado = np.asarray(
        observado,
        dtype=float,
    )

    previsto = np.asarray(
        previsto,
        dtype=float,
    )

    ss_res = np.sum(
        (observado - previsto) ** 2
    )

    ss_tot = np.sum(
        (
            observado
            - np.mean(observado)
        ) ** 2
    )

    if ss_tot == 0:
        return np.nan

    return 1 - (
        ss_res / ss_tot
    )


def nse(observado, previsto):

    observado = np.asarray(
        observado,
        dtype=float,
    )

    previsto = np.asarray(
        previsto,
        dtype=float,
    )

    numerador = np.sum(
        (observado - previsto) ** 2
    )

    denominador = np.sum(
        (
            observado
            - np.mean(observado)
        ) ** 2
    )

    if denominador == 0:
        return np.nan

    return 1 - (
        numerador / denominador
    )


def calcular_metricas(
    observado,
    previsto,
):

    return {
        "MAE": mae(
            observado,
            previsto,
        ),
        "RMSE": rmse(
            observado,
            previsto,
        ),
        "R2": r2(
            observado,
            previsto,
        ),
        "NSE": nse(
            observado,
            previsto,
        ),
    }


# ============================================================
# CARREGAMENTO DA BASE
# ============================================================

def carregar_base():

    if not ARQUIVO_BASE.exists():

        raise FileNotFoundError(
            "Arquivo da base de modelagem "
            "nao encontrado:\n"
            f"{ARQUIVO_BASE}"
        )

    df = pd.read_csv(
        ARQUIVO_BASE,
        encoding="utf-8-sig",
    )

    df["data"] = pd.to_datetime(
        df["data"],
        errors="coerce",
    )

    colunas_necessarias = (
        ["data"]
        + FEATURES
        + [TARGET]
    )

    faltantes = [
        coluna
        for coluna in colunas_necessarias
        if coluna not in df.columns
    ]

    if faltantes:

        raise ValueError(
            "Colunas ausentes na base: "
            + ", ".join(faltantes)
        )

    df = (
        df
        .sort_values("data")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# PREPARACAO DAS AMOSTRAS
# ============================================================

def preparar_amostras(df):

    colunas = (
        ["data"]
        + FEATURES
        + [TARGET]
    )

    base = df[
        colunas
    ].copy()

    quantidade_inicial = len(base)

    # Nao interpolamos dados ausentes.
    base = base.dropna(
        subset=FEATURES + [TARGET]
    )

    base = (
        base
        .sort_values("data")
        .reset_index(drop=True)
    )

    quantidade_final = len(base)

    removidas = (
        quantidade_inicial
        - quantidade_final
    )

    return (
        base,
        quantidade_inicial,
        quantidade_final,
        removidas,
    )


# ============================================================
# DIVISAO CRONOLOGICA
# ============================================================

def dividir_treino_teste(base):

    n = len(base)

    indice_corte = int(
        np.floor(
            n * PROPORCAO_TREINO
        )
    )

    treino = (
        base
        .iloc[:indice_corte]
        .copy()
    )

    teste = (
        base
        .iloc[indice_corte:]
        .copy()
    )

    if treino.empty or teste.empty:

        raise ValueError(
            "Treino ou teste ficou vazio."
        )

    return treino, teste


# ============================================================
# COTAS ELEVADAS
# ============================================================

def avaliar_cotas_elevadas(
    treino,
    teste,
):

    # Limiar calculado exclusivamente no treino.
    limiar_p90 = (
        treino[TARGET]
        .quantile(0.90)
    )

    elevados = teste[
        teste[TARGET]
        >= limiar_p90
    ].copy()

    return elevados, limiar_p90


# ============================================================
# GRAFICO 1
# ============================================================

def grafico_serie(previsoes):

    plt.figure(
        figsize=(16, 7)
    )

    plt.plot(
        previsoes["data_alvo"],
        previsoes[
            "cota_observada_Dmais1"
        ],
        linewidth=1.5,
        label="Observado",
    )

    plt.plot(
        previsoes["data_alvo"],
        previsoes[
            "previsao_persistencia"
        ],
        linewidth=1.1,
        alpha=0.8,
        label="Persistencia",
    )

    plt.plot(
        previsoes["data_alvo"],
        previsoes[
            "previsao_regressao_linear"
        ],
        linewidth=1.1,
        alpha=0.8,
        label="Regressao linear",
    )

    plt.xlabel("Data")

    plt.ylabel("Cota")

    plt.title(
        "Previsao de cota D+1 - periodo de teste"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        PASTA_SAIDA
        / "01_previsao_Dmais1_teste.png",
        dpi=180,
    )

    plt.close()


# ============================================================
# GRAFICO 2
# ============================================================

def grafico_observado_previsto(
    previsoes,
):

    observado = previsoes[
        "cota_observada_Dmais1"
    ]

    previsto = previsoes[
        "previsao_regressao_linear"
    ]

    minimo = min(
        observado.min(),
        previsto.min(),
    )

    maximo = max(
        observado.max(),
        previsto.max(),
    )

    plt.figure(
        figsize=(8, 8)
    )

    plt.scatter(
        observado,
        previsto,
        alpha=0.65,
    )

    plt.plot(
        [minimo, maximo],
        [minimo, maximo],
        linestyle="--",
        linewidth=1.2,
        label="Previsao perfeita",
    )

    plt.xlabel(
        "Cota observada em D+1"
    )

    plt.ylabel(
        "Cota prevista em D+1"
    )

    plt.title(
        "Observado x previsto - regressao linear"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        PASTA_SAIDA
        / "02_observado_previsto_regressao.png",
        dpi=180,
    )

    plt.close()


# ============================================================
# GRAFICO 3
# ============================================================

def grafico_residuos(
    previsoes,
):

    residuos = (
        previsoes[
            "cota_observada_Dmais1"
        ]
        - previsoes[
            "previsao_regressao_linear"
        ]
    )

    plt.figure(
        figsize=(16, 6)
    )

    plt.scatter(
        previsoes["data_alvo"],
        residuos,
        s=22,
        alpha=0.7,
    )

    plt.axhline(
        0,
        linestyle="--",
        linewidth=1.2,
    )

    plt.xlabel("Data")

    plt.ylabel(
        "Residuo (observado - previsto)"
    )

    plt.title(
        "Residuos da regressao linear - D+1"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        PASTA_SAIDA
        / "03_residuos_regressao_Dmais1.png",
        dpi=180,
    )

    plt.close()


# ============================================================
# EXECUCAO PRINCIPAL
# ============================================================

def main():

    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
    )

    print()
    print("=" * 78)
    print(
        "SOBERBOPREV - PRIMEIRO MODELO BASELINE"
    )
    print(
        "PREVISAO DE COTA D+1"
    )
    print("=" * 78)

    # --------------------------------------------------------
    # 1. CARREGAMENTO
    # --------------------------------------------------------

    df = carregar_base()

    (
        base,
        quantidade_inicial,
        quantidade_final,
        removidas,
    ) = preparar_amostras(df)

    print()
    print("PREPARACAO DAS AMOSTRAS")
    print("-" * 78)

    print(
        f"Dias no calendario: "
        f"{quantidade_inicial}"
    )

    print(
        f"Amostras completas para D+1: "
        f"{quantidade_final}"
    )

    print(
        f"Linhas nao utilizadas por dados ausentes: "
        f"{removidas}"
    )

    print()
    print(
        "Nenhuma interpolacao foi realizada."
    )

    # --------------------------------------------------------
    # 2. DIVISAO TEMPORAL
    # --------------------------------------------------------

    treino, teste = (
        dividir_treino_teste(base)
    )

    print()
    print("=" * 78)
    print("DIVISAO CRONOLOGICA")
    print("=" * 78)

    print()
    print(
        f"Treino: {len(treino)} amostras"
    )

    print(
        "Periodo de treino: "
        f"{treino['data'].min().date()} "
        "a "
        f"{treino['data'].max().date()}"
    )

    print()
    print(
        f"Teste: {len(teste)} amostras"
    )

    print(
        "Periodo de teste: "
        f"{teste['data'].min().date()} "
        "a "
        f"{teste['data'].max().date()}"
    )

    # --------------------------------------------------------
    # 3. MATRIZES
    # --------------------------------------------------------

    X_treino = (
        treino[FEATURES]
        .values
    )

    y_treino = (
        treino[TARGET]
        .values
    )

    X_teste = (
        teste[FEATURES]
        .values
    )

    y_teste = (
        teste[TARGET]
        .values
    )

    # --------------------------------------------------------
    # 4. PERSISTENCIA
    # --------------------------------------------------------

    previsao_persistencia = (
        teste["cota"]
        .values
    )

    metricas_persistencia = (
        calcular_metricas(
            y_teste,
            previsao_persistencia,
        )
    )

    # --------------------------------------------------------
    # 5. REGRESSAO LINEAR
    # --------------------------------------------------------

    modelo = RegressaoLinearOLS()

    modelo.fit(
        X_treino,
        y_treino,
    )

    previsao_regressao = (
        modelo.predict(
            X_teste
        )
    )

    metricas_regressao = (
        calcular_metricas(
            y_teste,
            previsao_regressao,
        )
    )

    # --------------------------------------------------------
    # 6. RESULTADOS GERAIS
    # --------------------------------------------------------

    metricas = pd.DataFrame(
        [
            {
                "modelo":
                    "Persistencia",
                **metricas_persistencia,
            },
            {
                "modelo":
                    "Regressao Linear",
                **metricas_regressao,
            },
        ]
    )

    print()
    print("=" * 78)
    print(
        "DESEMPENHO NO PERIODO DE TESTE"
    )
    print("=" * 78)
    print()

    print(
        metricas.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    # --------------------------------------------------------
    # 7. COEFICIENTES
    # --------------------------------------------------------

    coeficientes = pd.DataFrame(
        {
            "variavel":
                ["intercepto"]
                + FEATURES,
            "coeficiente":
                modelo.coeficientes,
        }
    )

    print()
    print("=" * 78)
    print(
        "COEFICIENTES DA REGRESSAO LINEAR"
    )
    print("=" * 78)
    print()

    print(
        coeficientes.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    # --------------------------------------------------------
    # 8. EVENTOS ELEVADOS
    # --------------------------------------------------------

    elevados, limiar_p90 = (
        avaliar_cotas_elevadas(
            treino,
            teste,
        )
    )

    print()
    print("=" * 78)
    print(
        "AVALIACAO EM COTAS ELEVADAS"
    )
    print("=" * 78)

    print()
    print(
        "P90 calculado somente "
        f"no treino: {limiar_p90:.2f}"
    )

    print(
        "Eventos do teste acima "
        f"desse limiar: {len(elevados)}"
    )

    previsoes_teste = (
        teste.copy()
    )

    previsoes_teste[
        "persistencia"
    ] = previsao_persistencia

    previsoes_teste[
        "regressao"
    ] = previsao_regressao

    if len(elevados) > 0:

        elevados_avaliados = (
            previsoes_teste.loc[
                elevados.index
            ].copy()
        )

        met_persist_elevado = (
            calcular_metricas(
                elevados_avaliados[
                    TARGET
                ],
                elevados_avaliados[
                    "persistencia"
                ],
            )
        )

        met_reg_elevado = (
            calcular_metricas(
                elevados_avaliados[
                    TARGET
                ],
                elevados_avaliados[
                    "regressao"
                ],
            )
        )

        metricas_elevados = (
            pd.DataFrame(
                [
                    {
                        "modelo":
                            "Persistencia",
                        **met_persist_elevado,
                    },
                    {
                        "modelo":
                            "Regressao Linear",
                        **met_reg_elevado,
                    },
                ]
            )
        )

        print()
        print(
            metricas_elevados.to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.4f}",
            )
        )

    else:

        metricas_elevados = (
            pd.DataFrame()
        )

    # --------------------------------------------------------
    # 9. TABELA DAS PREVISOES
    # --------------------------------------------------------

    previsoes = pd.DataFrame(
        {
            "data_origem":
                teste["data"].values,

            "data_alvo":
                (
                    teste["data"]
                    + pd.Timedelta(days=1)
                ).values,

            "cota_atual":
                teste["cota"].values,

            "cota_observada_Dmais1":
                y_teste,

            "previsao_persistencia":
                previsao_persistencia,

            "previsao_regressao_linear":
                previsao_regressao,
        }
    )

    previsoes[
        "erro_persistencia"
    ] = (
        previsoes[
            "cota_observada_Dmais1"
        ]
        - previsoes[
            "previsao_persistencia"
        ]
    )

    previsoes[
        "erro_regressao_linear"
    ] = (
        previsoes[
            "cota_observada_Dmais1"
        ]
        - previsoes[
            "previsao_regressao_linear"
        ]
    )

    # --------------------------------------------------------
    # 10. MAIORES ERROS
    # --------------------------------------------------------

    maiores_erros = (
        previsoes.copy()
    )

    maiores_erros[
        "erro_absoluto_regressao"
    ] = (
        maiores_erros[
            "erro_regressao_linear"
        ]
        .abs()
    )

    maiores_erros = (
        maiores_erros
        .sort_values(
            "erro_absoluto_regressao",
            ascending=False,
        )
        .head(20)
    )

    print()
    print("=" * 78)
    print(
        "20 MAIORES ERROS DA REGRESSAO NO TESTE"
    )
    print("=" * 78)
    print()

    print(
        maiores_erros[
            [
                "data_origem",
                "data_alvo",
                "cota_atual",
                "cota_observada_Dmais1",
                "previsao_regressao_linear",
                "erro_regressao_linear",
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.2f}",
        )
    )

    # --------------------------------------------------------
    # 11. SALVAR RESULTADOS
    # --------------------------------------------------------

    metricas.to_csv(
        PASTA_SAIDA
        / "metricas_Dmais1.csv",
        index=False,
    )

    coeficientes.to_csv(
        PASTA_SAIDA
        / "coeficientes_regressao_Dmais1.csv",
        index=False,
    )

    previsoes.to_csv(
        PASTA_SAIDA
        / "previsoes_Dmais1_teste.csv",
        index=False,
    )

    maiores_erros.to_csv(
        PASTA_SAIDA
        / "maiores_erros_Dmais1.csv",
        index=False,
    )

    if not metricas_elevados.empty:

        metricas_elevados.to_csv(
            PASTA_SAIDA
            / "metricas_cotas_elevadas_Dmais1.csv",
            index=False,
        )

    # --------------------------------------------------------
    # 12. GRAFICOS
    # --------------------------------------------------------

    grafico_serie(
        previsoes
    )

    grafico_observado_previsto(
        previsoes
    )

    grafico_residuos(
        previsoes
    )

    # --------------------------------------------------------
    # FINALIZACAO
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print(
        "MODELO BASELINE CONCLUIDO"
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
        "IMPORTANTE:"
    )

    print(
        "Este e um modelo experimental "
        "de previsao de cota D+1."
    )

    print(
        "Os resultados ainda nao constituem "
        "um sistema operacional de alerta "
        "de enchentes."
    )


if __name__ == "__main__":
    main()