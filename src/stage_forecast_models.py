from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor


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
    / "comparacao_modelos_Dmais1"
)

PASTA_SAIDA.mkdir(
    parents=True,
    exist_ok=True,
)

PROPORCAO_TREINO = 0.80

RANDOM_STATE = 42


# ============================================================
# VARIAVEIS
# ============================================================

FEATURES = [
    "cota",
    "precipitacao_mm",
    "chuva_lag_1d",
    "chuva_lag_2d",
    "chuva_lag_3d",
    "chuva_acum_2d",
    "chuva_acum_3d",
    "chuva_acum_5d",
    "chuva_acum_7d",
]

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

        X_design = np.column_stack(
            [
                np.ones(len(X)),
                X,
            ]
        )

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
# METRICAS DE REGRESSAO
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
# METRICAS PARA EVENTOS ELEVADOS
# ============================================================

def metricas_eventos(
    observado,
    previsto,
    limiar,
):

    observado = np.asarray(
        observado,
        dtype=float,
    )

    previsto = np.asarray(
        previsto,
        dtype=float,
    )

    evento_real = (
        observado >= limiar
    )

    evento_previsto = (
        previsto >= limiar
    )

    tp = int(
        np.sum(
            evento_real
            & evento_previsto
        )
    )

    fn = int(
        np.sum(
            evento_real
            & ~evento_previsto
        )
    )

    fp = int(
        np.sum(
            ~evento_real
            & evento_previsto
        )
    )

    tn = int(
        np.sum(
            ~evento_real
            & ~evento_previsto
        )
    )

    if (tp + fn) > 0:

        recall = (
            tp / (tp + fn)
        )

    else:

        recall = np.nan

    if (tp + fp) > 0:

        precisao = (
            tp / (tp + fp)
        )

    else:

        precisao = np.nan

    if (
        not np.isnan(recall)
        and not np.isnan(precisao)
        and (recall + precisao) > 0
    ):

        f1 = (
            2
            * precisao
            * recall
            / (
                precisao
                + recall
            )
        )

    else:

        f1 = np.nan

    if (fp + tn) > 0:

        taxa_falso_alarme = (
            fp / (fp + tn)
        )

    else:

        taxa_falso_alarme = np.nan

    return {
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn,
        "Recall_evento": recall,
        "Precisao_evento": precisao,
        "F1_evento": f1,
        "Taxa_falso_alarme": taxa_falso_alarme,
    }


# ============================================================
# BASE
# ============================================================

def carregar_base():

    if not ARQUIVO_BASE.exists():

        raise FileNotFoundError(
            "Base de modelagem nao encontrada:\n"
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
            "Colunas ausentes: "
            + ", ".join(faltantes)
        )

    df = (
        df
        .sort_values("data")
        .reset_index(drop=True)
    )

    return df


def preparar_amostras(df):

    colunas = (
        ["data"]
        + FEATURES
        + [TARGET]
    )

    base = df[
        colunas
    ].copy()

    inicial = len(base)

    base = base.dropna(
        subset=FEATURES + [TARGET]
    )

    base = (
        base
        .sort_values("data")
        .reset_index(drop=True)
    )

    final = len(base)

    return (
        base,
        inicial,
        final,
        inicial - final,
    )


# ============================================================
# DIVISAO CRONOLOGICA
# ============================================================

def dividir_treino_teste(base):

    indice_corte = int(
        np.floor(
            len(base)
            * PROPORCAO_TREINO
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
# GRAFICO COMPARATIVO
# ============================================================

def grafico_serie(previsoes):

    plt.figure(
        figsize=(16, 7)
    )

    plt.plot(
        previsoes["data_alvo"],
        previsoes["observado"],
        linewidth=1.6,
        label="Observado",
    )

    plt.plot(
        previsoes["data_alvo"],
        previsoes["persistencia"],
        linewidth=1.0,
        alpha=0.75,
        label="Persistencia",
    )

    plt.plot(
        previsoes["data_alvo"],
        previsoes["regressao_linear"],
        linewidth=1.0,
        alpha=0.75,
        label="Regressao linear",
    )

    plt.plot(
        previsoes["data_alvo"],
        previsoes["random_forest"],
        linewidth=1.2,
        alpha=0.85,
        label="Random Forest",
    )

    plt.xlabel("Data")

    plt.ylabel("Cota")

    plt.title(
        "Comparacao de modelos - previsao D+1"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        PASTA_SAIDA
        / "01_comparacao_modelos_Dmais1.png",
        dpi=180,
    )

    plt.close()


def grafico_observado_rf(
    previsoes,
):

    observado = previsoes[
        "observado"
    ]

    previsto = previsoes[
        "random_forest"
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
        "Observado x previsto - Random Forest"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        PASTA_SAIDA
        / "02_observado_previsto_random_forest.png",
        dpi=180,
    )

    plt.close()


def grafico_eventos(
    previsoes,
    limiar,
):

    eventos = previsoes[
        previsoes["observado"]
        >= limiar
    ].copy()

    if eventos.empty:
        return

    x = np.arange(
        len(eventos)
    )

    largura = 0.25

    plt.figure(
        figsize=(15, 7)
    )

    plt.bar(
        x - largura,
        eventos["observado"],
        width=largura,
        label="Observado",
    )

    plt.bar(
        x,
        eventos["regressao_linear"],
        width=largura,
        label="Regressao linear",
    )

    plt.bar(
        x + largura,
        eventos["random_forest"],
        width=largura,
        label="Random Forest",
    )

    plt.axhline(
        limiar,
        linestyle="--",
        linewidth=1.2,
        label="P90 do treino",
    )

    plt.xticks(
        x,
        eventos[
            "data_alvo"
        ].dt.strftime("%d/%m/%Y"),
        rotation=90,
    )

    plt.ylabel("Cota")

    plt.xlabel("Eventos observados")

    plt.title(
        "Desempenho nos eventos de cota elevada"
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    plt.tight_layout()

    plt.savefig(
        PASTA_SAIDA
        / "03_eventos_cota_elevada.png",
        dpi=180,
    )

    plt.close()


def grafico_importancia(
    modelo_rf,
):

    importancias = pd.DataFrame(
        {
            "variavel": FEATURES,
            "importancia":
                modelo_rf.feature_importances_,
        }
    )

    importancias = (
        importancias
        .sort_values(
            "importancia",
            ascending=True,
        )
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        importancias["variavel"],
        importancias["importancia"],
    )

    plt.xlabel(
        "Importancia no Random Forest"
    )

    plt.ylabel(
        "Variavel"
    )

    plt.title(
        "Importancia das variaveis - Random Forest"
    )

    plt.grid(
        axis="x",
        alpha=0.25,
    )

    plt.tight_layout()

    plt.savefig(
        PASTA_SAIDA
        / "04_importancia_variaveis_random_forest.png",
        dpi=180,
    )

    plt.close()

    return (
        importancias
        .sort_values(
            "importancia",
            ascending=False,
        )
    )


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
        "SOBERBOPREV - COMPARACAO DE MODELOS"
    )
    print(
        "PREVISAO DE COTA D+1"
    )
    print("=" * 78)

    # --------------------------------------------------------
    # BASE
    # --------------------------------------------------------

    df = carregar_base()

    (
        base,
        inicial,
        final,
        removidas,
    ) = preparar_amostras(df)

    print()
    print("PREPARACAO DAS AMOSTRAS")
    print("-" * 78)

    print(
        f"Dias no calendario: {inicial}"
    )

    print(
        f"Amostras completas: {final}"
    )

    print(
        f"Linhas nao utilizadas: {removidas}"
    )

    print()
    print(
        "Nenhuma interpolacao foi realizada."
    )

    # --------------------------------------------------------
    # DIVISAO
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
        f"Treino: {len(treino)}"
    )

    print(
        f"{treino['data'].min().date()} "
        f"a "
        f"{treino['data'].max().date()}"
    )

    print()
    print(
        f"Teste: {len(teste)}"
    )

    print(
        f"{teste['data'].min().date()} "
        f"a "
        f"{teste['data'].max().date()}"
    )

    # --------------------------------------------------------
    # MATRIZES
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
    # PERSISTENCIA
    # --------------------------------------------------------

    pred_persistencia = (
        teste["cota"]
        .values
    )

    # --------------------------------------------------------
    # REGRESSAO LINEAR
    # --------------------------------------------------------

    modelo_linear = (
        RegressaoLinearOLS()
    )

    modelo_linear.fit(
        X_treino,
        y_treino,
    )

    pred_linear = (
        modelo_linear.predict(
            X_teste
        )
    )

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    modelo_rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    modelo_rf.fit(
        X_treino,
        y_treino,
    )

    pred_rf = (
        modelo_rf.predict(
            X_teste
        )
    )

    # --------------------------------------------------------
    # METRICAS GERAIS
    # --------------------------------------------------------

    resultados = []

    for nome, previsao in [
        (
            "Persistencia",
            pred_persistencia,
        ),
        (
            "Regressao Linear",
            pred_linear,
        ),
        (
            "Random Forest",
            pred_rf,
        ),
    ]:

        resultados.append(
            {
                "modelo": nome,
                **calcular_metricas(
                    y_teste,
                    previsao,
                ),
            }
        )

    metricas = pd.DataFrame(
        resultados
    )

    print()
    print("=" * 78)
    print(
        "METRICAS GERAIS - TESTE"
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
    # LIMIAR P90
    # --------------------------------------------------------

    limiar_p90 = (
        treino[TARGET]
        .quantile(0.90)
    )

    print()
    print("=" * 78)
    print(
        "LIMIAR ESTATISTICO PARA EVENTOS"
    )
    print("=" * 78)

    print()
    print(
        f"P90 calculado apenas no treino: "
        f"{limiar_p90:.2f}"
    )

    print()
    print(
        "Este limiar e experimental e "
        "NAO representa cota oficial "
        "de alerta ou inundacao."
    )

    # --------------------------------------------------------
    # METRICAS DOS EVENTOS
    # --------------------------------------------------------

    resultados_eventos = []

    for nome, previsao in [
        (
            "Persistencia",
            pred_persistencia,
        ),
        (
            "Regressao Linear",
            pred_linear,
        ),
        (
            "Random Forest",
            pred_rf,
        ),
    ]:

        resultado = {
            "modelo": nome,
        }

        resultado.update(
            metricas_eventos(
                y_teste,
                previsao,
                limiar_p90,
            )
        )

        resultados_eventos.append(
            resultado
        )

    metricas_eventos_df = (
        pd.DataFrame(
            resultados_eventos
        )
    )

    print()
    print("=" * 78)
    print(
        "DETECCAO DE COTAS ELEVADAS"
    )
    print("=" * 78)
    print()

    print(
        metricas_eventos_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    # --------------------------------------------------------
    # ERRO SOMENTE NOS EVENTOS REAIS
    # --------------------------------------------------------

    mascara_eventos = (
        y_teste >= limiar_p90
    )

    quantidade_eventos = int(
        np.sum(
            mascara_eventos
        )
    )

    print()
    print(
        f"Eventos reais no teste: "
        f"{quantidade_eventos}"
    )

    metricas_altas = []

    if quantidade_eventos > 0:

        for nome, previsao in [
            (
                "Persistencia",
                pred_persistencia,
            ),
            (
                "Regressao Linear",
                pred_linear,
            ),
            (
                "Random Forest",
                pred_rf,
            ),
        ]:

            metricas_altas.append(
                {
                    "modelo": nome,
                    **calcular_metricas(
                        y_teste[
                            mascara_eventos
                        ],
                        previsao[
                            mascara_eventos
                        ],
                    ),
                }
            )

    metricas_altas_df = (
        pd.DataFrame(
            metricas_altas
        )
    )

    print()
    print("=" * 78)
    print(
        "ERRO NOS EVENTOS REAIS DE COTA ELEVADA"
    )
    print("=" * 78)
    print()

    if not metricas_altas_df.empty:

        print(
            metricas_altas_df.to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.4f}",
            )
        )

    # --------------------------------------------------------
    # PREVISOES
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

            "observado":
                y_teste,

            "persistencia":
                pred_persistencia,

            "regressao_linear":
                pred_linear,

            "random_forest":
                pred_rf,
        }
    )

    previsoes["evento_real"] = (
        previsoes["observado"]
        >= limiar_p90
    )

    previsoes[
        "evento_rf"
    ] = (
        previsoes["random_forest"]
        >= limiar_p90
    )

    previsoes[
        "erro_rf"
    ] = (
        previsoes["observado"]
        - previsoes["random_forest"]
    )

    previsoes[
        "erro_absoluto_rf"
    ] = (
        previsoes["erro_rf"]
        .abs()
    )

    # --------------------------------------------------------
    # MAIORES ERROS RF
    # --------------------------------------------------------

    maiores_erros_rf = (
        previsoes
        .sort_values(
            "erro_absoluto_rf",
            ascending=False,
        )
        .head(20)
        .copy()
    )

    print()
    print("=" * 78)
    print(
        "20 MAIORES ERROS - RANDOM FOREST"
    )
    print("=" * 78)
    print()

    print(
        maiores_erros_rf[
            [
                "data_origem",
                "data_alvo",
                "cota_atual",
                "observado",
                "random_forest",
                "erro_rf",
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.2f}",
        )
    )

    # --------------------------------------------------------
    # IMPORTANCIA DAS VARIAVEIS
    # --------------------------------------------------------

    importancias = (
        grafico_importancia(
            modelo_rf
        )
    )

    print()
    print("=" * 78)
    print(
        "IMPORTANCIA DAS VARIAVEIS - RANDOM FOREST"
    )
    print("=" * 78)
    print()

    print(
        importancias.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    # --------------------------------------------------------
    # SALVAR
    # --------------------------------------------------------

    metricas.to_csv(
        PASTA_SAIDA
        / "metricas_gerais_Dmais1.csv",
        index=False,
    )

    metricas_eventos_df.to_csv(
        PASTA_SAIDA
        / "metricas_deteccao_eventos_Dmais1.csv",
        index=False,
    )

    metricas_altas_df.to_csv(
        PASTA_SAIDA
        / "metricas_eventos_elevados_Dmais1.csv",
        index=False,
    )

    previsoes.to_csv(
        PASTA_SAIDA
        / "previsoes_comparacao_Dmais1.csv",
        index=False,
    )

    maiores_erros_rf.to_csv(
        PASTA_SAIDA
        / "maiores_erros_random_forest_Dmais1.csv",
        index=False,
    )

    importancias.to_csv(
        PASTA_SAIDA
        / "importancia_variaveis_random_forest.csv",
        index=False,
    )

    # --------------------------------------------------------
    # GRAFICOS
    # --------------------------------------------------------

    grafico_serie(
        previsoes
    )

    grafico_observado_rf(
        previsoes
    )

    grafico_eventos(
        previsoes,
        limiar_p90,
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print(
        "COMPARACAO DE MODELOS CONCLUIDA"
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
        "O P90 utilizado e apenas uma "
        "referencia estatistica experimental."
    )

    print(
        "Nenhum resultado deve ser interpretado "
        "como limiar oficial de inundacao."
    )


if __name__ == "__main__":
    main()