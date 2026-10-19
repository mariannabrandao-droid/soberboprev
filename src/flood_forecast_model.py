from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


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
    / "modelo_cheias_Dmais1"
)

PASTA_SAIDA.mkdir(
    parents=True,
    exist_ok=True,
)

FRACAO_TREINO = 0.80

RANDOM_STATE = 42


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
# PREPARACAO
# ============================================================

def preparar_base(df):

    base = df.copy()

    # --------------------------------------------------------
    # Chuva do dia seguinte.
    #
    # No treinamento historico, usamos a chuva realmente
    # observada em D+1.
    #
    # Isso representa um experimento com informacao
    # meteorologica perfeita.
    #
    # No uso operacional, esta variavel sera substituida
    # pela precipitacao prevista para D+1.
    # --------------------------------------------------------

    base["chuva_Dmais1"] = (
        base["precipitacao_mm"]
        .shift(-1)
    )

    # --------------------------------------------------------
    # Variacao futura da cota
    # --------------------------------------------------------

    base["delta_cota_Dmais1"] = (
        base["cota_Dmais1"]
        - base["cota"]
    )

    return base


# ============================================================
# VARIAVEIS DOS MODELOS
# ============================================================

VARIAVEIS_SEM_CHUVA_FUTURA = [
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

VARIAVEIS_COM_CHUVA_FUTURA = [
    "cota",
    "precipitacao_mm",
    "chuva_lag_1d",
    "chuva_lag_2d",
    "chuva_lag_3d",
    "chuva_acum_2d",
    "chuva_acum_3d",
    "chuva_acum_5d",
    "chuva_acum_7d",
    "chuva_Dmais1",
]

ALVO = "cota_Dmais1"


# ============================================================
# AMOSTRAS COMPARAVEIS
# ============================================================

def criar_amostras(base):

    # Todos os modelos serao avaliados exatamente
    # nas mesmas datas.
    colunas_necessarias = list(
        dict.fromkeys(
            VARIAVEIS_COM_CHUVA_FUTURA
            + [
                ALVO,
                "delta_cota_Dmais1",
            ]
        )
    )

    dados = base.dropna(
        subset=colunas_necessarias
    ).copy()

    return dados


# ============================================================
# DIVISAO CRONOLOGICA
# ============================================================

def dividir_treino_teste(dados):

    corte = int(
        len(dados)
        * FRACAO_TREINO
    )

    treino = (
        dados
        .iloc[:corte]
        .copy()
    )

    teste = (
        dados
        .iloc[corte:]
        .copy()
    )

    return treino, teste


# ============================================================
# METRICAS
# ============================================================

def calcular_metricas(
    observado,
    previsto,
):

    observado = np.asarray(
        observado,
        dtype=float,
    )

    previsto = np.asarray(
        previsto,
        dtype=float,
    )

    mae = mean_absolute_error(
        observado,
        previsto,
    )

    rmse = np.sqrt(
        mean_squared_error(
            observado,
            previsto,
        )
    )

    r2 = r2_score(
        observado,
        previsto,
    )

    denominador_nse = np.sum(
        (
            observado
            - np.mean(observado)
        ) ** 2
    )

    if denominador_nse == 0:
        nse = np.nan
    else:
        nse = (
            1
            - (
                np.sum(
                    (
                        observado
                        - previsto
                    ) ** 2
                )
                / denominador_nse
            )
        )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "NSE": nse,
    }


# ============================================================
# METRICAS DE DETECCAO DE EVENTO
# ============================================================

def calcular_deteccao(
    observado,
    previsto,
    limiar,
):

    observado_evento = (
        np.asarray(observado)
        >= limiar
    )

    previsto_evento = (
        np.asarray(previsto)
        >= limiar
    )

    tp = int(
        np.sum(
            observado_evento
            & previsto_evento
        )
    )

    fn = int(
        np.sum(
            observado_evento
            & ~previsto_evento
        )
    )

    fp = int(
        np.sum(
            ~observado_evento
            & previsto_evento
        )
    )

    tn = int(
        np.sum(
            ~observado_evento
            & ~previsto_evento
        )
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else np.nan
    )

    precisao = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else np.nan
    )

    if (
        np.isnan(recall)
        or np.isnan(precisao)
        or (recall + precisao) == 0
    ):
        f1 = np.nan
    else:
        f1 = (
            2
            * recall
            * precisao
            / (recall + precisao)
        )

    taxa_falso_alarme = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else np.nan
    )

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
# TREINAMENTO
# ============================================================

def treinar_modelos(
    treino,
    teste,
):

    y_treino = treino[ALVO]

    y_teste = teste[ALVO]

    # --------------------------------------------------------
    # 1. PERSISTENCIA
    # --------------------------------------------------------

    previsao_persistencia = (
        teste["cota"]
        .to_numpy()
    )

    # --------------------------------------------------------
    # 2. REGRESSAO SEM CHUVA FUTURA
    # --------------------------------------------------------

    regressao_sem = LinearRegression()

    regressao_sem.fit(
        treino[
            VARIAVEIS_SEM_CHUVA_FUTURA
        ],
        y_treino,
    )

    previsao_reg_sem = (
        regressao_sem.predict(
            teste[
                VARIAVEIS_SEM_CHUVA_FUTURA
            ]
        )
    )

    # --------------------------------------------------------
    # 3. REGRESSAO COM CHUVA FUTURA
    # --------------------------------------------------------

    regressao_com = LinearRegression()

    regressao_com.fit(
        treino[
            VARIAVEIS_COM_CHUVA_FUTURA
        ],
        y_treino,
    )

    previsao_reg_com = (
        regressao_com.predict(
            teste[
                VARIAVEIS_COM_CHUVA_FUTURA
            ]
        )
    )

    # --------------------------------------------------------
    # 4. RANDOM FOREST COM CHUVA FUTURA
    # --------------------------------------------------------

    random_forest = RandomForestRegressor(
        n_estimators=500,
        max_depth=10,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    random_forest.fit(
        treino[
            VARIAVEIS_COM_CHUVA_FUTURA
        ],
        y_treino,
    )

    previsao_rf = (
        random_forest.predict(
            teste[
                VARIAVEIS_COM_CHUVA_FUTURA
            ]
        )
    )

    previsoes = {
        "Persistencia":
            previsao_persistencia,

        "Regressao sem chuva futura":
            previsao_reg_sem,

        "Regressao com chuva futura":
            previsao_reg_com,

        "Random Forest com chuva futura":
            previsao_rf,
    }

    modelos = {
        "regressao_sem":
            regressao_sem,

        "regressao_com":
            regressao_com,

        "random_forest":
            random_forest,
    }

    return (
        modelos,
        previsoes,
        y_teste,
    )


# ============================================================
# IMPORTANCIA RANDOM FOREST
# ============================================================

def importancia_variaveis(
    random_forest,
):

    importancia = pd.DataFrame(
        {
            "variavel":
                VARIAVEIS_COM_CHUVA_FUTURA,

            "importancia":
                random_forest
                .feature_importances_,
        }
    )

    importancia = (
        importancia
        .sort_values(
            "importancia",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return importancia


# ============================================================
# GRAFICO DE SERIE
# ============================================================

def grafico_serie(
    teste,
):

    plt.figure(
        figsize=(16, 7)
    )

    plt.plot(
        teste["data"],
        teste["observado"],
        label="Observado",
        linewidth=1.5,
    )

    plt.plot(
        teste["data"],
        teste["persistencia"],
        label="Persistencia",
        linewidth=1.0,
        alpha=0.8,
    )

    plt.plot(
        teste["data"],
        teste["regressao_sem_chuva_futura"],
        label="Regressao sem chuva futura",
        linewidth=1.0,
    )

    plt.plot(
        teste["data"],
        teste["regressao_com_chuva_futura"],
        label="Regressao com chuva futura",
        linewidth=1.2,
    )

    plt.plot(
        teste["data"],
        teste["random_forest_com_chuva_futura"],
        label="Random Forest com chuva futura",
        linewidth=1.2,
    )

    plt.xlabel("Data")
    plt.ylabel("Cota")

    plt.title(
        "Previsao de cota D+1 - comparacao com chuva futura"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        PASTA_SAIDA
        / "01_previsao_cota_Dmais1.png",
        dpi=180,
    )

    plt.close()


# ============================================================
# GRAFICO OBSERVADO X PREVISTO
# ============================================================

def grafico_observado_previsto(
    teste,
):

    observado = teste["observado"]

    previsto = (
        teste[
            "random_forest_com_chuva_futura"
        ]
    )

    minimo = min(
        observado.min(),
        previsto.min(),
    )

    maximo = max(
        observado.max(),
        previsto.max(),
    )

    plt.figure(
        figsize=(9, 9)
    )

    plt.scatter(
        observado,
        previsto,
        alpha=0.6,
    )

    plt.plot(
        [minimo, maximo],
        [minimo, maximo],
        linestyle="--",
        label="Previsao perfeita",
    )

    plt.xlabel(
        "Cota observada em D+1"
    )

    plt.ylabel(
        "Cota prevista em D+1"
    )

    plt.title(
        "Observado x previsto - Random Forest com chuva futura"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        PASTA_SAIDA
        / "02_observado_previsto_RF.png",
        dpi=180,
    )

    plt.close()


# ============================================================
# GRAFICO IMPORTANCIA
# ============================================================

def grafico_importancia(
    importancia,
):

    dados = (
        importancia
        .sort_values(
            "importancia",
            ascending=True,
        )
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        dados["variavel"],
        dados["importancia"],
    )

    plt.xlabel(
        "Importancia"
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
        / "03_importancia_variaveis.png",
        dpi=180,
    )

    plt.close()


# ============================================================
# EXECUCAO
# ============================================================

def main():

    warnings.filterwarnings(
        "ignore"
    )

    print()
    print("=" * 78)
    print(
        "SOBERBOPREV - MODELO ORIENTADO A CHEIAS"
    )
    print(
        "PREVISAO DE COTA D+1 COM PRECIPITACAO FUTURA"
    )
    print("=" * 78)

    df = carregar_base()

    base = preparar_base(df)

    dados = criar_amostras(base)

    print()
    print("PREPARACAO DAS AMOSTRAS")
    print("-" * 78)

    print(
        f"Dias no calendario: {len(base)}"
    )

    print(
        f"Amostras completas: {len(dados)}"
    )

    print(
        f"Linhas nao utilizadas: "
        f"{len(base) - len(dados)}"
    )

    print()
    print(
        "Nenhuma interpolacao foi realizada."
    )

    treino, teste = (
        dividir_treino_teste(
            dados
        )
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
        f"{treino['data'].min().date()} "
        f"a {treino['data'].max().date()}"
    )

    print()
    print(
        f"Teste: {len(teste)} amostras"
    )

    print(
        f"{teste['data'].min().date()} "
        f"a {teste['data'].max().date()}"
    )

    (
        modelos,
        previsoes,
        y_teste,
    ) = treinar_modelos(
        treino,
        teste,
    )

    # --------------------------------------------------------
    # RESULTADO DETALHADO
    # --------------------------------------------------------

    resultado = pd.DataFrame(
        {
            "data":
                teste["data"].values,

            "cota_atual":
                teste["cota"].values,

            "chuva_Dmais1":
                teste[
                    "chuva_Dmais1"
                ].values,

            "observado":
                y_teste.values,

            "persistencia":
                previsoes[
                    "Persistencia"
                ],

            "regressao_sem_chuva_futura":
                previsoes[
                    "Regressao sem chuva futura"
                ],

            "regressao_com_chuva_futura":
                previsoes[
                    "Regressao com chuva futura"
                ],

            "random_forest_com_chuva_futura":
                previsoes[
                    "Random Forest com chuva futura"
                ],
        }
    )

    # --------------------------------------------------------
    # METRICAS GERAIS
    # --------------------------------------------------------

    linhas_metricas = []

    for nome, previsao in previsoes.items():

        metrica = calcular_metricas(
            y_teste,
            previsao,
        )

        linhas_metricas.append(
            {
                "modelo": nome,
                **metrica,
            }
        )

    metricas = pd.DataFrame(
        linhas_metricas
    )

    print()
    print("=" * 78)
    print("METRICAS GERAIS - TESTE")
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
    # LIMIAR DE COTA ELEVADA
    # Somente treino
    # --------------------------------------------------------

    p90_cota_treino = (
        treino[ALVO]
        .quantile(0.90)
    )

    print()
    print("=" * 78)
    print("REFERENCIA ESTATISTICA DE COTA ELEVADA")
    print("=" * 78)
    print()

    print(
        f"P90 calculado somente no treino: "
        f"{p90_cota_treino:.2f}"
    )

    print()
    print(
        "Este valor e experimental e nao representa "
        "cota oficial de alerta ou inundacao."
    )

    # --------------------------------------------------------
    # DETECCAO DE COTA ELEVADA
    # --------------------------------------------------------

    linhas_deteccao = []

    for nome, previsao in previsoes.items():

        deteccao = calcular_deteccao(
            y_teste,
            previsao,
            p90_cota_treino,
        )

        linhas_deteccao.append(
            {
                "modelo": nome,
                **deteccao,
            }
        )

    deteccao = pd.DataFrame(
        linhas_deteccao
    )

    print()
    print("=" * 78)
    print("DETECCAO DE COTAS ELEVADAS")
    print("=" * 78)
    print()

    print(
        deteccao.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    eventos_reais = (
        y_teste
        >= p90_cota_treino
    )

    print()
    print(
        f"Eventos reais no teste: "
        f"{eventos_reais.sum()}"
    )

    # --------------------------------------------------------
    # ERRO APENAS NAS COTAS ELEVADAS
    # --------------------------------------------------------

    metricas_eventos = []

    if eventos_reais.sum() > 1:

        observado_eventos = (
            y_teste[
                eventos_reais
            ]
        )

        for nome, previsao in previsoes.items():

            previsao_eventos = (
                np.asarray(previsao)[
                    eventos_reais.values
                ]
            )

            metrica = calcular_metricas(
                observado_eventos,
                previsao_eventos,
            )

            metricas_eventos.append(
                {
                    "modelo": nome,
                    **metrica,
                }
            )

    metricas_eventos = pd.DataFrame(
        metricas_eventos
    )

    print()
    print("=" * 78)
    print("ERRO NOS EVENTOS REAIS DE COTA ELEVADA")
    print("=" * 78)
    print()

    if not metricas_eventos.empty:

        print(
            metricas_eventos.to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.4f}",
            )
        )

    # --------------------------------------------------------
    # SUBIDAS RAPIDAS
    # --------------------------------------------------------

    p90_delta_treino = (
        treino[
            "delta_cota_Dmais1"
        ]
        .quantile(0.90)
    )

    resultado["delta_observado"] = (
        resultado["observado"]
        - resultado["cota_atual"]
    )

    resultado["delta_rf"] = (
        resultado[
            "random_forest_com_chuva_futura"
        ]
        - resultado["cota_atual"]
    )

    subida_real = (
        resultado["delta_observado"]
        >= p90_delta_treino
    )

    subida_prevista_rf = (
        resultado["delta_rf"]
        >= p90_delta_treino
    )

    tp_delta = int(
        (
            subida_real
            & subida_prevista_rf
        ).sum()
    )

    fn_delta = int(
        (
            subida_real
            & ~subida_prevista_rf
        ).sum()
    )

    fp_delta = int(
        (
            ~subida_real
            & subida_prevista_rf
        ).sum()
    )

    tn_delta = int(
        (
            ~subida_real
            & ~subida_prevista_rf
        ).sum()
    )

    recall_delta = (
        tp_delta
        / (tp_delta + fn_delta)
        if (tp_delta + fn_delta) > 0
        else np.nan
    )

    precisao_delta = (
        tp_delta
        / (tp_delta + fp_delta)
        if (tp_delta + fp_delta) > 0
        else np.nan
    )

    print()
    print("=" * 78)
    print("DETECCAO DE SUBIDAS RAPIDAS - RANDOM FOREST")
    print("=" * 78)
    print()

    print(
        f"P90 da variacao calculado no treino: "
        f"{p90_delta_treino:.2f}"
    )

    print(
        f"Subidas rapidas reais no teste: "
        f"{subida_real.sum()}"
    )

    print()
    print(
        f"TP: {tp_delta}"
    )

    print(
        f"FN: {fn_delta}"
    )

    print(
        f"FP: {fp_delta}"
    )

    print(
        f"TN: {tn_delta}"
    )

    print(
        f"Recall: {recall_delta:.4f}"
    )

    print(
        f"Precisao: {precisao_delta:.4f}"
    )

    # --------------------------------------------------------
    # IMPORTANCIA
    # --------------------------------------------------------

    importancia = (
        importancia_variaveis(
            modelos["random_forest"]
        )
    )

    print()
    print("=" * 78)
    print("IMPORTANCIA DAS VARIAVEIS - RANDOM FOREST")
    print("=" * 78)
    print()

    print(
        importancia.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    # --------------------------------------------------------
    # MAIORES EVENTOS
    # --------------------------------------------------------

    maiores_eventos = (
        resultado
        .sort_values(
            "observado",
            ascending=False,
        )
        .head(25)
        .copy()
    )

    print()
    print("=" * 78)
    print("25 MAIORES COTAS OBSERVADAS NO TESTE")
    print("=" * 78)
    print()

    print(
        maiores_eventos[
            [
                "data",
                "cota_atual",
                "chuva_Dmais1",
                "observado",
                "persistencia",
                "regressao_sem_chuva_futura",
                "regressao_com_chuva_futura",
                "random_forest_com_chuva_futura",
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.2f}",
        )
    )

    # --------------------------------------------------------
    # SALVAR
    # --------------------------------------------------------

    resultado.to_csv(
        PASTA_SAIDA
        / "previsoes_Dmais1.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metricas.to_csv(
        PASTA_SAIDA
        / "metricas_gerais.csv",
        index=False,
        encoding="utf-8-sig",
    )

    deteccao.to_csv(
        PASTA_SAIDA
        / "deteccao_cotas_elevadas.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metricas_eventos.to_csv(
        PASTA_SAIDA
        / "metricas_cotas_elevadas.csv",
        index=False,
        encoding="utf-8-sig",
    )

    importancia.to_csv(
        PASTA_SAIDA
        / "importancia_variaveis_RF.csv",
        index=False,
        encoding="utf-8-sig",
    )

    maiores_eventos.to_csv(
        PASTA_SAIDA
        / "25_maiores_cotas_teste.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # GRAFICOS
    # --------------------------------------------------------

    grafico_serie(
        resultado
    )

    grafico_observado_previsto(
        resultado
    )

    grafico_importancia(
        importancia
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("MODELO DE CHEIAS D+1 CONCLUIDO")
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
        "A chuva observada em D+1 foi utilizada apenas "
        "como experimento de informacao meteorologica perfeita."
    )

    print(
        "Na aplicacao operacional, ela devera ser substituida "
        "pela precipitacao prevista."
    )

    print()
    print(
        "Os limiares estatisticos utilizados nao representam "
        "cotas oficiais de alerta ou inundacao."
    )


if __name__ == "__main__":
    main()