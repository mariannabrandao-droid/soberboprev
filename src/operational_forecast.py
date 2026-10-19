from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression

from config import PONTE
from src.weather import (
    carregar_previsao_open_meteo,
    preparar_previsao_diaria_interface,
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
    / "previsao_operacional"
)

PASTA_SAIDA.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# VARIAVEIS DO MODELO
# ============================================================

VARIAVEIS = [
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
# CARREGAR BASE HISTORICA
# ============================================================

def carregar_base_historica():

    if not ARQUIVO_BASE.exists():
        raise FileNotFoundError(
            f"Base historica nao encontrada:\n{ARQUIVO_BASE}"
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
# PREPARAR TREINAMENTO
# ============================================================

def preparar_treinamento(df):

    base = df.copy()

    # No treinamento historico:
    # chuva_Dmais1 = chuva realmente observada em D+1.
    #
    # Na operacao:
    # chuva_Dmais1 = chuva prevista pelo Open-Meteo.

    base["chuva_Dmais1"] = (
        base["precipitacao_mm"]
        .shift(-1)
    )

    treinamento = (
        base
        .dropna(
            subset=VARIAVEIS + [ALVO]
        )
        .copy()
    )

    return treinamento


# ============================================================
# TREINAR MODELO
# ============================================================

def treinar_modelo():

    df = carregar_base_historica()

    treinamento = preparar_treinamento(df)

    modelo = LinearRegression()

    modelo.fit(
        treinamento[VARIAVEIS],
        treinamento[ALVO],
    )

    return modelo, treinamento


# ============================================================
# REFERENCIAS ESTATISTICAS
# ============================================================

def calcular_referencias_estatisticas(
    treinamento,
):

    return {
        "p90": float(
            treinamento["cota_Dmais1"]
            .quantile(0.90)
        ),
        "p95": float(
            treinamento["cota_Dmais1"]
            .quantile(0.95)
        ),
    }


# ============================================================
# CLASSIFICACAO EXPERIMENTAL
# ============================================================

def classificar_condicao(
    cota_prevista,
    referencias,
):

    p90 = referencias["p90"]
    p95 = referencias["p95"]

    if cota_prevista >= p95:
        return (
            "Cota muito elevada "
            "- referencia estatistica experimental"
        )

    if cota_prevista >= p90:
        return (
            "Cota elevada "
            "- referencia estatistica experimental"
        )

    return (
        "Abaixo da referencia estatistica "
        "de cota elevada"
    )


# ============================================================
# PREVISAO METEOROLOGICA
# ============================================================

def obter_chuva_prevista_Dmais1():
    """
    Consulta a Open-Meteo para o ponto da Ponte sobre
    o Rio Soberbo e retorna a precipitacao acumulada
    prevista para D+1.

    Retorna tambem a tabela diaria para auditoria.
    """

    previsao_horaria = (
        carregar_previsao_open_meteo(
            latitude=PONTE["lat"],
            longitude=PONTE["lon"],
            forecast_days=8,
        )
    )

    previsao_diaria = (
        preparar_previsao_diaria_interface(
            previsao_horaria
        )
    )

    if previsao_diaria.empty:
        raise ValueError(
            "A Open-Meteo nao retornou "
            "previsao diaria."
        )

    linha_d1 = previsao_diaria[
        previsao_diaria["horizonte"]
        == "D+1"
    ]

    if linha_d1.empty:
        raise ValueError(
            "Nao foi encontrada previsao "
            "meteorologica para D+1."
        )

    chuva_d1 = float(
        linha_d1.iloc[0][
            "precipitacao_mm"
        ]
    )

    return chuva_d1, previsao_diaria


# ============================================================
# PREPARAR INPUT OPERACIONAL
# ============================================================

def preparar_entrada_operacional(
    cota_atual,
    chuva_hoje,
    chuva_d1,
    chuva_d2,
    chuva_d3,
    chuva_d4,
    chuva_d5,
    chuva_d6,
    chuva_prevista_amanha,
):

    chuva_acum_2d = (
        chuva_hoje
        + chuva_d1
    )

    chuva_acum_3d = (
        chuva_hoje
        + chuva_d1
        + chuva_d2
    )

    chuva_acum_5d = (
        chuva_hoje
        + chuva_d1
        + chuva_d2
        + chuva_d3
        + chuva_d4
    )

    chuva_acum_7d = (
        chuva_hoje
        + chuva_d1
        + chuva_d2
        + chuva_d3
        + chuva_d4
        + chuva_d5
        + chuva_d6
    )

    entrada = pd.DataFrame(
        [
            {
                "cota":
                    float(cota_atual),

                "precipitacao_mm":
                    float(chuva_hoje),

                "chuva_lag_1d":
                    float(chuva_d1),

                "chuva_lag_2d":
                    float(chuva_d2),

                "chuva_lag_3d":
                    float(chuva_d3),

                "chuva_acum_2d":
                    float(chuva_acum_2d),

                "chuva_acum_3d":
                    float(chuva_acum_3d),

                "chuva_acum_5d":
                    float(chuva_acum_5d),

                "chuva_acum_7d":
                    float(chuva_acum_7d),

                "chuva_Dmais1":
                    float(
                        chuva_prevista_amanha
                    ),
            }
        ]
    )

    return entrada


# ============================================================
# PREVISAO OPERACIONAL
# ============================================================

def prever_cota_Dmais1(
    cota_atual,
    chuva_hoje,
    chuva_d1,
    chuva_d2,
    chuva_d3,
    chuva_d4,
    chuva_d5,
    chuva_d6,
    chuva_prevista_amanha=None,
):
    """
    Preve a cota do Rio Soberbo para D+1.

    Se chuva_prevista_amanha for None,
    consulta automaticamente a Open-Meteo.

    Isso permite:
    - uso automatico no Streamlit;
    - teste manual quando necessario.
    """

    modelo, treinamento = (
        treinar_modelo()
    )

    referencias = (
        calcular_referencias_estatisticas(
            treinamento
        )
    )

    previsao_diaria = None
    origem_chuva_futura = "manual"

    if chuva_prevista_amanha is None:

        (
            chuva_prevista_amanha,
            previsao_diaria,
        ) = obter_chuva_prevista_Dmais1()

        origem_chuva_futura = (
            "Open-Meteo"
        )

    entrada = preparar_entrada_operacional(
        cota_atual=cota_atual,
        chuva_hoje=chuva_hoje,
        chuva_d1=chuva_d1,
        chuva_d2=chuva_d2,
        chuva_d3=chuva_d3,
        chuva_d4=chuva_d4,
        chuva_d5=chuva_d5,
        chuva_d6=chuva_d6,
        chuva_prevista_amanha=
            chuva_prevista_amanha,
    )

    previsao = float(
        modelo.predict(
            entrada[VARIAVEIS]
        )[0]
    )

    # Evita resultado negativo
    previsao = max(
        previsao,
        0.0,
    )

    variacao = (
        previsao
        - float(cota_atual)
    )

    condicao = (
        classificar_condicao(
            previsao,
            referencias,
        )
    )

    resultado = {
        "cota_atual":
            float(cota_atual),

        "chuva_prevista_Dmais1":
            float(
                chuva_prevista_amanha
            ),

        "origem_chuva_futura":
            origem_chuva_futura,

        "cota_prevista_Dmais1":
            previsao,

        "variacao_prevista":
            variacao,

        "referencia_P90":
            referencias["p90"],

        "referencia_P95":
            referencias["p95"],

        "condicao":
            condicao,

        "previsao_meteorologica_diaria":
            previsao_diaria,
    }

    return resultado


# ============================================================
# TESTE OPERACIONAL
# ============================================================

def main():

    print()
    print("=" * 78)
    print(
        "SOBERBOPREV - PREVISAO OPERACIONAL D+1"
    )
    print("=" * 78)

    print()
    print(
        "A precipitacao futura sera consultada "
        "automaticamente na Open-Meteo."
    )

    print()
    print(
        "A cota e as chuvas antecedentes ainda "
        "sao entradas demonstrativas."
    )

    print(
        "Elas NAO representam a condicao atual "
        "do Rio Soberbo."
    )

    # --------------------------------------------------------
    # Somente para teste da integracao.
    #
    # A cota atual sera futuramente substituida
    # pelo sensor.
    #
    # As chuvas antecedentes tambem deverao ser
    # alimentadas por observacoes reais.
    # --------------------------------------------------------

    resultado = prever_cota_Dmais1(
        cota_atual=60.0,
        chuva_hoje=5.0,
        chuva_d1=2.0,
        chuva_d2=0.0,
        chuva_d3=4.0,
        chuva_d4=0.0,
        chuva_d5=1.0,
        chuva_d6=0.0,

        # IMPORTANTE:
        # nao fornecemos chuva_prevista_amanha.
        # O modulo buscara automaticamente
        # na Open-Meteo.
    )

    print()
    print("=" * 78)
    print(
        "PREVISAO METEOROLOGICA UTILIZADA"
    )
    print("=" * 78)
    print()

    previsao_diaria = resultado[
        "previsao_meteorologica_diaria"
    ]

    if previsao_diaria is not None:

        print(
            previsao_diaria.to_string(
                index=False
            )
        )

    print()
    print("=" * 78)
    print(
        "RESULTADO D+1"
    )
    print("=" * 78)
    print()

    print(
        f"Cota atual informada: "
        f"{resultado['cota_atual']:.2f}"
    )

    print(
        f"Chuva prevista D+1: "
        f"{resultado['chuva_prevista_Dmais1']:.2f} mm"
    )

    print(
        f"Origem da chuva futura: "
        f"{resultado['origem_chuva_futura']}"
    )

    print()

    print(
        f"Cota prevista D+1: "
        f"{resultado['cota_prevista_Dmais1']:.2f}"
    )

    print(
        f"Variacao prevista: "
        f"{resultado['variacao_prevista']:+.2f}"
    )

    print()
    print(
        f"P90 historico experimental: "
        f"{resultado['referencia_P90']:.2f}"
    )

    print(
        f"P95 historico experimental: "
        f"{resultado['referencia_P95']:.2f}"
    )

    print()
    print(
        "Condicao:"
    )

    print(
        resultado["condicao"]
    )

    print()
    print("=" * 78)
    print("STATUS DA ARQUITETURA")
    print("=" * 78)
    print()

    print(
        "[OK] Base historica ANA"
    )

    print(
        "[OK] Modelo D+1"
    )

    print(
        "[OK] Previsao meteorologica Open-Meteo"
    )

    print(
        "[OK] Chuva futura conectada ao modelo"
    )

    print(
        "[TEMPORARIO] Cota atual manual"
    )

    print(
        "[TEMPORARIO] Chuva antecedente manual"
    )

    print(
        "[FUTURO] Sensor de nivel UFF"
    )

    print()
    print(
        "P90 e P95 continuam sendo apenas "
        "referencias estatisticas experimentais."
    )


if __name__ == "__main__":
    main()