from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.rating_curve_calibration import (
    preparar_dados,
    calibrar_curva,
    curva_chave,
    calcular_metricas,
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

PASTA_PROJETO = Path(__file__).resolve().parent.parent

PASTA_SAIDA = (
    PASTA_PROJETO
    / "data"
    / "calibracao"
    / "estabilidade"
)

PASTA_SAIDA.mkdir(
    parents=True,
    exist_ok=True,
)


PERIODOS = {
    "1976-1981": (
        pd.Timestamp("1976-01-01"),
        pd.Timestamp("1981-12-31"),
    ),
    "1983-1989": (
        pd.Timestamp("1983-01-01"),
        pd.Timestamp("1989-12-31"),
    ),
    "2003-2011": (
        pd.Timestamp("2003-01-01"),
        pd.Timestamp("2011-12-31"),
    ),
}


# Quantidade mínima apenas para permitir
# uma tentativa exploratória de ajuste.
MINIMO_MEDICOES_AJUSTE = 8


# ============================================================
# SEPARAÇÃO DOS PERÍODOS
# ============================================================

def separar_periodos(
    dados: pd.DataFrame,
):

    conjuntos = {}

    for nome, (
        inicio,
        fim,
    ) in PERIODOS.items():

        periodo = dados[
            (dados["Data"] >= inicio)
            & (dados["Data"] <= fim)
        ].copy()

        periodo = (
            periodo
            .sort_values("Data")
            .reset_index(drop=True)
        )

        conjuntos[nome] = periodo

    return conjuntos


# ============================================================
# DIAGNÓSTICO DOS PERÍODOS
# ============================================================

def diagnosticar_periodos(
    conjuntos: dict,
):

    linhas = []

    print("\n" + "=" * 72)
    print("DIAGNÓSTICO DOS PERÍODOS")
    print("=" * 72)

    for nome, df in conjuntos.items():

        print("\n" + "-" * 72)
        print(f"PERÍODO {nome}")
        print("-" * 72)

        n = len(df)

        print(
            f"Número de medições: {n}"
        )

        if df.empty:

            print(
                "Nenhuma medição disponível."
            )

            linhas.append(
                {
                    "Periodo": nome,
                    "N": 0,
                    "DataInicial": None,
                    "DataFinal": None,
                    "CotaMin": np.nan,
                    "CotaMax": np.nan,
                    "VazaoMin": np.nan,
                    "VazaoMax": np.nan,
                }
            )

            continue

        print(
            f"Primeira medição: "
            f"{df['Data'].min().date()}"
        )

        print(
            f"Última medição: "
            f"{df['Data'].max().date()}"
        )

        print(
            f"Cota: "
            f"{df['Cota'].min():.2f}"
            f" a "
            f"{df['Cota'].max():.2f}"
        )

        print(
            f"Vazão: "
            f"{df['Vazao'].min():.3f}"
            f" a "
            f"{df['Vazao'].max():.3f}"
            f" m³/s"
        )

        linhas.append(
            {
                "Periodo": nome,
                "N": n,
                "DataInicial": df["Data"].min(),
                "DataFinal": df["Data"].max(),
                "CotaMin": df["Cota"].min(),
                "CotaMax": df["Cota"].max(),
                "VazaoMin": df["Vazao"].min(),
                "VazaoMax": df["Vazao"].max(),
            }
        )

    diagnostico = pd.DataFrame(
        linhas
    )

    diagnostico.to_csv(
        PASTA_SAIDA
        / "01_diagnostico_periodos.csv",
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8-sig",
    )

    return diagnostico


# ============================================================
# CALIBRAÇÃO POR PERÍODO
# ============================================================

def calibrar_periodos(
    conjuntos: dict,
):

    parametros_periodos = {}

    print("\n" + "=" * 72)
    print("CALIBRAÇÃO INDEPENDENTE POR PERÍODO")
    print("=" * 72)

    for nome, df in conjuntos.items():

        print("\n" + "-" * 72)
        print(f"PERÍODO {nome}")
        print("-" * 72)

        if len(df) < MINIMO_MEDICOES_AJUSTE:

            print(
                "Dados insuficientes para ajuste "
                "independente confiável."
            )

            print(
                f"Foram encontradas apenas "
                f"{len(df)} medições."
            )

            parametros_periodos[nome] = None

            continue

        try:

            parametros = calibrar_curva(
                df
            )

            parametros_periodos[nome] = (
                parametros
            )

            print(
                f"a  = "
                f"{parametros['a']:.8f}"
            )

            print(
                f"H0 = "
                f"{parametros['h0']:.4f}"
            )

            print(
                f"b  = "
                f"{parametros['b']:.6f}"
            )

            previsto = curva_chave(
                df["Cota"].to_numpy(
                    dtype=float
                ),
                parametros["a"],
                parametros["h0"],
                parametros["b"],
            )

            metricas = calcular_metricas(
                df["Vazao"],
                previsto,
            )

            print(
                f"MAE  = "
                f"{metricas['MAE']:.3f} m³/s"
            )

            print(
                f"RMSE = "
                f"{metricas['RMSE']:.3f} m³/s"
            )

            print(
                f"NSE  = "
                f"{metricas['NSE']:.4f}"
            )

        except Exception as erro:

            print(
                "Não foi possível ajustar "
                "uma curva estável."
            )

            print(
                f"Motivo: {erro}"
            )

            parametros_periodos[nome] = None

    return parametros_periodos


# ============================================================
# FAIXA COMUM DE COTAS
# ============================================================

def determinar_faixa_comum(
    conjuntos: dict,
    parametros_periodos: dict,
):

    cotas_minimas = []
    cotas_maximas = []

    periodos_validos = []

    for nome, df in conjuntos.items():

        if (
            parametros_periodos.get(nome)
            is None
        ):
            continue

        if df.empty:
            continue

        cotas_minimas.append(
            float(df["Cota"].min())
        )

        cotas_maximas.append(
            float(df["Cota"].max())
        )

        periodos_validos.append(nome)

    if len(periodos_validos) < 2:

        return None

    cota_min_comum = max(
        cotas_minimas
    )

    cota_max_comum = min(
        cotas_maximas
    )

    if (
        cota_min_comum
        >= cota_max_comum
    ):
        return None

    return (
        cota_min_comum,
        cota_max_comum,
        periodos_validos,
    )


# ============================================================
# COMPARAÇÃO DAS CURVAS
# ============================================================

def comparar_curvas(
    conjuntos: dict,
    parametros_periodos: dict,
):

    faixa = determinar_faixa_comum(
        conjuntos,
        parametros_periodos,
    )

    print("\n" + "=" * 72)
    print("FAIXA COMUM PARA COMPARAÇÃO")
    print("=" * 72)

    if faixa is None:

        print(
            "Não existe faixa comum suficiente "
            "entre pelo menos duas curvas ajustadas."
        )

        return None

    (
        cota_min,
        cota_max,
        periodos_validos,
    ) = faixa

    print(
        f"Cota comum mínima: "
        f"{cota_min:.2f}"
    )

    print(
        f"Cota comum máxima: "
        f"{cota_max:.2f}"
    )

    print(
        "Períodos comparáveis: "
        + ", ".join(periodos_validos)
    )

    cotas = np.linspace(
        cota_min,
        cota_max,
        100,
    )

    comparacao = pd.DataFrame(
        {
            "Cota": cotas,
        }
    )

    for nome in periodos_validos:

        parametros = (
            parametros_periodos[nome]
        )

        comparacao[
            f"Q_{nome}"
        ] = curva_chave(
            cotas,
            parametros["a"],
            parametros["h0"],
            parametros["b"],
        )

    comparacao.to_csv(
        PASTA_SAIDA
        / "02_comparacao_curvas_faixa_comum.csv",
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8-sig",
    )

    return (
        comparacao,
        cota_min,
        cota_max,
        periodos_validos,
    )


# ============================================================
# COTAS DE REFERÊNCIA
# ============================================================

def gerar_tabela_referencia(
    comparacao_info,
    parametros_periodos,
):

    if comparacao_info is None:
        return None

    (
        comparacao,
        cota_min,
        cota_max,
        periodos_validos,
    ) = comparacao_info

    cotas_referencia = np.linspace(
        cota_min,
        cota_max,
        6,
    )

    linhas = []

    for cota in cotas_referencia:

        linha = {
            "Cota": float(cota),
        }

        valores = []

        for nome in periodos_validos:

            p = parametros_periodos[nome]

            q = float(
                curva_chave(
                    np.array([cota]),
                    p["a"],
                    p["h0"],
                    p["b"],
                )[0]
            )

            linha[
                f"Q_{nome}"
            ] = q

            valores.append(q)

        if len(valores) >= 2:

            media = np.mean(
                valores
            )

            amplitude = (
                max(valores)
                - min(valores)
            )

            if media != 0:

                diferenca_percentual = (
                    amplitude
                    / media
                    * 100
                )

            else:

                diferenca_percentual = (
                    np.nan
                )

            linha[
                "Amplitude_Q_m3s"
            ] = amplitude

            linha[
                "Amplitude_percentual"
            ] = diferenca_percentual

        linhas.append(linha)

    tabela = pd.DataFrame(
        linhas
    )

    tabela.to_csv(
        PASTA_SAIDA
        / "03_vazoes_cotas_referencia.csv",
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 72)
    print("VAZÕES PARA COTAS DE REFERÊNCIA")
    print("=" * 72)

    print(
        tabela.to_string(
            index=False
        )
    )

    return tabela


# ============================================================
# VALIDAÇÃO CRUZADA TEMPORAL
# ============================================================

def validacao_cruzada_temporal(
    conjuntos: dict,
    parametros_periodos: dict,
):

    linhas = []

    print("\n" + "=" * 72)
    print("APLICAÇÃO DE CADA CURVA AOS OUTROS PERÍODOS")
    print("=" * 72)

    for curva_nome, parametros in (
        parametros_periodos.items()
    ):

        if parametros is None:
            continue

        for dados_nome, df in (
            conjuntos.items()
        ):

            if df.empty:
                continue

            previsto = curva_chave(
                df["Cota"].to_numpy(
                    dtype=float
                ),
                parametros["a"],
                parametros["h0"],
                parametros["b"],
            )

            metricas = calcular_metricas(
                df["Vazao"],
                previsto,
            )

            linhas.append(
                {
                    "CurvaCalibradaEm": (
                        curva_nome
                    ),
                    "AplicadaAoPeriodo": (
                        dados_nome
                    ),
                    "N": len(df),
                    "MAE": metricas["MAE"],
                    "RMSE": metricas["RMSE"],
                    "Vies": metricas["Vies"],
                    "R2": metricas["R2"],
                    "NSE": metricas["NSE"],
                }
            )

    resultado = pd.DataFrame(
        linhas
    )

    resultado.to_csv(
        PASTA_SAIDA
        / "04_validacao_cruzada_temporal.csv",
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8-sig",
    )

    if not resultado.empty:

        print(
            resultado.to_string(
                index=False
            )
        )

    return resultado


# ============================================================
# SALVAMENTO DOS PARÂMETROS
# ============================================================

def salvar_parametros(
    conjuntos: dict,
    parametros_periodos: dict,
):

    linhas = []

    for nome, parametros in (
        parametros_periodos.items()
    ):

        df = conjuntos[nome]

        if parametros is None:

            linhas.append(
                {
                    "Periodo": nome,
                    "N": len(df),
                    "a": np.nan,
                    "H0": np.nan,
                    "b": np.nan,
                    "CurvaAjustada": False,
                }
            )

            continue

        linhas.append(
            {
                "Periodo": nome,
                "N": len(df),
                "a": parametros["a"],
                "H0": parametros["h0"],
                "b": parametros["b"],
                "CurvaAjustada": True,
            }
        )

    tabela = pd.DataFrame(
        linhas
    )

    tabela.to_csv(
        PASTA_SAIDA
        / "05_parametros_por_periodo.csv",
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8-sig",
    )

    return tabela


# ============================================================
# GRÁFICO DAS CURVAS POR PERÍODO
# ============================================================

def grafico_curvas_periodos(
    conjuntos,
    parametros_periodos,
):

    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    for nome, df in conjuntos.items():

        if df.empty:
            continue

        ax.scatter(
            df["Cota"],
            df["Vazao"],
            alpha=0.45,
            label=f"Medições {nome}",
        )

        parametros = (
            parametros_periodos.get(nome)
        )

        if parametros is None:
            continue

        h = np.linspace(
            df["Cota"].min(),
            df["Cota"].max(),
            300,
        )

        q = curva_chave(
            h,
            parametros["a"],
            parametros["h0"],
            parametros["b"],
        )

        ax.plot(
            h,
            q,
            linewidth=2,
            label=f"Curva {nome}",
        )

    ax.set_xlabel(
        "Cota observada"
    )

    ax.set_ylabel(
        "Vazão observada (m³/s)"
    )

    ax.set_title(
        "Curvas-chave independentes por período"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        PASTA_SAIDA
        / "06_curvas_por_periodo.png",
        dpi=220,
    )

    plt.close(fig)


# ============================================================
# GRÁFICO APENAS NA FAIXA COMUM
# ============================================================

def grafico_faixa_comum(
    comparacao_info,
):

    if comparacao_info is None:
        return

    (
        comparacao,
        cota_min,
        cota_max,
        periodos_validos,
    ) = comparacao_info

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    for nome in periodos_validos:

        ax.plot(
            comparacao["Cota"],
            comparacao[
                f"Q_{nome}"
            ],
            linewidth=2,
            label=nome,
        )

    ax.set_xlabel(
        "Cota observada"
    )

    ax.set_ylabel(
        "Vazão calculada (m³/s)"
    )

    ax.set_title(
        "Comparação das curvas-chave "
        "na faixa comum observada"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        title="Período de calibração"
    )

    fig.tight_layout()

    fig.savefig(
        PASTA_SAIDA
        / "07_curvas_faixa_comum.png",
        dpi=220,
    )

    plt.close(fig)


# ============================================================
# GRÁFICO DOS PARÂMETROS
# ============================================================

def grafico_parametros(
    tabela_parametros,
):

    validos = tabela_parametros[
        tabela_parametros[
            "CurvaAjustada"
        ]
    ].copy()

    if validos.empty:
        return

    for parametro in [
        "a",
        "H0",
        "b",
    ]:

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.bar(
            validos["Periodo"],
            validos[parametro],
        )

        ax.set_xlabel(
            "Período"
        )

        ax.set_ylabel(
            parametro
        )

        ax.set_title(
            f"Variação temporal do "
            f"parâmetro {parametro}"
        )

        ax.grid(
            axis="y",
            alpha=0.25,
        )

        fig.tight_layout()

        fig.savefig(
            PASTA_SAIDA
            / (
                f"08_parametro_"
                f"{parametro}.png"
            ),
            dpi=220,
        )

        plt.close(fig)


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 72)
    print("ANÁLISE DE ESTABILIDADE TEMPORAL DA CURVA-CHAVE")
    print("=" * 72)

    dados = preparar_dados()

    print(
        f"\nTotal de medições válidas: "
        f"{len(dados)}"
    )

    print(
        f"Período total: "
        f"{dados['Data'].min().date()}"
        f" a "
        f"{dados['Data'].max().date()}"
    )

    conjuntos = separar_periodos(
        dados
    )

    diagnostico = diagnosticar_periodos(
        conjuntos
    )

    parametros_periodos = (
        calibrar_periodos(
            conjuntos
        )
    )

    salvar_parametros(
        conjuntos,
        parametros_periodos,
    )

    comparacao_info = comparar_curvas(
        conjuntos,
        parametros_periodos,
    )

    gerar_tabela_referencia(
        comparacao_info,
        parametros_periodos,
    )

    validacao_cruzada_temporal(
        conjuntos,
        parametros_periodos,
    )

    grafico_curvas_periodos(
        conjuntos,
        parametros_periodos,
    )

    grafico_faixa_comum(
        comparacao_info
    )

    tabela_parametros = (
        salvar_parametros(
            conjuntos,
            parametros_periodos,
        )
    )

    grafico_parametros(
        tabela_parametros
    )

    print("\n" + "=" * 72)
    print("ANÁLISE CONCLUÍDA")
    print("=" * 72)

    print(
        "\nResultados salvos em:"
    )

    print(PASTA_SAIDA)

    print(
        "\nIMPORTANTE:"
    )

    print(
        "Esta análise é diagnóstica. "
        "Não substitua ainda a curva utilizada "
        "pelo SoberboPrev."
    )