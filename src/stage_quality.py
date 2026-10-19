from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURAÇÕES
# ============================================================

PASTA_RAIZ = Path(__file__).resolve().parents[1]

ARQUIVO_DATASET = (
    PASTA_RAIZ
    / "data"
    / "chuva_cota"
    / "dataset_chuva_cota_59245200.csv"
)

PASTA_SAIDA = (
    PASTA_RAIZ
    / "data"
    / "qualidade_cota"
)

PASTA_SAIDA.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def localizar_coluna(df, candidatos):
    """
    Localiza uma coluna ignorando diferenças entre
    maiúsculas e minúsculas.
    """

    mapa = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for candidato in candidatos:
        chave = candidato.lower()

        if chave in mapa:
            return mapa[chave]

    return None


def carregar_dataset():
    """
    Carrega o dataset chuva × cota já consolidado anteriormente.
    """

    if not ARQUIVO_DATASET.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado:\n{ARQUIVO_DATASET}"
        )

    df = pd.read_csv(
    ARQUIVO_DATASET,
    sep=",",
    encoding="utf-8-sig",
    )

    coluna_data = localizar_coluna(
        df,
        ["data", "date"],
    )

    coluna_cota = localizar_coluna(
        df,
        ["cota", "cota_observada"],
    )

    coluna_chuva = localizar_coluna(
        df,
        [
            "precipitacao_mm",
            "precipitacao",
            "chuva",
        ],
    )

    if coluna_data is None:
        raise ValueError(
            "Não foi possível localizar a coluna de data."
        )

    if coluna_cota is None:
        raise ValueError(
            "Não foi possível localizar a coluna de cota."
        )

    if coluna_chuva is None:
        raise ValueError(
            "Não foi possível localizar a coluna de precipitação."
        )

    df = df.rename(
        columns={
            coluna_data: "data",
            coluna_cota: "cota",
            coluna_chuva: "precipitacao_mm",
        }
    )

    df["data"] = pd.to_datetime(
        df["data"],
        errors="coerce",
    )

    df["cota"] = pd.to_numeric(
        df["cota"],
        errors="coerce",
    )

    df["precipitacao_mm"] = pd.to_numeric(
        df["precipitacao_mm"],
        errors="coerce",
    )

    df = (
        df
        .dropna(
            subset=[
                "data",
                "cota",
            ]
        )
        .sort_values("data")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# ANÁLISE DE QUALIDADE
# ============================================================

def analisar_qualidade(df):
    resultado = df.copy()

    # --------------------------------------------------------
    # 1. Variação da cota entre dias
    # --------------------------------------------------------

    resultado["cota_dia_anterior"] = (
        resultado["cota"].shift(1)
    )

    resultado["cota_dia_seguinte"] = (
        resultado["cota"].shift(-1)
    )

    resultado["delta_cota_1d"] = (
        resultado["cota"]
        - resultado["cota_dia_anterior"]
    )

    resultado["delta_cota_abs_1d"] = (
        resultado["delta_cota_1d"].abs()
    )

    # --------------------------------------------------------
    # 2. Precipitação antecedente
    # --------------------------------------------------------

    resultado["chuva_1d_antes"] = (
        resultado["precipitacao_mm"].shift(1)
    )

    resultado["chuva_2d_antes"] = (
        resultado["precipitacao_mm"].shift(2)
    )

    resultado["chuva_3d_antes"] = (
        resultado["precipitacao_mm"].shift(3)
    )

    resultado["chuva_acum_3d_incluindo_dia"] = (
        resultado["precipitacao_mm"]
        .rolling(
            window=3,
            min_periods=1,
        )
        .sum()
    )

    resultado["chuva_acum_7d_incluindo_dia"] = (
        resultado["precipitacao_mm"]
        .rolling(
            window=7,
            min_periods=1,
        )
        .sum()
    )

    # --------------------------------------------------------
    # 3. Limites estatísticos
    # --------------------------------------------------------

    p95 = resultado["cota"].quantile(0.95)
    p99 = resultado["cota"].quantile(0.99)
    p995 = resultado["cota"].quantile(0.995)

    q1 = resultado["cota"].quantile(0.25)
    q3 = resultado["cota"].quantile(0.75)

    iqr = q3 - q1

    limite_iqr = q3 + 3 * iqr

    delta_p99 = (
        resultado["delta_cota_abs_1d"]
        .dropna()
        .quantile(0.99)
    )

    # --------------------------------------------------------
    # 4. Sinais de atenção
    # --------------------------------------------------------

    resultado["flag_cota_p99"] = (
        resultado["cota"] > p99
    )

    resultado["flag_cota_extrema_iqr"] = (
        resultado["cota"] > limite_iqr
    )

    resultado["flag_salto_extremo"] = (
        resultado["delta_cota_abs_1d"]
        > delta_p99
    )

    resultado["flag_cota_alta_sem_chuva"] = (
        (resultado["cota"] > p95)
        & (
            resultado[
                "chuva_acum_3d_incluindo_dia"
            ] <= 1.0
        )
    )

    # --------------------------------------------------------
    # 5. Número de sinais simultâneos
    # --------------------------------------------------------

    colunas_flags = [
        "flag_cota_p99",
        "flag_cota_extrema_iqr",
        "flag_salto_extremo",
        "flag_cota_alta_sem_chuva",
    ]

    resultado["numero_flags"] = (
        resultado[colunas_flags]
        .astype(int)
        .sum(axis=1)
    )

    # --------------------------------------------------------
    # 6. Classificação diagnóstica
    # --------------------------------------------------------

    resultado["classificacao"] = "Sem sinal relevante"

    resultado.loc[
        resultado["numero_flags"] == 1,
        "classificacao",
    ] = "Revisar"

    resultado.loc[
        resultado["numero_flags"] == 2,
        "classificacao",
    ] = "Suspeito"

    resultado.loc[
        resultado["numero_flags"] >= 3,
        "classificacao",
    ] = "Altamente suspeito"

    limites = {
        "P95_cota": p95,
        "P99_cota": p99,
        "P99_5_cota": p995,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "limite_extremo_IQR": limite_iqr,
        "P99_delta_absoluto": delta_p99,
    }

    return resultado, limites


# ============================================================
# GRÁFICOS
# ============================================================

def gerar_grafico_serie(df):
    plt.figure(
        figsize=(16, 7)
    )

    plt.plot(
        df["data"],
        df["cota"],
        linewidth=1.2,
        label="Cota observada",
    )

    suspeitos = df[
        df["numero_flags"] >= 2
    ]

    plt.scatter(
        suspeitos["data"],
        suspeitos["cota"],
        s=70,
        label="Observações suspeitas",
        zorder=5,
    )

    plt.xlabel("Data")
    plt.ylabel("Cota observada")
    plt.title(
        "Controle de qualidade da série de cotas"
    )

    plt.legend()
    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    arquivo = (
        PASTA_SAIDA
        / "01_controle_qualidade_cota.png"
    )

    plt.savefig(
        arquivo,
        dpi=180,
    )

    plt.close()


def gerar_grafico_delta(df):
    plt.figure(
        figsize=(16, 7)
    )

    plt.scatter(
        df["data"],
        df["delta_cota_1d"],
        s=15,
        alpha=0.7,
    )

    plt.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    plt.xlabel("Data")
    plt.ylabel(
        "Variação diária da cota"
    )

    plt.title(
        "Variação da cota entre dias consecutivos"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    arquivo = (
        PASTA_SAIDA
        / "02_variacao_diaria_cota.png"
    )

    plt.savefig(
        arquivo,
        dpi=180,
    )

    plt.close()


# ============================================================
# EXECUÇÃO
# ============================================================

def main():

    print()
    print("=" * 72)
    print(
        "CONTROLE DE QUALIDADE DA SÉRIE DE COTAS — RIO SOBERBO"
    )
    print("=" * 72)

    df = carregar_dataset()

    print()
    print(
        f"Número de observações analisadas: {len(df)}"
    )

    print(
        "Período:",
        df["data"].min().date(),
        "a",
        df["data"].max().date(),
    )

    resultado, limites = analisar_qualidade(
        df
    )

    print()
    print("=" * 72)
    print("LIMITES ESTATÍSTICOS")
    print("=" * 72)

    for nome, valor in limites.items():
        print(
            f"{nome}: {valor:.3f}"
        )

    print()
    print("=" * 72)
    print("CLASSIFICAÇÃO DAS OBSERVAÇÕES")
    print("=" * 72)

    print(
        resultado[
            "classificacao"
        ].value_counts()
    )

    suspeitos = resultado[
        resultado["numero_flags"] >= 1
    ].copy()

    suspeitos = suspeitos.sort_values(
        [
            "numero_flags",
            "cota",
        ],
        ascending=[
            False,
            False,
        ],
    )

    print()
    print("=" * 72)
    print("OBSERVAÇÕES PARA REVISÃO")
    print("=" * 72)

    colunas_exibir = [
        "data",
        "cota",
        "cota_dia_anterior",
        "cota_dia_seguinte",
        "delta_cota_1d",
        "precipitacao_mm",
        "chuva_1d_antes",
        "chuva_2d_antes",
        "chuva_3d_antes",
        "chuva_acum_3d_incluindo_dia",
        "numero_flags",
        "classificacao",
    ]

    print(
        suspeitos[
            colunas_exibir
        ]
        .head(30)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Salvar resultados
    # --------------------------------------------------------

    resultado.to_csv(
        PASTA_SAIDA
        / "diagnostico_completo_cotas.csv",
        index=False,
    )

    suspeitos.to_csv(
        PASTA_SAIDA
        / "observacoes_para_revisao.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "indicador": list(
                limites.keys()
            ),
            "valor": list(
                limites.values()
            ),
        }
    ).to_csv(
        PASTA_SAIDA
        / "limites_estatisticos.csv",
        index=False,
    )

    gerar_grafico_serie(
        resultado
    )

    gerar_grafico_delta(
        resultado
    )

    print()
    print("=" * 72)
    print("PROCESSAMENTO CONCLUÍDO")
    print("=" * 72)

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
        "Os registros sinalizados não foram excluídos."
    )

    print(
        "As flags representam apenas critérios de revisão."
    )


if __name__ == "__main__":
    main()