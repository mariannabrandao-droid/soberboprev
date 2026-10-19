from pathlib import Path
import pandas as pd


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
    / "analise_sincronismo"
)

PASTA_SAIDA.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CARREGAMENTO
# ============================================================

def carregar_base():

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
# ANALISE
# ============================================================

def main():

    print()
    print("=" * 78)
    print("SOBERBOPREV - SINCRONISMO CHUVA X ELEVACAO DO RIO")
    print("=" * 78)

    df = carregar_base()

    # --------------------------------------------------------
    # Variacao da cota para o dia seguinte
    # --------------------------------------------------------

    df["delta_Dmais1"] = (
        df["cota_Dmais1"]
        - df["cota"]
    )

    # --------------------------------------------------------
    # Chuva do dia seguinte
    #
    # Como a base possui calendario diario completo,
    # shift(-1) representa efetivamente D+1.
    # --------------------------------------------------------

    df["chuva_Dmais1"] = (
        df["precipitacao_mm"]
        .shift(-1)
    )

    # --------------------------------------------------------
    # Selecionar as 25 maiores elevacoes
    # --------------------------------------------------------

    eventos = (
        df.dropna(
            subset=[
                "cota",
                "cota_Dmais1",
                "delta_Dmais1",
                "chuva_Dmais1",
            ]
        )
        .sort_values(
            "delta_Dmais1",
            ascending=False,
        )
        .head(25)
        .copy()
    )

    eventos["data_evento"] = (
        eventos["data"]
        + pd.Timedelta(days=1)
    )

    colunas = [
        "data",
        "data_evento",
        "cota",
        "cota_Dmais1",
        "delta_Dmais1",
        "chuva_lag_2d",
        "chuva_lag_1d",
        "precipitacao_mm",
        "chuva_Dmais1",
        "chuva_acum_3d",
        "chuva_acum_7d",
    ]

    resultado = eventos[
        colunas
    ].copy()

    resultado = resultado.rename(
        columns={
            "precipitacao_mm": "chuva_D",
            "cota": "cota_D",
            "cota_Dmais1": "cota_Dmais1",
        }
    )

    # --------------------------------------------------------
    # Quanto da chuva associada aos grandes eventos aparece
    # no proprio dia da elevacao?
    # --------------------------------------------------------

    resultado["chuva_antecedente_3d"] = (
        resultado["chuva_D"]
        + resultado["chuva_lag_1d"]
        + resultado["chuva_lag_2d"]
    )

    resultado["chuva_com_Dmais1"] = (
        resultado["chuva_antecedente_3d"]
        + resultado["chuva_Dmais1"]
    )

    # --------------------------------------------------------
    # Salvar
    # --------------------------------------------------------

    resultado.to_csv(
        PASTA_SAIDA
        / "sincronismo_25_maiores_eventos.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Impressao
    # --------------------------------------------------------

    print()
    print("25 MAIORES ELEVACOES E CHUVA ASSOCIADA")
    print("-" * 78)
    print()

    print(
        resultado[
            [
                "data",
                "data_evento",
                "cota_D",
                "cota_Dmais1",
                "delta_Dmais1",
                "chuva_D",
                "chuva_Dmais1",
                "chuva_antecedente_3d",
                "chuva_com_Dmais1",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    # --------------------------------------------------------
    # Comparacao simples
    # --------------------------------------------------------

    media_antes = (
        resultado[
            "chuva_antecedente_3d"
        ].mean()
    )

    media_dia_evento = (
        resultado[
            "chuva_Dmais1"
        ].mean()
    )

    eventos_sem_chuva_antes = (
        resultado[
            "chuva_antecedente_3d"
        ]
        <= 0
    ).sum()

    eventos_com_chuva_dia = (
        resultado[
            "chuva_Dmais1"
        ]
        > 0
    ).sum()

    print()
    print("=" * 78)
    print("RESUMO")
    print("=" * 78)
    print()

    print(
        f"Chuva antecedente media (D-2 ate D): "
        f"{media_antes:.2f} mm"
    )

    print(
        f"Chuva media no proprio dia da elevacao (D+1): "
        f"{media_dia_evento:.2f} mm"
    )

    print(
        f"Eventos sem chuva nos 3 dias antecedentes: "
        f"{eventos_sem_chuva_antes} de {len(resultado)}"
    )

    print(
        f"Eventos com chuva registrada no proprio dia da elevacao: "
        f"{eventos_com_chuva_dia} de {len(resultado)}"
    )

    print()
    print("=" * 78)
    print("ANALISE CONCLUIDA")
    print("=" * 78)

    print()
    print("Arquivo salvo em:")
    print(
        PASTA_SAIDA
        / "sincronismo_25_maiores_eventos.csv"
    )

    print()
    print(
        "Nenhum dado original foi modificado."
    )


if __name__ == "__main__":
    main()