from pathlib import Path
import calendar

import numpy as np
import pandas as pd


# ============================================================
# CAMINHOS
# ============================================================

RAIZ = Path(__file__).resolve().parents[1]
PASTA_ANA = RAIZ / "Chuvas - AN"

ARQUIVO_COTAS = PASTA_ANA / "59245200_Cotas.csv"
ARQUIVO_DESCARGA = PASTA_ANA / "59245200_ResumoDescarga.csv"


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def localizar_cabecalho(caminho):
    """
    Localiza a linha real do cabeçalho nos arquivos ANA.
    """

    codificacoes = [
        "latin1",
        "utf-8-sig",
        "cp1252",
    ]

    for encoding in codificacoes:

        try:

            with open(
                caminho,
                "r",
                encoding=encoding,
            ) as arquivo:

                for numero_linha, linha in enumerate(arquivo):

                    if linha.strip().startswith(
                        "EstacaoCodigo;"
                    ):
                        return numero_linha, encoding

        except UnicodeDecodeError:
            continue

    raise ValueError(
        f"Não foi possível localizar o cabeçalho em:\n{caminho}"
    )


def ler_csv_ana(caminho):

    linha_cabecalho, encoding = localizar_cabecalho(
        caminho
    )

    df = pd.read_csv(
        caminho,
        sep=";",
        encoding=encoding,
        skiprows=linha_cabecalho,
        dtype=str,
        engine="python",
    )

    df.columns = [
        str(coluna).strip()
        for coluna in df.columns
    ]

    return df


def converter_numero(valor):

    if pd.isna(valor):
        return np.nan

    texto = str(valor).strip().replace('"', "")

    if texto == "":
        return np.nan

    texto = texto.replace(
        ".",
        "",
    ).replace(
        ",",
        ".",
    )

    try:
        return float(texto)

    except ValueError:
        return np.nan


# ============================================================
# EXTRAÇÃO DA SÉRIE DIÁRIA DE COTAS
# ============================================================

def extrair_cotas_diarias():

    df = ler_csv_ana(
        ARQUIVO_COTAS
    )

    print()
    print("Colunas encontradas no arquivo de cotas:")
    print(
        " | ".join(df.columns)
    )

    df["NivelConsistencia"] = pd.to_numeric(
        df["NivelConsistencia"],
        errors="coerce",
    )

    df["_data_base"] = pd.to_datetime(
        df["Data"],
        dayfirst=True,
        errors="coerce",
    )

    # Se houver mais de uma linha para o mesmo mês,
    # prioriza o maior nível de consistência.
    df = (
        df
        .sort_values(
            [
                "_data_base",
                "NivelConsistencia",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .drop_duplicates(
            subset=["_data_base"],
            keep="first",
        )
    )

    registros = []

    for _, linha in df.iterrows():

        data_base = linha["_data_base"]

        if pd.isna(data_base):
            continue

        ano = data_base.year
        mes = data_base.month

        dias_mes = calendar.monthrange(
            ano,
            mes,
        )[1]

        for dia in range(
            1,
            dias_mes + 1,
        ):

            coluna = f"Cota{dia:02d}"

            if coluna not in df.columns:
                continue

            cota = converter_numero(
                linha.get(coluna)
            )

            if pd.isna(cota):
                continue

            data = pd.Timestamp(
                year=ano,
                month=mes,
                day=dia,
            )

            registros.append(
                {
                    "Data": data,
                    "Cota": cota,
                    "NivelConsistencia":
                        linha["NivelConsistencia"],
                }
            )

    resultado = pd.DataFrame(
        registros
    )

    resultado = (
        resultado
        .sort_values("Data")
        .reset_index(drop=True)
    )

    return resultado


# ============================================================
# LEITURA DAS MEDIÇÕES DE DESCARGA
# ============================================================

def extrair_descargas():

    df = ler_csv_ana(
        ARQUIVO_DESCARGA
    )

    print()
    print("Colunas encontradas no resumo de descarga:")
    print(
        " | ".join(df.columns)
    )

    if "Data" not in df.columns:

        raise ValueError(
            "A coluna Data não foi encontrada "
            "no resumo de descarga."
        )

    df["Data"] = pd.to_datetime(
        df["Data"],
        dayfirst=True,
        errors="coerce",
    )

    colunas_numericas = [
        "Cota",
        "Vazao",
        "AreaMolhada",
        "Largura",
        "VelMedia",
        "Profundidade",
    ]

    for coluna in colunas_numericas:

        if coluna in df.columns:

            df[coluna] = df[coluna].apply(
                converter_numero
            )

    df = (
        df
        .dropna(
            subset=["Data"]
        )
        .sort_values("Data")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# VERIFICAÇÃO DO EVENTO
# ============================================================

def main():

    print()
    print("=" * 72)
    print(
        "VERIFICAÇÃO DO REGISTRO DE COTA 710 — ESTAÇÃO 59245200"
    )
    print("=" * 72)

    print()
    print(
        "Nenhum dado será alterado nesta etapa."
    )

    # --------------------------------------------------------
    # COTAS DIÁRIAS
    # --------------------------------------------------------

    cotas = extrair_cotas_diarias()

    inicio = pd.Timestamp(
        "1981-02-01"
    )

    fim = pd.Timestamp(
        "1981-02-10"
    )

    periodo_cotas = cotas[
        (cotas["Data"] >= inicio)
        & (cotas["Data"] <= fim)
    ].copy()

    print()
    print("=" * 72)
    print("COTAS DIÁRIAS — 01/02/1981 A 10/02/1981")
    print("=" * 72)
    print()

    if periodo_cotas.empty:

        print(
            "Nenhuma cota encontrada."
        )

    else:

        print(
            periodo_cotas.to_string(
                index=False
            )
        )

    # Registro específico
    evento = cotas[
        cotas["Data"]
        == pd.Timestamp("1981-02-06")
    ]

    print()
    print("=" * 72)
    print("REGISTRO ESPECÍFICO — 06/02/1981")
    print("=" * 72)
    print()

    if evento.empty:

        print(
            "Não foi localizado registro para 06/02/1981."
        )

    else:

        print(
            evento.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # DESCARGAS
    # --------------------------------------------------------

    descargas = extrair_descargas()

    inicio_curto = pd.Timestamp(
        "1981-02-01"
    )

    fim_curto = pd.Timestamp(
        "1981-02-10"
    )

    descarga_curta = descargas[
        (descargas["Data"] >= inicio_curto)
        & (descargas["Data"] <= fim_curto)
    ].copy()

    print()
    print("=" * 72)
    print(
        "MEDIÇÕES DE DESCARGA — 01/02/1981 A 10/02/1981"
    )
    print("=" * 72)
    print()

    if descarga_curta.empty:

        print(
            "Nenhuma medição de descarga "
            "foi realizada nesse intervalo."
        )

    else:

        colunas_exibir = [
            coluna
            for coluna in [
                "Data",
                "Hora",
                "NumMedicao",
                "Cota",
                "Vazao",
                "AreaMolhada",
                "VelMedia",
                "NivelConsistencia",
            ]
            if coluna in descarga_curta.columns
        ]

        print(
            descarga_curta[
                colunas_exibir
            ].to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # INTERVALO AMPLIADO
    # --------------------------------------------------------

    inicio_ampliado = pd.Timestamp(
        "1981-01-01"
    )

    fim_ampliado = pd.Timestamp(
        "1981-03-15"
    )

    descarga_ampliada = descargas[
        (descargas["Data"] >= inicio_ampliado)
        & (descargas["Data"] <= fim_ampliado)
    ].copy()

    print()
    print("=" * 72)
    print(
        "MEDIÇÕES DE DESCARGA — 01/01/1981 A 15/03/1981"
    )
    print("=" * 72)
    print()

    if descarga_ampliada.empty:

        print(
            "Nenhuma medição disponível."
        )

    else:

        colunas_exibir = [
            coluna
            for coluna in [
                "Data",
                "Hora",
                "NumMedicao",
                "Cota",
                "Vazao",
                "AreaMolhada",
                "VelMedia",
                "NivelConsistencia",
            ]
            if coluna in descarga_ampliada.columns
        ]

        print(
            descarga_ampliada[
                colunas_exibir
            ].to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # MEDIÇÃO MAIS PRÓXIMA
    # --------------------------------------------------------

    data_evento = pd.Timestamp(
        "1981-02-06"
    )

    if not descargas.empty:

        descargas[
            "distancia_evento_dias"
        ] = (
            descargas["Data"]
            - data_evento
        ).abs().dt.days

        mais_proximas = (
            descargas
            .sort_values(
                "distancia_evento_dias"
            )
            .head(10)
        )

        print()
        print("=" * 72)
        print(
            "10 MEDIÇÕES DE DESCARGA MAIS PRÓXIMAS DE 06/02/1981"
        )
        print("=" * 72)
        print()

        colunas_exibir = [
            coluna
            for coluna in [
                "Data",
                "Hora",
                "NumMedicao",
                "Cota",
                "Vazao",
                "AreaMolhada",
                "VelMedia",
                "distancia_evento_dias",
            ]
            if coluna in mais_proximas.columns
        ]

        print(
            mais_proximas[
                colunas_exibir
            ].to_string(
                index=False
            )
        )

    print()
    print("=" * 72)
    print("VERIFICAÇÃO CONCLUÍDA")
    print("=" * 72)

    print()
    print(
        "IMPORTANTE: nenhuma observação foi "
        "corrigida ou excluída."
    )


if __name__ == "__main__":
    main()