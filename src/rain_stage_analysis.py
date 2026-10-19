from pathlib import Path
import calendar

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURAÇÕES
# ============================================================

RAIZ = Path(__file__).resolve().parents[1]

PASTA_ANA = RAIZ / "Chuvas - AN"

ARQUIVO_CHUVA = (
    PASTA_ANA
    / "2242094_Chuvas.csv"
)

ARQUIVO_COTA = (
    PASTA_ANA
    / "59245200_Cotas.csv"
)

PASTA_SAIDA = (
    RAIZ
    / "data"
    / "chuva_cota"
)

PASTA_SAIDA.mkdir(
    parents=True,
    exist_ok=True,
)

# Registro identificado na etapa de controle de qualidade.
# Ele NÃO será apagado da base de auditoria.
DATA_REGISTRO_SUSPEITO = pd.Timestamp(
    "1981-02-06"
)

COTA_REGISTRO_SUSPEITO = 710.0


# ============================================================
# LEITURA DOS ARQUIVOS ANA
# ============================================================

def localizar_cabecalho(caminho):
    """
    Localiza a linha real do cabeçalho dos arquivos
    exportados pelo HidroWeb/ANA.
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

                for numero_linha, linha in enumerate(
                    arquivo
                ):

                    if linha.strip().startswith(
                        "EstacaoCodigo;"
                    ):
                        return numero_linha, encoding

        except UnicodeDecodeError:
            continue

    raise ValueError(
        f"Não foi possível localizar o cabeçalho em:\n"
        f"{caminho}"
    )


def ler_csv_ana(caminho):
    """
    Lê um CSV da ANA após localizar o cabeçalho real.
    """

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
    """
    Converte números no padrão brasileiro para float.
    """

    if pd.isna(valor):
        return np.nan

    texto = (
        str(valor)
        .strip()
        .replace('"', "")
    )

    if texto == "":
        return np.nan

    texto = (
        texto
        .replace(".", "")
        .replace(",", ".")
    )

    try:
        return float(texto)

    except ValueError:
        return np.nan


# ============================================================
# EXPANSÃO DA PRECIPITAÇÃO MENSAL PARA DIÁRIA
# ============================================================

def extrair_chuva_diaria():

    df = ler_csv_ana(
        ARQUIVO_CHUVA
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

    # Para meses duplicados, priorizamos a linha
    # de maior nível de consistência.
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

            coluna = f"Chuva{dia:02d}"

            if coluna not in df.columns:
                continue

            chuva = converter_numero(
                linha.get(coluna)
            )

            if pd.isna(chuva):
                continue

            data = pd.Timestamp(
                year=ano,
                month=mes,
                day=dia,
            )

            registros.append(
                {
                    "data": data,
                    "precipitacao_mm": chuva,
                    "consistencia_chuva":
                        linha["NivelConsistencia"],
                }
            )

    resultado = pd.DataFrame(
        registros
    )

    resultado = (
        resultado
        .sort_values("data")
        .drop_duplicates(
            subset=["data"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    return resultado


# ============================================================
# EXPANSÃO DAS COTAS MENSAIS PARA DIÁRIAS
# ============================================================

def extrair_cota_diaria():

    df = ler_csv_ana(
        ARQUIVO_COTA
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
                    "data": data,
                    "cota": cota,
                    "consistencia_cota":
                        linha["NivelConsistencia"],
                }
            )

    resultado = pd.DataFrame(
        registros
    )

    resultado = (
        resultado
        .sort_values("data")
        .drop_duplicates(
            subset=["data"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    return resultado


# ============================================================
# CONSTRUÇÃO DO CALENDÁRIO
# ============================================================

def construir_base_calendario(
    chuva,
    cota,
):
    """
    Cria um calendário diário completo no período de
    sobreposição entre as séries.

    Isso evita que shift(1) signifique simplesmente
    'linha anterior'. A partir daqui, cada linha
    representa exatamente um dia do calendário.
    """

    inicio = max(
        chuva["data"].min(),
        cota["data"].min(),
    )

    fim = min(
        chuva["data"].max(),
        cota["data"].max(),
    )

    calendario = pd.DataFrame(
        {
            "data": pd.date_range(
                start=inicio,
                end=fim,
                freq="D",
            )
        }
    )

    base = calendario.merge(
        chuva,
        on="data",
        how="left",
    )

    base = base.merge(
        cota,
        on="data",
        how="left",
    )

    base["tem_chuva"] = (
        base["precipitacao_mm"].notna()
    )

    base["tem_cota"] = (
        base["cota"].notna()
    )

    base["dia_completo"] = (
        base["tem_chuva"]
        & base["tem_cota"]
    )

    return base


# ============================================================
# FEATURES TEMPORAIS
# ============================================================

def criar_features_calendario(base):
    """
    Como a base foi reindexada para frequência diária,
    shift(1) agora corresponde de fato a D-1,
    shift(2) a D-2 etc.
    """

    df = base.copy()

    # --------------------------------------------------------
    # CHUVA DEFASADA
    # --------------------------------------------------------

    for lag in range(1, 8):

        df[
            f"chuva_lag_{lag}d"
        ] = (
            df["precipitacao_mm"]
            .shift(lag)
        )

    # --------------------------------------------------------
    # CHUVA ACUMULADA
    # --------------------------------------------------------

    for janela in [
        2,
        3,
        5,
        7,
    ]:

        df[
            f"chuva_acum_{janela}d"
        ] = (
            df["precipitacao_mm"]
            .rolling(
                window=janela,
                min_periods=janela,
            )
            .sum()
        )

    # --------------------------------------------------------
    # COTA ANTECEDENTE
    # --------------------------------------------------------

    for lag in [
        1,
        2,
        3,
    ]:

        df[
            f"cota_lag_{lag}d"
        ] = (
            df["cota"]
            .shift(lag)
        )

    # --------------------------------------------------------
    # ALVOS FUTUROS
    # --------------------------------------------------------

    for horizonte in range(
        1,
        8,
    ):

        df[
            f"cota_Dmais{horizonte}"
        ] = (
            df["cota"]
            .shift(-horizonte)
        )

    return df


# ============================================================
# CONTROLE DO REGISTRO 710
# ============================================================

def marcar_registro_suspeito(df):

    resultado = df.copy()

    resultado[
        "flag_registro_suspeito"
    ] = False

    mascara = (
        (
            resultado["data"]
            == DATA_REGISTRO_SUSPEITO
        )
        & (
            resultado["cota"]
            == COTA_REGISTRO_SUSPEITO
        )
    )

    resultado.loc[
        mascara,
        "flag_registro_suspeito",
    ] = True

    return resultado


def criar_base_modelagem(df):
    """
    Cria a base destinada à modelagem.

    O registro 710 é retirado da variável cota
    e dos alvos/features que dependem diretamente dele,
    sem modificar a base de auditoria.

    Não é feita interpolação.
    """

    modelo = df.copy()

    mascara = (
        modelo[
            "flag_registro_suspeito"
        ]
    )

    # Não substituímos por outro nível.
    modelo.loc[
        mascara,
        "cota",
    ] = np.nan

    # IMPORTANTE:
    # depois de retirar a observação suspeita,
    # precisamos recalcular todas as variáveis
    # que dependem da cota.
    for lag in [
        1,
        2,
        3,
    ]:

        modelo[
            f"cota_lag_{lag}d"
        ] = (
            modelo["cota"]
            .shift(lag)
        )

    for horizonte in range(
        1,
        8,
    ):

        modelo[
            f"cota_Dmais{horizonte}"
        ] = (
            modelo["cota"]
            .shift(-horizonte)
        )

    return modelo


# ============================================================
# DIAGNÓSTICO DE LACUNAS
# ============================================================

def diagnosticar_lacunas(base):

    total_dias = len(base)

    chuva_ausente = int(
        base[
            "precipitacao_mm"
        ].isna().sum()
    )

    cota_ausente = int(
        base[
            "cota"
        ].isna().sum()
    )

    ambos_presentes = int(
        base[
            "dia_completo"
        ].sum()
    )

    diagnostico = pd.DataFrame(
        {
            "indicador": [
                "Total de dias no calendário",
                "Dias sem precipitação",
                "Dias sem cota",
                "Dias com chuva e cota",
            ],
            "valor": [
                total_dias,
                chuva_ausente,
                cota_ausente,
                ambos_presentes,
            ],
        }
    )

    return diagnostico


# ============================================================
# CORRELAÇÕES
# ============================================================

def calcular_correlacoes(df):

    variaveis = [
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
    ]

    registros = []

    for variavel in variaveis:

        dados = df[
            [
                variavel,
                "cota",
            ]
        ].dropna()

        if len(dados) < 3:

            correlacao = np.nan

        else:

            correlacao = (
                dados[variavel]
                .corr(
                    dados["cota"]
                )
            )

        registros.append(
            {
                "variavel": variavel,
                "n": len(dados),
                "correlacao_com_cota":
                    correlacao,
            }
        )

    resultado = pd.DataFrame(
        registros
    )

    resultado = resultado.sort_values(
        "correlacao_com_cota",
        ascending=False,
    )

    return resultado


# ============================================================
# EVENTOS DE COTA ELEVADA
# ============================================================

def identificar_eventos_elevados(df):

    base_valida = df[
        df["cota"].notna()
    ].copy()

    p90 = (
        base_valida["cota"]
        .quantile(0.90)
    )

    p95 = (
        base_valida["cota"]
        .quantile(0.95)
    )

    base_valida[
        "acima_p90"
    ] = (
        base_valida["cota"]
        >= p90
    )

    base_valida[
        "acima_p95"
    ] = (
        base_valida["cota"]
        >= p95
    )

    eventos = base_valida[
        base_valida["acima_p90"]
    ].copy()

    eventos = eventos.sort_values(
        "cota",
        ascending=False,
    )

    return eventos, p90, p95


# ============================================================
# GRÁFICOS
# ============================================================

def grafico_correlacao_lags(
    correlacoes,
):

    nomes = [
        "precipitacao_mm",
        "chuva_lag_1d",
        "chuva_lag_2d",
        "chuva_lag_3d",
        "chuva_lag_4d",
        "chuva_lag_5d",
        "chuva_lag_6d",
        "chuva_lag_7d",
    ]

    dados = (
        correlacoes
        .set_index("variavel")
        .reindex(nomes)
        .reset_index()
    )

    rotulos = [
        "Mesmo dia",
        "D-1",
        "D-2",
        "D-3",
        "D-4",
        "D-5",
        "D-6",
        "D-7",
    ]

    plt.figure(
        figsize=(11, 6)
    )

    plt.bar(
        rotulos,
        dados[
            "correlacao_com_cota"
        ],
    )

    plt.axhline(
        0,
        linewidth=1,
    )

    plt.ylabel(
        "Correlação de Pearson com a cota"
    )

    plt.xlabel(
        "Defasagem da precipitação"
    )

    plt.title(
        "Precipitação × cota — defasagens calendárias"
    )

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    plt.tight_layout()

    plt.savefig(
        PASTA_SAIDA
        / "01_correlacao_defasagens_chuva_cota.png",
        dpi=180,
    )

    plt.close()


def grafico_acumulados(
    correlacoes,
):

    nomes = [
        "precipitacao_mm",
        "chuva_acum_2d",
        "chuva_acum_3d",
        "chuva_acum_5d",
        "chuva_acum_7d",
    ]

    dados = (
        correlacoes
        .set_index("variavel")
        .reindex(nomes)
        .reset_index()
    )

    rotulos = [
        "Dia",
        "2 dias",
        "3 dias",
        "5 dias",
        "7 dias",
    ]

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        rotulos,
        dados[
            "correlacao_com_cota"
        ],
    )

    plt.axhline(
        0,
        linewidth=1,
    )

    plt.ylabel(
        "Correlação de Pearson com a cota"
    )

    plt.xlabel(
        "Janela de precipitação"
    )

    plt.title(
        "Precipitação acumulada × cota"
    )

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    plt.tight_layout()

    plt.savefig(
        PASTA_SAIDA
        / "02_correlacao_chuva_acumulada_cota.png",
        dpi=180,
    )

    plt.close()


def grafico_serie_temporal(
    df_auditoria,
    df_modelo,
):

    fig, eixo_cota = plt.subplots(
        figsize=(16, 7)
    )

    eixo_chuva = (
        eixo_cota.twinx()
    )

    dados = df_modelo[
        df_modelo[
            "cota"
        ].notna()
    ]

    eixo_cota.plot(
        dados["data"],
        dados["cota"],
        linewidth=1.2,
        label="Cota",
    )

    eixo_chuva.bar(
        df_modelo["data"],
        df_modelo[
            "precipitacao_mm"
        ],
        alpha=0.25,
        width=1.0,
        label="Precipitação",
    )

    suspeito = df_auditoria[
        df_auditoria[
            "flag_registro_suspeito"
        ]
    ]

    if not suspeito.empty:

        eixo_cota.scatter(
            suspeito["data"],
            suspeito["cota"],
            s=80,
            marker="x",
            label="Registro suspeito preservado na auditoria",
            zorder=10,
        )

    eixo_cota.set_xlabel(
        "Data"
    )

    eixo_cota.set_ylabel(
        "Cota"
    )

    eixo_chuva.set_ylabel(
        "Precipitação (mm)"
    )

    eixo_chuva.invert_yaxis()

    eixo_cota.set_title(
        "Série histórica de precipitação e cota"
    )

    linhas1, labels1 = (
        eixo_cota.get_legend_handles_labels()
    )

    linhas2, labels2 = (
        eixo_chuva.get_legend_handles_labels()
    )

    eixo_cota.legend(
        linhas1 + linhas2,
        labels1 + labels2,
        loc="upper left",
    )

    eixo_cota.grid(
        alpha=0.20,
    )

    fig.tight_layout()

    fig.savefig(
        PASTA_SAIDA
        / "03_serie_temporal_chuva_cota.png",
        dpi=180,
    )

    plt.close(fig)


def grafico_dados_ausentes(
    df,
):

    ausencias = pd.DataFrame(
        {
            "data": df["data"],
            "chuva_ausente":
                df["precipitacao_mm"].isna().astype(int),
            "cota_ausente":
                df["cota"].isna().astype(int),
        }
    )

    plt.figure(
        figsize=(16, 5)
    )

    chuva = ausencias[
        ausencias["chuva_ausente"] == 1
    ]

    cota = ausencias[
        ausencias["cota_ausente"] == 1
    ]

    plt.scatter(
        chuva["data"],
        np.zeros(len(chuva)),
        marker="|",
        s=80,
        label="Chuva ausente",
    )

    plt.scatter(
        cota["data"],
        np.ones(len(cota)),
        marker="|",
        s=80,
        label="Cota ausente",
    )

    plt.yticks(
        [0, 1],
        [
            "Precipitação",
            "Cota",
        ],
    )

    plt.xlabel(
        "Data"
    )

    plt.title(
        "Lacunas identificadas no calendário diário"
    )

    plt.legend()

    plt.grid(
        axis="x",
        alpha=0.20,
    )

    plt.tight_layout()

    plt.savefig(
        PASTA_SAIDA
        / "04_lacunas_serie_historica.png",
        dpi=180,
    )

    plt.close()


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

def main():

    print()
    print("=" * 76)
    print(
        "ANÁLISE CHUVA × COTA — RIO SOBERBO"
    )
    print(
        "VERSÃO COM CONTROLE DE CALENDÁRIO E QUALIDADE"
    )
    print("=" * 76)

    # --------------------------------------------------------
    # 1. LEITURA
    # --------------------------------------------------------

    chuva = extrair_chuva_diaria()
    cota = extrair_cota_diaria()

    print()
    print("PRECIPITAÇÃO")
    print(
        f"Observações disponíveis: {len(chuva)}"
    )
    print(
        f"Período: "
        f"{chuva['data'].min().date()} "
        f"a "
        f"{chuva['data'].max().date()}"
    )

    print()
    print("COTA")
    print(
        f"Observações disponíveis: {len(cota)}"
    )
    print(
        f"Período: "
        f"{cota['data'].min().date()} "
        f"a "
        f"{cota['data'].max().date()}"
    )

    # --------------------------------------------------------
    # 2. CALENDÁRIO REAL
    # --------------------------------------------------------

    base = construir_base_calendario(
        chuva,
        cota,
    )

    diagnostico = diagnosticar_lacunas(
        base
    )

    print()
    print("=" * 76)
    print("DIAGNÓSTICO DO CALENDÁRIO")
    print("=" * 76)
    print()

    print(
        diagnostico.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 3. FEATURES
    # --------------------------------------------------------

    base = criar_features_calendario(
        base
    )

    # --------------------------------------------------------
    # 4. AUDITORIA
    # --------------------------------------------------------

    base_auditoria = (
        marcar_registro_suspeito(
            base
        )
    )

    quantidade_suspeitos = int(
        base_auditoria[
            "flag_registro_suspeito"
        ].sum()
    )

    print()
    print("=" * 76)
    print("CONTROLE DO REGISTRO SUSPEITO")
    print("=" * 76)

    print()
    print(
        f"Registros especificamente sinalizados: "
        f"{quantidade_suspeitos}"
    )

    if quantidade_suspeitos > 0:

        print()
        print(
            base_auditoria.loc[
                base_auditoria[
                    "flag_registro_suspeito"
                ],
                [
                    "data",
                    "cota",
                    "precipitacao_mm",
                    "consistencia_cota",
                ],
            ].to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # 5. BASE PARA MODELAGEM
    # --------------------------------------------------------

    base_modelo = criar_base_modelagem(
        base_auditoria
    )

    # --------------------------------------------------------
    # 6. CORRELAÇÕES RECALCULADAS
    # --------------------------------------------------------

    correlacoes = calcular_correlacoes(
        base_modelo
    )

    print()
    print("=" * 76)
    print("CORRELAÇÕES RECALCULADAS")
    print("=" * 76)
    print()

    print(
        correlacoes.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 7. EVENTOS ELEVADOS
    # --------------------------------------------------------

    eventos, p90, p95 = (
        identificar_eventos_elevados(
            base_modelo
        )
    )

    print()
    print("=" * 76)
    print("COTAS ELEVADAS — REFERÊNCIA ESTATÍSTICA")
    print("=" * 76)

    print()
    print(
        f"P90 da série após tratamento: "
        f"{p90:.2f}"
    )

    print(
        f"P95 da série após tratamento: "
        f"{p95:.2f}"
    )

    print()
    print(
        "ATENÇÃO: P90 e P95 são referências "
        "estatísticas exploratórias."
    )

    print(
        "Eles NÃO representam cotas oficiais "
        "de atenção, alerta ou inundação."
    )

    print()
    print("20 maiores cotas preservadas:")
    print()

    colunas_eventos = [
        coluna
        for coluna in [
            "data",
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
        if coluna in eventos.columns
    ]

    print(
        eventos[
            colunas_eventos
        ]
        .head(20)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 8. SALVAR ARQUIVOS
    # --------------------------------------------------------

    base_auditoria.to_csv(
        PASTA_SAIDA
        / "dataset_chuva_cota_auditoria.csv",
        index=False,
    )

    base_modelo.to_csv(
        PASTA_SAIDA
        / "dataset_chuva_cota_modelagem.csv",
        index=False,
    )

    correlacoes.to_csv(
        PASTA_SAIDA
        / "correlacoes_chuva_cota_corrigidas.csv",
        index=False,
    )

    eventos.to_csv(
        PASTA_SAIDA
        / "eventos_cotas_elevadas_corrigidos.csv",
        index=False,
    )

    diagnostico.to_csv(
        PASTA_SAIDA
        / "diagnostico_lacunas.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 9. GRÁFICOS
    # --------------------------------------------------------

    grafico_correlacao_lags(
        correlacoes
    )

    grafico_acumulados(
        correlacoes
    )

    grafico_serie_temporal(
        base_auditoria,
        base_modelo,
    )

    grafico_dados_ausentes(
        base_auditoria
    )

    # --------------------------------------------------------
    # 10. FINALIZAÇÃO
    # --------------------------------------------------------

    print()
    print("=" * 76)
    print("PROCESSAMENTO CONCLUÍDO")
    print("=" * 76)

    print()
    print(
        "Arquivos gerados em:"
    )

    print(
        PASTA_SAIDA
    )

    print()
    print("Principais arquivos:")

    print(
        "- dataset_chuva_cota_auditoria.csv"
    )

    print(
        "- dataset_chuva_cota_modelagem.csv"
    )

    print(
        "- correlacoes_chuva_cota_corrigidas.csv"
    )

    print(
        "- eventos_cotas_elevadas_corrigidos.csv"
    )

    print(
        "- diagnostico_lacunas.csv"
    )

    print()
    print(
        "O registro de 710 foi preservado na base "
        "de auditoria e convertido em ausente somente "
        "na base destinada à modelagem."
    )

    print(
        "Nenhuma interpolação foi realizada."
    )


if __name__ == "__main__":
    main()