from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit


# ============================================================
# CONFIGURAÇÕES
# ============================================================

PASTA_PROJETO = Path(__file__).resolve().parent.parent
PASTA_DADOS = PASTA_PROJETO / "Chuvas - AN"
PASTA_SAIDA = PASTA_PROJETO / "data" / "calibracao"

PASTA_SAIDA.mkdir(
    parents=True,
    exist_ok=True,
)

ESTACAO = "59245200"

# Período mais denso e relativamente homogêneo.
INICIO_PERIODO_BASE = pd.Timestamp("1976-01-01")
FIM_PERIODO_BASE = pd.Timestamp("1981-12-31")

# Os últimos 20% cronológicos desse período serão
# reservados para validação interna.
FRACAO_CALIBRACAO = 0.80


# ============================================================
# LOCALIZAÇÃO DO ARQUIVO
# ============================================================

def localizar_arquivo(padrao: str) -> Path:

    encontrados = sorted(
        PASTA_DADOS.glob(f"{padrao}*.csv")
    )

    if not encontrados:
        raise FileNotFoundError(
            f"Nenhum arquivo iniciado por '{padrao}' "
            f"foi encontrado em:\n{PASTA_DADOS}"
        )

    return encontrados[0]


# ============================================================
# LEITURA ANA
# ============================================================

def ler_csv_ana(caminho: Path) -> pd.DataFrame:

    texto = caminho.read_text(
        encoding="cp1252",
        errors="replace",
    )

    linhas = texto.splitlines()

    indice_cabecalho = None

    for i, linha in enumerate(linhas):

        if (
            "EstacaoCodigo" in linha
            and "Data" in linha
            and ";" in linha
        ):
            indice_cabecalho = i
            break

    if indice_cabecalho is None:
        raise ValueError(
            "Cabeçalho da tabela ANA não localizado."
        )

    tabela_texto = "\n".join(
        linhas[indice_cabecalho:]
    )

    df = pd.read_csv(
        StringIO(tabela_texto),
        sep=";",
        decimal=",",
        low_memory=False,
    )

    return df


# ============================================================
# PREPARAÇÃO
# ============================================================

def preparar_dados() -> pd.DataFrame:

    caminho = localizar_arquivo(
        f"{ESTACAO}_ResumoDescarga"
    )

    print("\nArquivo utilizado:")
    print(caminho)

    df = ler_csv_ana(caminho)

    dados = df[
        [
            "Data",
            "Cota",
            "Vazao",
        ]
    ].copy()

    dados["Data"] = pd.to_datetime(
        dados["Data"],
        dayfirst=True,
        errors="coerce",
    )

    dados["Cota"] = pd.to_numeric(
        dados["Cota"],
        errors="coerce",
    )

    dados["Vazao"] = pd.to_numeric(
        dados["Vazao"],
        errors="coerce",
    )

    dados = dados.dropna(
        subset=[
            "Data",
            "Cota",
            "Vazao",
        ]
    )

    dados = dados[
        (dados["Cota"] > 0)
        & (dados["Vazao"] > 0)
    ]

    dados = dados.drop_duplicates(
        subset=[
            "Data",
            "Cota",
            "Vazao",
        ]
    )

    dados = (
        dados
        .sort_values("Data")
        .reset_index(drop=True)
    )

    return dados


# ============================================================
# CURVA-CHAVE
# ============================================================

def curva_chave(
    h,
    a,
    h0,
    b,
):
    """
    Q = a * (H - H0)^b
    """

    diferenca = np.maximum(
        h - h0,
        1e-6,
    )

    return a * (
        diferenca ** b
    )


# ============================================================
# SEPARAÇÃO DOS CONJUNTOS
# ============================================================

def separar_conjuntos(
    dados: pd.DataFrame,
):

    periodo_base = dados[
        (dados["Data"] >= INICIO_PERIODO_BASE)
        & (dados["Data"] <= FIM_PERIODO_BASE)
    ].copy()

    posteriores = dados[
        dados["Data"] > FIM_PERIODO_BASE
    ].copy()

    periodo_base = (
        periodo_base
        .sort_values("Data")
        .reset_index(drop=True)
    )

    ponto_corte = int(
        len(periodo_base)
        * FRACAO_CALIBRACAO
    )

    calibracao = (
        periodo_base
        .iloc[:ponto_corte]
        .copy()
    )

    validacao = (
        periodo_base
        .iloc[ponto_corte:]
        .copy()
    )

    return (
        calibracao,
        validacao,
        posteriores,
    )


# ============================================================
# AJUSTE DA CURVA
# ============================================================

def calibrar_curva(
    calibracao: pd.DataFrame,
):

    h = calibracao["Cota"].to_numpy(
        dtype=float
    )

    q = calibracao["Vazao"].to_numpy(
        dtype=float
    )

    h_min = float(h.min())

    chute_inicial = [
        0.001,
        h_min - 5.0,
        2.0,
    ]

    limites_inferiores = [
        1e-8,
        -200.0,
        0.2,
    ]

    limites_superiores = [
        100.0,
        h_min - 0.001,
        5.0,
    ]

    parametros, covariancia = curve_fit(
        curva_chave,
        h,
        q,
        p0=chute_inicial,
        bounds=(
            limites_inferiores,
            limites_superiores,
        ),
        maxfev=100000,
    )

    a, h0, b = parametros

    return {
        "a": float(a),
        "h0": float(h0),
        "b": float(b),
        "covariancia": covariancia,
    }


# ============================================================
# MÉTRICAS
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

    erro = previsto - observado

    mae = np.mean(
        np.abs(erro)
    )

    rmse = np.sqrt(
        np.mean(
            erro ** 2
        )
    )

    vies = np.mean(
        erro
    )

    denominador = np.sum(
        (
            observado
            - np.mean(observado)
        ) ** 2
    )

    if denominador > 0:

        r2 = 1 - (
            np.sum(
                erro ** 2
            )
            / denominador
        )

        nse = r2

    else:
        r2 = np.nan
        nse = np.nan

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "Vies": float(vies),
        "R2": float(r2),
        "NSE": float(nse),
    }


# ============================================================
# APLICAÇÃO DA CURVA
# ============================================================

def avaliar_conjunto(
    df: pd.DataFrame,
    parametros: dict,
    nome_grupo: str,
):

    resultado = df.copy()

    resultado["VazaoCalculada"] = curva_chave(
        resultado["Cota"].to_numpy(
            dtype=float
        ),
        parametros["a"],
        parametros["h0"],
        parametros["b"],
    )

    resultado["Erro"] = (
        resultado["VazaoCalculada"]
        - resultado["Vazao"]
    )

    resultado["ErroAbsoluto"] = (
        resultado["Erro"].abs()
    )

    resultado["ErroPercentual"] = (
        resultado["ErroAbsoluto"]
        / resultado["Vazao"]
        * 100
    )

    resultado["Grupo"] = nome_grupo

    metricas = calcular_metricas(
        resultado["Vazao"],
        resultado["VazaoCalculada"],
    )

    return resultado, metricas


# ============================================================
# IMPRESSÃO DAS MÉTRICAS
# ============================================================

def imprimir_metricas(
    titulo: str,
    df: pd.DataFrame,
    metricas: dict,
):

    print("\n" + "=" * 70)
    print(titulo)
    print("=" * 70)

    print(
        f"Número de observações: "
        f"{len(df)}"
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
        f"Viés = "
        f"{metricas['Vies']:.3f} m³/s"
    )

    print(
        f"R²   = "
        f"{metricas['R2']:.4f}"
    )

    print(
        f"NSE  = "
        f"{metricas['NSE']:.4f}"
    )


# ============================================================
# AVALIAÇÃO DOS EXTREMOS
# ============================================================

def avaliar_extremos(
    resultado: pd.DataFrame,
    nome: str,
):

    if resultado.empty:
        return

    limite = resultado[
        "Vazao"
    ].quantile(0.90)

    extremos = resultado[
        resultado["Vazao"] >= limite
    ].copy()

    if len(extremos) < 2:
        return

    metricas = calcular_metricas(
        extremos["Vazao"],
        extremos["VazaoCalculada"],
    )

    imprimir_metricas(
        f"{nome} — 10% MAIORES VAZÕES",
        extremos,
        metricas,
    )


# ============================================================
# GRÁFICOS
# ============================================================

def gerar_grafico_curva(
    calibracao,
    validacao,
    posteriores,
    parametros,
):

    todos = pd.concat(
        [
            calibracao,
            validacao,
            posteriores,
        ],
        ignore_index=True,
    )

    h_min = todos["Cota"].min()
    h_max = todos["Cota"].max()

    h_curva = np.linspace(
        h_min,
        h_max,
        500,
    )

    q_curva = curva_chave(
        h_curva,
        parametros["a"],
        parametros["h0"],
        parametros["b"],
    )

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    ax.scatter(
        calibracao["Cota"],
        calibracao["Vazao"],
        label="Calibração",
        alpha=0.65,
    )

    ax.scatter(
        validacao["Cota"],
        validacao["Vazao"],
        label="Validação interna",
        alpha=0.85,
    )

    if not posteriores.empty:

        ax.scatter(
            posteriores["Cota"],
            posteriores["Vazao"],
            label="Teste temporal posterior",
            marker="x",
            s=70,
        )

    ax.plot(
        h_curva,
        q_curva,
        label="Curva calibrada",
        linewidth=2,
    )

    ax.set_xlabel(
        "Cota observada"
    )

    ax.set_ylabel(
        "Vazão (m³/s)"
    )

    ax.set_title(
        "Curva-chave calibrada e validação"
    )

    ax.legend()

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        PASTA_SAIDA
        / "05_curva_calibrada_validacao.png",
        dpi=220,
    )

    plt.close(fig)


def gerar_grafico_observado_previsto(
    resultados,
):

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    maior = 0

    for resultado, nome in resultados:

        if resultado.empty:
            continue

        ax.scatter(
            resultado["Vazao"],
            resultado["VazaoCalculada"],
            label=nome,
            alpha=0.75,
        )

        maior = max(
            maior,
            resultado["Vazao"].max(),
            resultado["VazaoCalculada"].max(),
        )

    ax.plot(
        [0, maior],
        [0, maior],
        linestyle="--",
        label="Linha 1:1",
    )

    ax.set_xlabel(
        "Vazão observada ANA (m³/s)"
    )

    ax.set_ylabel(
        "Vazão calculada (m³/s)"
    )

    ax.set_title(
        "Vazão observada × vazão calculada"
    )

    ax.legend()

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        PASTA_SAIDA
        / "06_observado_vs_calculado.png",
        dpi=220,
    )

    plt.close(fig)


def gerar_grafico_residuos_temporais(
    resultado_total,
):

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.scatter(
        resultado_total["Data"],
        resultado_total["Erro"],
        alpha=0.75,
    )

    ax.axhline(
        0,
        linestyle="--",
    )

    ax.set_xlabel(
        "Data da medição"
    )

    ax.set_ylabel(
        "Erro Q calculada - Q observada (m³/s)"
    )

    ax.set_title(
        "Evolução temporal dos resíduos da curva-chave"
    )

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        PASTA_SAIDA
        / "07_residuos_ao_longo_do_tempo.png",
        dpi=220,
    )

    plt.close(fig)


# ============================================================
# SALVAMENTO
# ============================================================

def salvar_resultados(
    resultados,
    parametros,
):

    resultado_total = pd.concat(
        resultados,
        ignore_index=True,
    )

    resultado_total.to_csv(
        PASTA_SAIDA
        / "resultado_calibracao_validacao.csv",
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8-sig",
    )

    parametros_df = pd.DataFrame(
        [
            {
                "Estacao": ESTACAO,
                "a": parametros["a"],
                "H0": parametros["h0"],
                "b": parametros["b"],
            }
        ]
    )

    parametros_df.to_csv(
        PASTA_SAIDA
        / "parametros_curva_chave.csv",
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8-sig",
    )

    return resultado_total


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    dados = preparar_dados()

    (
        calibracao,
        validacao,
        posteriores,
    ) = separar_conjuntos(
        dados
    )

    print("\n" + "=" * 70)
    print("SEPARAÇÃO DOS DADOS")
    print("=" * 70)

    print(
        f"Calibração: "
        f"{len(calibracao)} medições"
    )

    print(
        f"Validação interna: "
        f"{len(validacao)} medições"
    )

    print(
        f"Teste temporal posterior: "
        f"{len(posteriores)} medições"
    )

    print(
        f"\nCalibração:"
        f" {calibracao['Data'].min().date()}"
        f" a {calibracao['Data'].max().date()}"
    )

    print(
        f"Validação:"
        f" {validacao['Data'].min().date()}"
        f" a {validacao['Data'].max().date()}"
    )

    if not posteriores.empty:

        print(
            f"Teste posterior:"
            f" {posteriores['Data'].min().date()}"
            f" a {posteriores['Data'].max().date()}"
        )

    parametros = calibrar_curva(
        calibracao
    )

    print("\n" + "=" * 70)
    print("PARÂMETROS CALIBRADOS")
    print("=" * 70)

    print(
        f"a  = {parametros['a']:.8f}"
    )

    print(
        f"H0 = {parametros['h0']:.4f}"
    )

    print(
        f"b  = {parametros['b']:.6f}"
    )

    print("\nEquação obtida:")

    print(
        "Q = "
        f"{parametros['a']:.8f} "
        "* "
        f"(H - ({parametros['h0']:.4f}))"
        f"^{parametros['b']:.6f}"
    )

    (
        resultado_calibracao,
        metricas_calibracao,
    ) = avaliar_conjunto(
        calibracao,
        parametros,
        "Calibracao",
    )

    (
        resultado_validacao,
        metricas_validacao,
    ) = avaliar_conjunto(
        validacao,
        parametros,
        "Validacao interna",
    )

    (
        resultado_posterior,
        metricas_posterior,
    ) = avaliar_conjunto(
        posteriores,
        parametros,
        "Teste temporal posterior",
    )

    imprimir_metricas(
        "DESEMPENHO — CALIBRAÇÃO",
        resultado_calibracao,
        metricas_calibracao,
    )

    imprimir_metricas(
        "DESEMPENHO — VALIDAÇÃO INTERNA",
        resultado_validacao,
        metricas_validacao,
    )

    if not resultado_posterior.empty:

        imprimir_metricas(
            "DESEMPENHO — TESTE TEMPORAL POSTERIOR",
            resultado_posterior,
            metricas_posterior,
        )

    avaliar_extremos(
        resultado_validacao,
        "VALIDAÇÃO INTERNA",
    )

    avaliar_extremos(
        resultado_posterior,
        "TESTE TEMPORAL POSTERIOR",
    )

    resultado_total = salvar_resultados(
        [
            resultado_calibracao,
            resultado_validacao,
            resultado_posterior,
        ],
        parametros,
    )

    gerar_grafico_curva(
        calibracao,
        validacao,
        posteriores,
        parametros,
    )

    gerar_grafico_observado_previsto(
        [
            (
                resultado_calibracao,
                "Calibração",
            ),
            (
                resultado_validacao,
                "Validação interna",
            ),
            (
                resultado_posterior,
                "Teste temporal posterior",
            ),
        ]
    )

    gerar_grafico_residuos_temporais(
        resultado_total
    )

    print("\n" + "=" * 70)
    print("PROCESSAMENTO CONCLUÍDO")
    print("=" * 70)

    print("\nResultados salvos em:")
    print(PASTA_SAIDA)

    print(
        "\nNÃO substitua ainda os parâmetros "
        "do rating_curve.py."
    )