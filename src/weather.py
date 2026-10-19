from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests


OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

TIMEZONE_PADRAO = "America/Sao_Paulo"


def carregar_previsao_open_meteo(
    latitude: float,
    longitude: float,
    forecast_days: int = 8,
    timezone: str = TIMEZONE_PADRAO,
) -> pd.DataFrame:
    """
    Consulta a API Open-Meteo e retorna a previsão horária
    de precipitação para o ponto informado.

    Retorna:
        - data_hora
        - precipitacao_mm
        - probabilidade_precipitacao_pct
    """

    parametros = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "precipitation,precipitation_probability",
        "forecast_days": forecast_days,
        "timezone": timezone,
    }

    resposta = requests.get(
        OPEN_METEO_URL,
        params=parametros,
        timeout=30,
    )

    resposta.raise_for_status()

    dados = resposta.json()

    if "hourly" not in dados:
        raise ValueError(
            "A resposta da Open-Meteo não contém dados horários."
        )

    horario = dados["hourly"]

    if (
        "time" not in horario
        or "precipitation" not in horario
        or "precipitation_probability" not in horario
    ):
        raise ValueError(
            "A resposta da Open-Meteo não contém todas as "
            "variáveis meteorológicas esperadas."
        )

    df = pd.DataFrame(
        {
            "data_hora": pd.to_datetime(horario["time"]),
            "precipitacao_mm": pd.to_numeric(
                horario["precipitation"],
                errors="coerce",
            ),
            "probabilidade_precipitacao_pct": pd.to_numeric(
                horario["precipitation_probability"],
                errors="coerce",
            ),
        }
    )

    df = (
        df
        .dropna(subset=["data_hora"])
        .sort_values("data_hora")
        .reset_index(drop=True)
    )

    df["precipitacao_mm"] = (
        df["precipitacao_mm"]
        .fillna(0.0)
    )

    return df


def obter_horario_atual(
    timezone: str = TIMEZONE_PADRAO,
) -> pd.Timestamp:
    """
    Retorna o horário atual no fuso definido,
    arredondado para o início da hora.

    O timezone é removido ao final para ficar compatível
    com as datas retornadas pela Open-Meteo.
    """

    agora = datetime.now(
        ZoneInfo(timezone)
    )

    agora = pd.Timestamp(agora)

    agora = agora.floor("h")

    return agora.tz_localize(None)


def filtrar_previsao_futura(
    df: pd.DataFrame,
    agora: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Remove horas anteriores ao momento atual da previsão.
    """

    if df.empty:
        return df.copy()

    if agora is None:
        agora = obter_horario_atual()

    futuro = df[
        df["data_hora"] >= agora
    ].copy()

    return futuro.reset_index(drop=True)


def resumo_previsao(
    df: pd.DataFrame,
    agora: pd.Timestamp | None = None,
) -> dict:
    """
    Calcula os principais indicadores da previsão futura.

    Diferentemente da versão inicial, as próximas 24 h
    e 72 h são calculadas a partir do horário atual.
    """

    futuro = filtrar_previsao_futura(
        df,
        agora=agora,
    )

    if futuro.empty:
        return {
            "chuva_24h_mm": 0.0,
            "chuva_72h_mm": 0.0,
            "chuva_total_mm": 0.0,
            "max_chuva_horaria_mm": 0.0,
            "max_probabilidade_pct": 0.0,
        }

    inicio = futuro["data_hora"].min()

    limite_24h = inicio + pd.Timedelta(hours=24)
    limite_72h = inicio + pd.Timedelta(hours=72)

    janela_24h = futuro[
        futuro["data_hora"] < limite_24h
    ]

    janela_72h = futuro[
        futuro["data_hora"] < limite_72h
    ]

    chuva_24h = (
        janela_24h["precipitacao_mm"].sum()
    )

    chuva_72h = (
        janela_72h["precipitacao_mm"].sum()
    )

    chuva_total = (
        futuro["precipitacao_mm"].sum()
    )

    max_chuva_horaria = (
        futuro["precipitacao_mm"].max()
    )

    max_probabilidade = (
        futuro["probabilidade_precipitacao_pct"]
        .max()
    )

    return {
        "chuva_24h_mm": float(chuva_24h),
        "chuva_72h_mm": float(chuva_72h),
        "chuva_total_mm": float(chuva_total),
        "max_chuva_horaria_mm": float(
            max_chuva_horaria
        ),
        "max_probabilidade_pct": float(
            max_probabilidade
        ),
    }


def agregar_previsao_diaria(
    df: pd.DataFrame,
    agora: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Agrega somente a previsão futura por dia.

    Para o dia atual, considera apenas as horas que ainda
    não ocorreram no momento da consulta.
    """

    futuro = filtrar_previsao_futura(
        df,
        agora=agora,
    )

    if futuro.empty:
        return pd.DataFrame(
            columns=[
                "data",
                "precipitacao_mm",
                "probabilidade_max_pct",
            ]
        )

    diario = futuro.copy()

    diario["data"] = (
        diario["data_hora"].dt.date
    )

    diario = (
        diario
        .groupby(
            "data",
            as_index=False,
        )
        .agg(
            precipitacao_mm=(
                "precipitacao_mm",
                "sum",
            ),
            probabilidade_max_pct=(
                "probabilidade_precipitacao_pct",
                "max",
            ),
        )
    )

    diario["precipitacao_mm"] = (
        diario["precipitacao_mm"]
        .round(1)
    )

    return diario


def preparar_previsao_diaria_interface(
    df: pd.DataFrame,
    agora: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Prepara o resumo diário para apresentação no Streamlit,
    incluindo os rótulos Hoje, D+1, D+2...
    """

    diario = agregar_previsao_diaria(
        df,
        agora=agora,
    )

    if diario.empty:
        return diario

    diario = diario.copy()

    primeira_data = diario["data"].iloc[0]

    diario["horizonte"] = [
        (
            "Hoje"
            if data == primeira_data
            else f"D+{(data - primeira_data).days}"
        )
        for data in diario["data"]
    ]

    return diario[
        [
            "horizonte",
            "data",
            "precipitacao_mm",
            "probabilidade_max_pct",
        ]
    ]