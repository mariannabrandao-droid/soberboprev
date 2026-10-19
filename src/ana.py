from __future__ import annotations

from pathlib import Path
import calendar
import pandas as pd


def carregar_chuva_ana(caminho: str | Path, preferir_consistido: bool = True) -> pd.DataFrame:
    """
    Lê o CSV mensal exportado pelo Hidro/ANA e o converte para uma série diária.

    O arquivo da ANA possui metadados antes do cabeçalho, separador ';',
    codificação latin-1 e colunas Chuva01...Chuva31.

    Retorna colunas:
        data, chuva_mm, nivel_consistencia, estacao_codigo
    """
    caminho = Path(caminho)

    df = pd.read_csv(
        caminho,
        sep=";",
        encoding="latin1",
        skiprows=14,
        decimal=",",
        low_memory=False,
    )

    df["Data"] = pd.to_datetime(df["Data"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Data"]).copy()

    if preferir_consistido and (df["NivelConsistencia"] == 2).any():
        df = df[df["NivelConsistencia"] == 2].copy()

    chuva_cols = [f"Chuva{i:02d}" for i in range(1, 32) if f"Chuva{i:02d}" in df.columns]

    registros = []
    for _, row in df.iterrows():
        ano = int(row["Data"].year)
        mes = int(row["Data"].month)
        ultimo_dia = calendar.monthrange(ano, mes)[1]

        for dia in range(1, ultimo_dia + 1):
            col = f"Chuva{dia:02d}"
            if col not in chuva_cols:
                continue

            valor = pd.to_numeric(row[col], errors="coerce")
            if pd.isna(valor):
                continue

            registros.append(
                {
                    "data": pd.Timestamp(ano, mes, dia),
                    "chuva_mm": float(valor),
                    "nivel_consistencia": int(row["NivelConsistencia"]),
                    "estacao_codigo": str(row["EstacaoCodigo"]),
                }
            )

    serie = pd.DataFrame(registros)
    if serie.empty:
        return serie

    serie = (
        serie.sort_values("data")
        .drop_duplicates(subset=["data"], keep="last")
        .reset_index(drop=True)
    )
    return serie


def resumo_chuva(serie: pd.DataFrame) -> dict:
    if serie.empty:
        return {}

    chuva = serie["chuva_mm"].dropna()
    positivos = chuva[chuva > 0]

    idx_max = chuva.idxmax()
    return {
        "inicio": serie["data"].min(),
        "fim": serie["data"].max(),
        "n_dias": int(len(serie)),
        "n_dias_chuva": int((chuva > 0).sum()),
        "max_diaria_mm": float(chuva.max()),
        "data_maxima": serie.loc[idx_max, "data"],
        "p90_mm": float(chuva.quantile(0.90)),
        "p95_mm": float(chuva.quantile(0.95)),
        "p99_mm": float(chuva.quantile(0.99)),
        "media_dias_chuvosos_mm": float(positivos.mean()) if not positivos.empty else 0.0,
    }
