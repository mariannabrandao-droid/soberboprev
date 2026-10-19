from __future__ import annotations
import numpy as np
import pandas as pd


def gerar_niveis_sinteticos(
    nivel_base: float = 0.55,
    nivel_pico: float = 1.50,
    n_passos: int = 17,
    passo_min: int = 15,
    inicio: str = "2026-09-03 08:00",
):
    """
    Gera uma onda de nível apenas para demonstração da arquitetura.
    Não representa evento observado no Rio Soberbo.
    """
    n_passos = max(int(n_passos), 7)
    t = np.linspace(-2.5, 2.5, n_passos)

    # Onda assimétrica simples: subida relativamente rápida e recessão mais lenta.
    gauss = np.exp(-(t ** 2))
    assimetria = 1 + 0.12 * np.maximum(t, 0)
    onda = gauss * assimetria
    onda = onda / onda.max()

    nivel = nivel_base + (nivel_pico - nivel_base) * onda

    datas = pd.date_range(inicio, periods=n_passos, freq=f"{passo_min}min")
    return pd.DataFrame({"data_hora": datas, "nivel_m": nivel})
