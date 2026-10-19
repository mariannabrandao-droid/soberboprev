from __future__ import annotations
import numpy as np


def nivel_para_vazao(nivel_m, a: float, h0: float, b: float):
    """
    Curva-chave experimental:
        Q = a * (H - H0)^b, para H > H0
        Q = 0, caso contrário

    Os parâmetros desta V1 NÃO são calibrados para o Rio Soberbo.
    """
    h = np.asarray(nivel_m, dtype=float)
    excedente = np.maximum(h - h0, 0.0)
    q = a * np.power(excedente, b)
    return q
