from __future__ import annotations
import numpy as np


def coeficientes_muskingum(k_h: float, x: float, dt_h: float):
    if k_h <= 0:
        raise ValueError("K deve ser maior que zero.")
    if not (0 <= x <= 0.5):
        raise ValueError("X deve estar entre 0 e 0,5.")
    if dt_h <= 0:
        raise ValueError("Δt deve ser maior que zero.")

    den = k_h - k_h * x + 0.5 * dt_h
    c0 = (-k_h * x + 0.5 * dt_h) / den
    c1 = (k_h * x + 0.5 * dt_h) / den
    c2 = (k_h - k_h * x - 0.5 * dt_h) / den

    limite_inf = 2 * k_h * x
    limite_sup = 2 * k_h * (1 - x)
    condicao_nao_negativa = limite_inf <= dt_h <= limite_sup

    return {
        "C0": c0,
        "C1": c1,
        "C2": c2,
        "limite_dt_min_h": limite_inf,
        "limite_dt_max_h": limite_sup,
        "condicao_recomendada": condicao_nao_negativa,
    }


def rotear_muskingum(entrada, k_h: float, x: float, dt_h: float, saida_inicial=None):
    """
    O(t+Δt) = C0 I(t+Δt) + C1 I(t) + C2 O(t)
    """
    entrada = np.asarray(entrada, dtype=float)
    if entrada.ndim != 1 or len(entrada) < 2:
        raise ValueError("A série de entrada precisa ter pelo menos dois valores.")

    coefs = coeficientes_muskingum(k_h, x, dt_h)

    saida = np.zeros_like(entrada, dtype=float)
    saida[0] = entrada[0] if saida_inicial is None else float(saida_inicial)

    for i in range(len(entrada) - 1):
        saida[i + 1] = (
            coefs["C0"] * entrada[i + 1]
            + coefs["C1"] * entrada[i]
            + coefs["C2"] * saida[i]
        )

    return saida, coefs
