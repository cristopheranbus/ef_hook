"""Solución analítica de viga curva (Winkler) para la corona (sección 8.5)."""
from __future__ import annotations

import numpy as np

from .geometry import Y0, Y_TOPE, YT


def winkler_circular(M: float, N: float, R: float, c: float):
    """Tensión en la fibra interior de una barra curva de sección circular.

    R: radio del eje [mm]; c: radio de la sección [mm]. Devuelve un dict con
    r_n, ē, σ_i y el factor de curvatura K_i = σ_flexión / (M/Z).
    """
    A = np.pi * c ** 2
    rn = c ** 2 / (2 * (R - np.sqrt(R ** 2 - c ** 2)))
    e = R - rn
    ri = R - c
    s_bend = M * (rn - ri) / (A * e * ri)
    s_i = s_bend + N / A
    Z = np.pi * c ** 3 / 4
    return dict(rn=rn, e=e, ri=ri, A=A, sigma_bend=s_bend, sigma_i=s_i,
                straight=M / Z, Ki=s_bend / (M / Z))


def crown_moment(F: float = 100.0):
    """Brazo y momento en la corona en el caso 3: e = y_t - ȳ_carga."""
    y_load = 0.5 * (Y0 + Y_TOPE)
    arm = YT - y_load
    return arm, F * arm
