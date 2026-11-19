"""Fluencia lenta bajo carga sostenida (caso 3).

Con viscoelasticidad lineal y coeficiente de Poisson constante, el principio
de correspondencia y el teorema de escalado (sección 4.2 del informe) dan:

* la tensión bajo carga constante no cambia con el tiempo, σ(x, t) = σ(x, 0);
* la apertura crece con la flexibilidad de fluencia J(t) = [1 + φ(t)] / E:

      Δ(t) = Δ_ref · (E_ref / E_f) · (F / F_ref) · [1 + φ(t)].

φ(t) es el coeficiente de fluencia del material impreso. Las fichas técnicas
no lo dan, y la literatura disponible [14, 15] solo ordena los materiales
cualitativamente. Por eso aquí:

1. se usa la ley de Findley, φ(t) = (t / τ)^n, con parámetros de escenario
   (no medidos) que respetan ese orden, para estimar órdenes de magnitud; y
2. se entrega `fit_findley`, para calibrar φ(t) con un ensayo casero de
   peso colgado sobre piezas impresas (ver scripts/ajuste_fluencia.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit


@dataclass(frozen=True)
class FindleyScenario:
    """φ(t) = (t/τ)^n, t en horas. Valores de ESCENARIO, no de ensayo."""

    material: str
    tau_h: float
    n: float
    note: str

    def phi(self, t_h):
        return (np.asarray(t_h, dtype=float) / self.tau_h) ** self.n


# Escenarios a 23 °C y tensión moderada (≤ 30 % de σ_t). Se eligieron para
# reproducir el orden cualitativo de [14, 15]: PC el más estable, PLA el peor
# de los rígidos, poliamida sin fibra la que más fluye y la fibra la reduce.
# φ(1000 h) resultante: PC ≈ 0,3; PAHT-CF ≈ 0,5; ABS ≈ 0,6; ASA ≈ 0,7;
# PETG ≈ 0,8; PLA ≈ 1,0.
SCENARIOS = {
    "PC": FindleyScenario("PC", 5.5e5, 0.20, "el más estable [14]"),
    "PAHT-CF": FindleyScenario("PAHT-CF", 3.2e4, 0.20, "PA con fibra: la fibra reduce la fluencia [15]"),
    "ABS": FindleyScenario("ABS", 1.3e4, 0.20, "moderada [14, 15]"),
    "ASA": FindleyScenario("ASA", 6.0e3, 0.20, "similar al ABS; rotura a 96 h a 29 °C sin fibra [15]"),
    "PETG": FindleyScenario("PETG", 3.0e3, 0.20, "sin datos comparables; se supone entre ASA y PLA"),
    "PLA": FindleyScenario("PLA", 1.0e3, 0.20, "la peor de los rígidos [14]"),
}


def opening_creep(op0: float, phi):
    """Apertura en el tiempo a partir de la apertura instantánea op0."""
    return op0 * (1.0 + np.asarray(phi))


def load_for_opening(op_lim: float, op_per_N: float, phi_t: float) -> float:
    """Carga sostenida máxima que no supera op_lim tras el tiempo t."""
    return op_lim / (op_per_N * (1.0 + phi_t))


def fit_findley(t_h, opening, op0=None):
    """Ajusta Δ(t) = Δ0 [1 + (t/τ)^n] a mediciones de un ensayo de carga sostenida.

    t_h: tiempos en horas (> 0); opening: aperturas medidas [mm]; op0: apertura
    instantánea medida al cargar (si se omite, también se ajusta).
    Devuelve (Δ0, τ [h], n).
    """
    t = np.asarray(t_h, float)
    y = np.asarray(opening, float)
    if op0 is None:
        f = lambda tt, d0, lt, n: d0 * (1 + (tt / 10 ** lt) ** n)  # noqa: E731
        p, _ = curve_fit(f, t, y, p0=[y[0], 3.0, 0.25], maxfev=20000)
        return float(p[0]), float(10 ** p[1]), float(p[2])
    f = lambda tt, lt, n: op0 * (1 + (tt / 10 ** lt) ** n)  # noqa: E731
    p, _ = curve_fit(f, t, y, p0=[3.0, 0.25], maxfev=20000)
    return float(op0), float(10 ** p[0]), float(p[1])
