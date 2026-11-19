"""Calibra la fluencia con un ensayo casero de carga sostenida.

Ensayo sugerido: colgar (o empujar) las patas con una carga constante conocida
F por pata, medir con un calibre la apertura entre las puntas justo al cargar
y luego a 1, 3, 10, 24, 72 y 168 h. Guardar un CSV con columnas:

    t_h,apertura_mm

(la primera fila con t_h = 0 es la apertura instantánea). Uso:

    python scripts/ajuste_fluencia.py ensayo.csv [--limite 3.0] [--anios 5]

Imprime los parámetros de Findley, Δ(t) = Δ0 [1 + (t/τ)^n], y la apertura
extrapolada al plazo pedido (la extrapolación más allá de ~10 veces la
duración del ensayo es incierta).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ganchofem.creep import fit_findley  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--limite", type=float, default=None, help="apertura admisible [mm]")
    ap.add_argument("--anios", type=float, default=5.0)
    a = ap.parse_args()
    d = np.genfromtxt(a.csv, delimiter=",", names=True)
    t, y = d["t_h"], d["apertura_mm"]
    op0 = float(y[t == 0][0]) if np.any(t == 0) else None
    m = t > 0
    d0, tau, n = fit_findley(t[m], y[m], op0)
    T = a.anios * 8760
    yT = d0 * (1 + (T / tau) ** n)
    print(f"Δ0 = {d0:.3f} mm, τ = {tau:.3g} h, n = {n:.3f}")
    print(f"φ(1000 h) = {(1000 / tau) ** n:.2f};  apertura a {a.anios:g} años ≈ {yT:.2f} mm")
    if a.limite:
        if yT > a.limite:
            t_lim = tau * (a.limite / d0 - 1) ** (1 / n)
            print(f"Se alcanza el límite de {a.limite} mm a las ≈ {t_lim:.0f} h ({t_lim / 8760:.2f} años)")
        else:
            print(f"No se alcanza el límite de {a.limite} mm en {a.anios:g} años")


if __name__ == "__main__":
    main()
