"""Geometría paramétrica del gancho en U (sección 2.1 del informe).

El eje del gancho vive en el plano XY (plano de impresión):

* dos patas rectas en x = ±a, desde y0 hasta yb = yt - Rb,
* dos arcos de radio Rb centrados en (±xc, yb), con xc = a - Rb,
* un tramo recto superior en y = yt para |x| <= xc.

Una sección elíptica de semiejes A(s) (en el plano) y B (en z), desplazada
δ(s) en la dirección normal, se barre a lo largo del eje. El sólido es

    Ω = { p : φ(p) <= 0 },  φ = ((d_n - δ)/A)^2 + (z/B)^2 - 1

donde d_n es la distancia en el plano al eje, positiva hacia afuera de la U.
Todas las longitudes están en mm.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

A_LEG = 22.5   # semiancho del eje (patas en x = ±a)
Y0 = -30.0     # extremo inferior de las patas
YT = 24.0      # altura del eje del tramo superior
R_SEC = 6.0    # radio de la sección Ø12
Y_TOPE = -22.5  # las puntas de 7,5 mm entran en los topes (y <= -22,5)


@dataclass(frozen=True)
class Geometry:
    """Parámetros de una variante del gancho."""

    name: str
    Rb: float                 # radio de esquina medido en el eje [mm]
    A_top: float = R_SEC      # semieje en el plano en el tramo superior [mm]
    k_delta: float = 0.0      # -1: engrosado hacia adentro, +1: hacia afuera
    B: float = R_SEC          # semieje en z [mm]
    y_tr0: float = -15.0      # inicio de la transición de sección
    y_tr1: float = 5.0        # fin de la transición de sección
    label: str = field(default="", compare=False)

    # ------------------------------------------------------------------
    @property
    def a(self) -> float:
        return A_LEG

    @property
    def xc(self) -> float:
        return A_LEG - self.Rb

    @property
    def yb(self) -> float:
        return YT - self.Rb

    @property
    def bbox(self):
        """Caja que contiene al sólido (xmin, xmax, ymin, ymax, zmin, zmax)."""
        amax = max(self.A_top, R_SEC) + abs(self.k_delta) * max(self.A_top - R_SEC, 0)
        r = amax + 1.0
        return (-A_LEG - r, A_LEG + r, Y0, YT + r, -self.B, self.B)

    # ------------------------------------------------------------------
    def section(self, y_axis):
        """Semieje A y desplazamiento δ en función de la altura del punto del eje.

        La sección vale Ø12 bajo y_tr0 y alcanza A_top sobre y_tr1, con una
        transición suave (smoothstep) entre ambas. δ = k_delta (A - 6) mantiene
        fijo el contorno exterior cuando k_delta = -1.
        """
        t = np.clip((y_axis - self.y_tr0) / (self.y_tr1 - self.y_tr0), 0.0, 1.0)
        s = t * t * (3.0 - 2.0 * t)
        A = R_SEC + (self.A_top - R_SEC) * s
        delta = self.k_delta * (A - R_SEC)
        return A, delta

    # ------------------------------------------------------------------
    def project(self, x, y):
        """Proyección analítica de (x, y) sobre el eje.

        Devuelve (d_n, y_axis, dist): distancia normal con signo (positiva hacia
        afuera de la U), altura del punto del eje más cercano y distancia en el
        plano a ese punto (incluye la componente tangencial fuera del tramo).
        """
        x = np.abs(np.asarray(x, dtype=float))  # simetría respecto de x = 0
        y = np.asarray(y, dtype=float)
        a, xc, yb, Rb = A_LEG, self.xc, self.yb, self.Rb

        # --- pata recta: x = a, y en [Y0, yb]
        yl = np.clip(y, Y0, yb)
        dn_leg = x - a
        dist_leg = np.hypot(x - a, y - yl)
        # Bajo Y0 el sólido termina en un corte plano.
        dist_leg = np.where(y < Y0, np.inf, dist_leg)

        # --- arco: centro (xc, yb), ángulo entre 0 (pata) y 90° (tramo superior)
        dx, dy = x - xc, y - yb
        r = np.hypot(dx, dy)
        th = np.arctan2(dy, dx)
        thc = np.clip(th, 0.0, np.pi / 2)
        px, py = xc + Rb * np.cos(thc), yb + Rb * np.sin(thc)
        dist_arc = np.hypot(x - px, y - py)
        dn_arc = np.where((th >= 0) & (th <= np.pi / 2), r - Rb,
                          np.hypot(x - px, y - py) * np.sign(
                              (x - px) * np.cos(thc) + (y - py) * np.sin(thc)))
        y_arc = py

        # --- tramo superior: y = yt, x en [0, xc]
        if xc > 1e-9:
            xt = np.clip(x, 0.0, xc)
            dist_top = np.hypot(x - xt, y - YT)
            dn_top = y - YT
        else:
            dist_top = np.full_like(x, np.inf)
            dn_top = np.zeros_like(x)

        dists = np.stack([dist_leg, dist_arc, dist_top])
        idx = np.argmin(dists, axis=0)
        dn = np.choose(idx, [dn_leg, dn_arc, dn_top])
        y_axis = np.choose(idx, [yl, y_arc, np.full_like(y, YT)])
        dist = np.min(dists, axis=0)
        return dn, y_axis, dist

    # ------------------------------------------------------------------
    def phi(self, x, y, z):
        """Función implícita φ (<= 0 dentro del sólido) y datos auxiliares."""
        dn, y_axis, dist = self.project(x, y)
        A, delta = self.section(y_axis)
        # Fuera del rango de los tramos (bajo Y0) el punto queda fuera.
        phi = ((dn - delta) / A) ** 2 + (np.asarray(z) / self.B) ** 2 - 1.0
        phi = np.where(np.isfinite(dist), phi, np.inf)
        phi = np.where(np.asarray(y) < Y0, np.inf, phi)
        return phi, dn, delta, A

    def depth(self, x, y, z):
        """Profundidad aproximada bajo la superficie, d ≈ (1 - sqrt(φ+1)) min(A, B)."""
        phi, _, _, A = self.phi(x, y, z)
        return (1.0 - np.sqrt(np.clip(phi + 1.0, 0.0, None))) * np.minimum(A, self.B)


# ----------------------------------------------------------------------
# Catálogo de geometrías (secciones 2.1 y 9.2)
# ----------------------------------------------------------------------
GEOMETRIES = {
    "G1": Geometry("G1", 6.0, label="R6 (v5 actual)"),
    "G2": Geometry("G2", 9.0, label="R9"),
    "G3": Geometry("G3", 12.0, label="R12"),
    "G4": Geometry("G4", 15.0, label="R15 (v8)"),
    "G5": Geometry("G5", 18.0, label="R18"),
    "G6": Geometry("G6", 22.5, label="Arco R22,5 (v7)"),
    "G7": Geometry("G7", 15.0, A_top=7.0, k_delta=-1.0, label="R15 + 14 mm adentro"),
    "G8": Geometry("G8", 15.0, A_top=8.0, k_delta=-1.0, label="R15 + 16 mm adentro"),
    "G9": Geometry("G9", 22.5, A_top=7.0, k_delta=-1.0, label="Arco + 14 mm adentro"),
    "G10": Geometry("G10", 22.5, A_top=8.0, k_delta=-1.0, label="Arco + 16 mm adentro"),
    # Variantes engrosadas hacia afuera (series A y B). Cambian el contorno.
    "G4o14": Geometry("G4o14", 15.0, A_top=7.0, k_delta=+1.0, label="R15 + 14 mm afuera"),
    "G4o16": Geometry("G4o16", 15.0, A_top=8.0, k_delta=+1.0, label="R15 + 16 mm afuera"),
    "G6o14": Geometry("G6o14", 22.5, A_top=7.0, k_delta=+1.0, label="Arco + 14 mm afuera"),
    "G6o16": Geometry("G6o16", 22.5, A_top=8.0, k_delta=+1.0, label="Arco + 16 mm afuera"),
}


def window_area(geom: Geometry, res: float = 0.05) -> float:
    """Área libre de la ventana interior en el plano medio, en la mitad x >= 0.

    Es el área entre la pata, el tramo superior y el eje de simetría, desde
    y = Y0 hacia arriba, que no está ocupada por material (convención del
    informe original, que reporta 792 mm² para la v5).
    """
    xs = np.arange(res / 2, A_LEG, res)
    ys = np.arange(Y0 + res / 2, YT + 10, res)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    phi, dn, delta, _ = geom.phi(X, Y, np.zeros_like(X))
    solid = phi <= 0
    # Interior: lado interior de la U (d_n - δ < 0) y no sólido. Para cada
    # columna x, se toma el aire bajo la primera celda sólida desde abajo
    # dentro de la U.
    inside = (~solid) & (dn - delta < 0)
    # Bajo el tramo superior: debe existir material por encima en la columna.
    above = np.flip(np.cumsum(np.flip(solid, axis=1), axis=1), axis=1) > 0
    return float(np.sum(inside & above) * res * res)
