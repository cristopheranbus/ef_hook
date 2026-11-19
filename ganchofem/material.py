"""Modelo homogeneizado cáscara–núcleo de la pieza impresa (sección 4) y datos de fichas técnicas (sección 6)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LINE_WIDTH = 0.42  # ancho de línea [mm]


@dataclass(frozen=True)
class PrintProfile:
    name: str
    t_s: float | None   # espesor de cáscara [mm]; None = macizo
    k: float            # módulo relativo del relleno E*/E_s
    rho_rel: float      # densidad relativa del relleno
    label: str = ""

    def with_k(self, k: float) -> "PrintProfile":
        return PrintProfile(self.name, self.t_s, k, self.rho_rel, self.label)


PROFILES = {
    "macizo": PrintProfile("macizo", None, 1.0, 1.0, "Macizo"),
    "2p15": PrintProfile("2p15", 1.0, 0.08, 0.15, "2 paredes, 15 %"),
    "5p25": PrintProfile("5p25", 2.2, 0.25, 0.25, "5 paredes, 25 % gyroid"),
}


def gibson_ashby(rho_rel: float, n: float, C: float = 1.0) -> float:
    """E*/E_s = C (ρ*/ρ_s)^n."""
    return C * rho_rel ** n


def element_modulus(depth: np.ndarray, profile: PrintProfile, E_s: float):
    """Módulo por elemento y máscara de cáscara."""
    if profile.t_s is None:
        shell = np.ones_like(depth, dtype=bool)
    else:
        shell = depth <= profile.t_s
    E = np.where(shell, E_s, profile.k * E_s)
    return E, shell


# ----------------------------------------------------------------------
# Fichas técnicas (Bambu Lab, probetas 100 % macizas, recocidas y secas)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Filament:
    name: str
    E_t: float      # módulo de tracción XY [MPa]
    E_f: float      # módulo de flexión XY [MPa]
    st_xy: float    # resistencia a tracción XY [MPa]
    st_z: float     # resistencia a tracción Z [MPa]
    eb_xy: float    # alargamiento a la rotura XY [%]
    impact: float   # impacto con entalla XY [kJ/m²]
    hdt: float      # HDT a 1,8 MPa [°C]
    rho: float      # densidad [g/cm³]
    ref: str


FILAMENTS = {
    "PLA": Filament("PLA Basic", 2680, 2750, 39, 35, 12.2, 7.9, 54, 1.24, "[16]"),
    "PETG": Filament("PETG HF", 1810, 2050, 34, 23, 8.6, 6.2, 62, 1.28, "[17]"),
    "ASA": Filament("ASA", 2450, 1920, 37, 31, 9.2, 19.6, 92, 1.05, "[18]"),
    "ABS": Filament("ABS", 3800, 1880, 41, 32, 10.5, 21.5, 84, 1.05, "[19]"),
    "PC": Filament("PC", 2110, 2310, 62, 56, 3.8, 7.5, 117, 1.20, "[20]"),
    "PAHT-CF": Filament("PAHT-CF", 3860, 4120, 88, 64, 10.4, 22.8, 170, 1.06, "[21]"),
}

E_REF = 3500.0  # módulo de referencia de las simulaciones [MPa]
NU_REF = 0.35
