"""Definición de las 63 simulaciones (series A–D, sección 7 del informe)."""
from __future__ import annotations

from .model import SimSpec

BASE_GEOMS = ["G1", "G2", "G3", "G4", "G5", "G6"]
INNER_GEOMS = ["G7", "G8", "G9", "G10"]
OUTER_GEOMS = ["G4o14", "G4o16", "G6o14", "G6o16"]


def _series_ab(series: str, case: int):
    specs = []
    # G1–G6 con el perfil recomendado (5 paredes, 25 %)
    for g in BASE_GEOMS:
        specs.append(SimSpec(f"{series}-{g}-5p25", g, case, "5p25", series=series))
    # v5, v8 y v7 macizas y con el perfil actual
    for g in ("G1", "G4", "G6"):
        for p in ("macizo", "2p15"):
            specs.append(SimSpec(f"{series}-{g}-{p}", g, case, p, series=series))
    # Variantes engrosadas hacia afuera
    for g in OUTER_GEOMS:
        specs.append(SimSpec(f"{series}-{g}-5p25", g, case, "5p25", series=series))
    return specs


def _series_c():
    specs = []
    for g in BASE_GEOMS + INNER_GEOMS:
        specs.append(SimSpec(f"C-{g}-5p25", g, 3, "5p25", series="C"))
    for g in ("G1", "G4", "G6"):
        for p in ("macizo", "2p15"):
            specs.append(SimSpec(f"C-{g}-{p}", g, 3, p, series="C"))
    specs.append(SimSpec("C-G10-macizo", "G10", 3, "macizo", series="C"))
    return specs


def _series_d():
    s = []
    # Convergencia de malla (macizo)
    for h in (1.0, 0.75, 0.4):
        s.append(SimSpec(f"D-G6-macizo-h{h}", "G6", 3, "macizo", h=h, series="D", note="convergencia"))
    for h in (1.0, 0.75):
        s.append(SimSpec(f"D-G1-macizo-h{h}", "G1", 3, "macizo", h=h, series="D", note="convergencia"))
    s.append(SimSpec("D-G6-5p25-h0.75", "G6", 3, "5p25", h=0.75, series="D", note="convergencia"))
    s.append(SimSpec("D-G1-5p25-h0.75", "G1", 3, "5p25", h=0.75, series="D", note="convergencia"))
    # Independencia respecto de E (módulo de flexión del ABS)
    s.append(SimSpec("D-G6-5p25-E1880", "G6", 3, "5p25", E=1880.0, series="D", note="E"))
    # Coeficiente de Poisson
    for nu in (0.30, 0.40):
        s.append(SimSpec(f"D-G6-5p25-nu{nu:.2f}", "G6", 3, "5p25", nu=nu, series="D", note="nu"))
    # Módulo del relleno (Gibson–Ashby)
    s.append(SimSpec("D-G6-5p25-k0.0625", "G6", 3, "5p25", k=0.0625, series="D", note="k"))
    s.append(SimSpec("D-G1-2p15-k0.15", "G1", 3, "2p15", k=0.15, series="D", note="k"))
    s.append(SimSpec("D-G1-2p15-k0.0225", "G1", 3, "2p15", k=0.0225, series="D", note="k"))
    s.append(SimSpec("D-G10-2p15", "G10", 3, "2p15", series="D", note="perfil"))
    return s


def all_specs():
    return _series_ab("A", 1) + _series_ab("B", 2) + _series_c() + _series_d()


if __name__ == "__main__":
    sp = all_specs()
    from collections import Counter
    print(len(sp), Counter(s.series for s in sp))
