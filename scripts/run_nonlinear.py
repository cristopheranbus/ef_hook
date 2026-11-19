"""Análisis no lineal (plasticidad + grandes desplazamientos) del caso 3.

Uso:
    python scripts/run_nonlinear.py [--jobs 3] [--h 1.0]

Para cada material y geometría (v5 actual con 2 paredes; v7 y G10 con 5
paredes) traza la curva fuerza–apertura hasta la rotura estimada y guarda
results/nonlinear/*.json. Incluye un control de malla (h = 0,75 mm) y uno de
endurecimiento (H = 0) para PLA en la v7.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ganchofem.material import FILAMENTS  # noqa: E402
from ganchofem.nonlinear import NLMaterial, TLPlasticModel  # noqa: E402

OUT = ROOT / "results" / "nonlinear"
H_FRAC = 0.01
CASES = [("G1", "2p15"), ("G6", "5p25"), ("G10", "5p25")]


def nl_material(key, h_frac=H_FRAC):
    f = FILAMENTS[key]
    return NLMaterial(key, f.E_t, 0.35, f.st_xy, h_frac * f.E_t, f.eb_xy / 100)


def job(tag, key, geom, prof, h, h_frac):
    out = OUT / f"{tag}.json"
    if out.exists():
        return tag, "existente"
    m = TLPlasticModel(geom, prof, nl_material(key, h_frac), h=h)
    r = m.run(open_max=40.0, n_steps=40)
    d = asdict(r)
    d.update(tag=tag, h_frac=h_frac)
    out.write_text(json.dumps(d, indent=1))
    return tag, f"F_offset={r.F_offset} F_rupt={r.F_rupture} t={r.seconds:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--h", type=float, default=1.0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(f"{g}-{p}-{k}", k, g, p, a.h, H_FRAC) for g, p in CASES for k in FILAMENTS]
    jobs += [("G6-5p25-PLA-h0.75", "PLA", "G6", "5p25", 0.75, H_FRAC),
             ("G6-5p25-PLA-H0", "PLA", "G6", "5p25", a.h, 0.0)]
    with ProcessPoolExecutor(a.jobs) as ex:
        for f in as_completed([ex.submit(job, *j) for j in jobs]):
            print(*f.result(), flush=True)


if __name__ == "__main__":
    main()
