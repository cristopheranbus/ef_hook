"""Corre las 63 simulaciones (series A–D) y guarda un JSON por simulación.

Uso:
    python scripts/run_all.py [--jobs 3] [--only C] [--force]

Es reanudable: salta las simulaciones cuyo JSON ya existe. Para las
simulaciones del caso 3 con h = 0,5 mm guarda además los campos del plano
medio (results/fields/*.npz), que usan las figuras.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ganchofem.model import run  # noqa: E402
from ganchofem.series import all_specs  # noqa: E402

SIMS = ROOT / "results" / "sims"
FIELDS = ROOT / "results" / "fields"


def _work(spec):
    keep = spec.case == 3 and spec.h == 0.5 and spec.E == 3500.0 and spec.nu == 0.35 and spec.k is None
    res, fields = run(spec, keep_fields=keep)
    (SIMS / f"{spec.sim_id}.json").write_text(json.dumps(asdict(res), indent=1, ensure_ascii=False))
    if fields is not None:
        np.savez_compressed(FIELDS / f"{spec.sim_id}.npz", **fields)
    return spec.sim_id, res.s1_max, res.opening, res.iterations, res.t_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--only", default=None, help="serie (A, B, C o D)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    SIMS.mkdir(parents=True, exist_ok=True)
    FIELDS.mkdir(parents=True, exist_ok=True)
    specs = [s for s in all_specs() if a.only is None or s.series == a.only]
    if not a.force:
        specs = [s for s in specs if not (SIMS / f"{s.sim_id}.json").exists()]
    # Las más grandes primero para equilibrar la carga
    specs.sort(key=lambda s: s.h)
    print(f"{len(specs)} simulaciones por correr", flush=True)
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = [ex.submit(_work, s) for s in specs]
        for f in as_completed(futs):
            sid, s1, op, it, t = f.result()
            print(f"{sid:28s} σ1max = {s1:8.2f} MPa  apertura = {op:7.3f} mm  it = {it:5d}  {t:6.1f} s", flush=True)


if __name__ == "__main__":
    main()
