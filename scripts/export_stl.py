"""Exporta las geometrías del gancho a STL imprimibles (modelos_stl/).

Uso:
    python scripts/export_stl.py [--res 0.15] [--all]

Cada STL se obtiene por marching cubes sobre la función implícita φ de
ganchofem.geometry (la misma que usa el MEF), en mm, con la pieza acostada en
el plano XY (orientación de impresión del estudio: la U en el plano de capas)
y apoyada en z = 0. --all incluye las variantes engrosadas hacia afuera.
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import trimesh
from skimage.measure import marching_cubes

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ganchofem.geometry import GEOMETRIES, R_SEC, Y0  # noqa: E402

OUT = ROOT / "modelos_stl"


def implicit(geom, X, Y, Z):
    """φ con el corte plano de las puntas en y = y0 como una superficie más."""
    phi = geom.phi(X, np.maximum(Y, Y0), Z)[0]
    return np.maximum(phi, (Y0 - Y) / R_SEC)


def write_stl(path, verts, faces, name):
    tri = verts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    with open(path, "wb") as f:
        f.write(name.encode()[:80].ljust(80, b" "))
        f.write(struct.pack("<I", len(faces)))
        data = np.zeros(len(faces), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
        data["n"], data["v"] = n, tri
        f.write(data.tobytes())


def export(name, res):
    g = GEOMETRIES[name]
    x0, x1, y0, y1, z0, z1 = g.bbox
    pad = 2 * res
    xs = np.arange(x0 - pad, x1 + pad, res)
    ys = np.arange(y0 - pad, y1 + pad, res)
    zs = np.arange(z0 - pad, z1 + pad, res)
    vol = np.empty((xs.size, ys.size, zs.size), dtype=np.float32)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    for k, z in enumerate(zs):
        vol[:, :, k] = implicit(g, X, Y, np.full_like(X, z))
    verts, faces, _, _ = marching_cubes(vol, level=0.0, spacing=(res, res, res))
    verts += np.array([xs[0], ys[0], zs[0]])
    verts[:, 2] -= verts[:, 2].min()  # apoyada en la cama
    # Limpieza: marching cubes deja triángulos de área nula donde un vértice
    # cae sobre un nodo de la grilla; se eliminan para que la malla sea cerrada.
    m = trimesh.Trimesh(verts, faces, process=True)
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    # En el canto vivo de la v5 quedan astillas de 2 triángulos: se conserva
    # solo el cuerpo principal.
    m = max(m.split(only_watertight=False), key=lambda b: len(b.faces))
    m.fix_normals()
    assert m.is_watertight and m.euler_number == 2, f"{name}: malla no cerrada"
    path = OUT / f"gancho_{name}.stl"
    write_stl(path, m.vertices.astype(np.float32), m.faces, f"gancho {name} {g.label}")
    ext = m.bounds[1] - m.bounds[0]
    return path, len(m.faces), m.volume / 1000, ext


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=float, default=0.25)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    names = [n for n in GEOMETRIES if a.all or "o" not in n]
    lines = ["# Modelos imprimibles", "", "Generados con `python scripts/export_stl.py` desde la misma función implícita del MEF.",
             "Unidades: mm. Orientación: la U acostada en el plano XY (como en el estudio), apoyada en z = 0.",
             "Todas las mallas son cerradas (watertight).", "",
             "| Archivo | Geometría | Triángulos | Volumen [cm³] | Tamaño X × Y × Z [mm] |", "|---|---|---|---|---|"]
    for n in names:
        p, nf, v, e = export(n, a.res)
        print(f"{p.name}: {nf} triángulos, {v:.2f} cm³, {e[0]:.1f} × {e[1]:.1f} × {e[2]:.1f} mm")
        lines.append(f"| {p.name} | {GEOMETRIES[n].label} | {nf} | {v:.2f} | {e[0]:.1f} × {e[1]:.1f} × {e[2]:.1f} |")
    (OUT / "indice.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
