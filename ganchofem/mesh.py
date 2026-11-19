"""Malla de vóxeles alineada con los planos x = 0, y = 0 y z = 0 (sección 3.4).

Un cubo de lado h es elemento si su centroide cumple φ(x_c) <= 0. La grilla
tiene nodos en múltiplos enteros de h, de modo que siempre existe un plano de
nodos en x = 0 (necesario para el apoyo de simetría del caso 3; ver la nota de
control de calidad de la sección 7 del informe).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .element import NAT
from .geometry import Geometry


@dataclass
class VoxelMesh:
    h: float
    ijk: np.ndarray        # (ne, 3) índices enteros del vóxel (esquina inferior)
    conn: np.ndarray       # (ne, 8) nodos de cada elemento (numeración compacta)
    nodes: np.ndarray      # (nn, 3) coordenadas de los nodos [mm]
    centroids: np.ndarray  # (ne, 3)
    node_count: np.ndarray  # (nn,) número de elementos que comparte cada nodo

    @property
    def ne(self) -> int:
        return len(self.conn)

    @property
    def nn(self) -> int:
        return len(self.nodes)

    @property
    def surface_nodes(self) -> np.ndarray:
        """Nodos que pertenecen a menos de 8 elementos."""
        return self.node_count < 8


def build_mesh(geom: Geometry, h: float) -> VoxelMesh:
    x0, x1, y0, y1, z0, z1 = geom.bbox
    i0, i1 = int(np.floor(x0 / h)) - 1, int(np.ceil(x1 / h)) + 1
    j0, j1 = int(np.floor(y0 / h)) - 1, int(np.ceil(y1 / h)) + 1
    k0, k1 = int(np.floor(z0 / h)) - 1, int(np.ceil(z1 / h)) + 1
    I = np.arange(i0, i1)
    J = np.arange(j0, j1)
    K = np.arange(k0, k1)

    # Evaluación de φ por capas en z para limitar memoria
    elems = []
    XI, YJ = np.meshgrid(I, J, indexing="ij")
    xc, yc = (XI + 0.5) * h, (YJ + 0.5) * h
    for k in K:
        zc = np.full_like(xc, (k + 0.5) * h)
        phi = geom.phi(xc, yc, zc)[0]
        m = phi <= 0
        if m.any():
            elems.append(np.column_stack([XI[m], YJ[m], np.full(m.sum(), k)]))
    ijk = np.concatenate(elems).astype(np.int64)

    # Nodos de cada elemento en índices de grilla
    offs = ((NAT + 1) / 2).astype(np.int64)  # (8, 3) en {0, 1}
    nijk = ijk[:, None, :] + offs[None, :, :]  # (ne, 8, 3)
    ni, nj, nk = I.size + 1, J.size + 1, K.size + 1
    lin = ((nijk[..., 0] - i0) * nj + (nijk[..., 1] - j0)) * nk + (nijk[..., 2] - k0)
    uniq, inv, counts = np.unique(lin.ravel(), return_inverse=True, return_counts=True)
    conn = inv.reshape(-1, 8).astype(np.int64)
    gi = uniq // (nj * nk) + i0
    gj = (uniq // nk) % nj + j0
    gk = uniq % nk + k0
    nodes = np.column_stack([gi, gj, gk]).astype(float) * h
    centroids = (ijk + 0.5) * h
    return VoxelMesh(h=h, ijk=ijk, conn=conn, nodes=nodes, centroids=centroids,
                     node_count=counts)
