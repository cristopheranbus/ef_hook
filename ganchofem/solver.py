"""Ensamblaje disperso y solución del sistema (secciones 3.6 y 3.8)."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

LU_LIMIT = 60_000  # bajo este número de incógnitas se usa LU dispersa


def element_dofs(conn: np.ndarray) -> np.ndarray:
    """Grados de libertad (ne, 24) en orden [u0x, u0y, u0z, u1x, ...]."""
    return (3 * conn[:, :, None] + np.arange(3)[None, None, :]).reshape(len(conn), 24)


def assemble(conn: np.ndarray, E: np.ndarray, Ke_unit: np.ndarray, ndof: int) -> sp.csr_matrix:
    """K = Σ_e L_eᵀ (E_e K̂_e) L_e, en COO por bloques y convertido a CSR."""
    dofs = element_dofs(conn).astype(np.int32)
    ne = len(conn)
    K = None
    chunk = 40_000
    for s in range(0, ne, chunk):
        d = dofs[s:s + chunk]
        rows = np.repeat(d, 24, axis=1).ravel()
        cols = np.tile(d, (1, 24)).ravel()
        vals = (E[s:s + chunk, None, None] * Ke_unit[None]).ravel()
        Kc = sp.coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()
        K = Kc if K is None else K + Kc
    K.sum_duplicates()
    return K


@dataclass
class SolveInfo:
    method: str
    n: int
    nnz: int
    iterations: int
    residual: float
    seconds: float


def pcg_jacobi(K: sp.csr_matrix, F: np.ndarray, rtol: float = 1e-9, maxiter: int = 20_000):
    """Gradiente conjugado precondicionado con Jacobi, M = diag(K)."""
    Minv = 1.0 / K.diagonal()
    U = np.zeros_like(F)
    r = F.copy()
    z = Minv * r
    p = z.copy()
    rz = r @ z
    nF = np.linalg.norm(F)
    it = 0
    while it < maxiter:
        Kp = K @ p
        alpha = rz / (p @ Kp)
        U += alpha * p
        r -= alpha * Kp
        it += 1
        if np.linalg.norm(r) <= rtol * nF:
            break
        z = Minv * r
        rz_new = r @ z
        p = z + (rz_new / rz) * p
        rz = rz_new
    return U, it


def solve(K: sp.csr_matrix, F: np.ndarray, rtol: float = 1e-9):
    t0 = time.perf_counter()
    n = K.shape[0]
    if n < LU_LIMIT:
        U = spla.spsolve(K.tocsc(), F)
        method, it = "LU disperso", 0
    else:
        U, it = pcg_jacobi(K, F, rtol)
        method = "PCG-Jacobi"
    res = float(np.linalg.norm(F - K @ U) / np.linalg.norm(F))
    info = SolveInfo(method, n, K.nnz, it, res, time.perf_counter() - t0)
    return U, info
