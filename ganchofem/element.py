"""Elemento hexaédrico trilineal de 8 nodos sobre un cubo de lado h (secciones 3.4–3.5)."""
from __future__ import annotations

import numpy as np

# Nodos en coordenadas naturales, orden estándar (cara inferior antihoraria, luego superior)
NAT = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
], dtype=float)


def lame(E: float, nu: float):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return lam, mu


def D_matrix(E: float, nu: float) -> np.ndarray:
    """Matriz constitutiva isótropa en notación de Voigt [xx, yy, zz, xy, yz, xz]."""
    lam, mu = lame(E, nu)
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[np.arange(3), np.arange(3)] = lam + 2 * mu
    D[3, 3] = D[4, 4] = D[5, 5] = mu
    return D


def dN_dxi(xi, eta, zeta) -> np.ndarray:
    """Derivadas de las funciones de forma respecto de (ξ, η, ζ); forma (8, 3)."""
    g = np.empty((8, 3))
    for a, (xa, ya, za) in enumerate(NAT):
        g[a, 0] = xa * (1 + eta * ya) * (1 + zeta * za) / 8
        g[a, 1] = ya * (1 + xi * xa) * (1 + zeta * za) / 8
        g[a, 2] = za * (1 + xi * xa) * (1 + eta * ya) / 8
    return g


def B_matrix(xi, eta, zeta, h: float) -> np.ndarray:
    """Matriz deformación–desplazamiento (6 × 24) del cubo de lado h."""
    dN = dN_dxi(xi, eta, zeta) * (2.0 / h)  # J = h/2 I
    B = np.zeros((6, 24))
    for a in range(8):
        Nx, Ny, Nz = dN[a]
        c = 3 * a
        B[0, c] = Nx
        B[1, c + 1] = Ny
        B[2, c + 2] = Nz
        B[3, c], B[3, c + 1] = Ny, Nx
        B[4, c + 1], B[4, c + 2] = Nz, Ny
        B[5, c], B[5, c + 2] = Nz, Nx
    return B


def unit_stiffness(h: float, nu: float) -> np.ndarray:
    """K̂_e(ν) con E = 1, integrada con Gauss 2 × 2 × 2 (exacta para el cubo)."""
    D = D_matrix(1.0, nu)
    g = 1.0 / np.sqrt(3.0)
    detJ = (h / 2) ** 3
    K = np.zeros((24, 24))
    for xi in (-g, g):
        for eta in (-g, g):
            for zeta in (-g, g):
                B = B_matrix(xi, eta, zeta, h)
                K += B.T @ D @ B * detJ
    return K
