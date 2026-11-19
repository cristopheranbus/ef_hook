import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ganchofem.analytic import crown_moment, winkler_circular
from ganchofem.element import B_matrix, D_matrix, unit_stiffness
from ganchofem.geometry import GEOMETRIES
from ganchofem.mesh import build_mesh


def test_element_rigid_body_modes():
    K = unit_stiffness(0.5, 0.35)
    assert np.allclose(K, K.T)
    ev = np.linalg.eigvalsh(K)
    assert np.sum(np.abs(ev) < 1e-10 * ev.max()) == 6  # sin modos espurios
    assert ev.min() > -1e-10


def test_patch_uniform_strain():
    """Un campo lineal de desplazamientos da deformación exacta en todo el elemento."""
    h = 0.5
    from ganchofem.element import NAT
    X = (NAT + 1) / 2 * h
    eps0 = np.array([1e-3, -2e-4, 5e-4, 3e-4, -1e-4, 2e-4])
    G = np.array([[eps0[0], eps0[3] / 2, eps0[5] / 2],
                  [eps0[3] / 2, eps0[1], eps0[4] / 2],
                  [eps0[5] / 2, eps0[4] / 2, eps0[2]]])
    u = (X @ G.T).ravel()
    for p in [(0, 0, 0), (0.3, -0.7, 0.9), (-1, 1, -1)]:
        assert np.allclose(B_matrix(*p, h) @ u, eps0)


@pytest.mark.parametrize("name,ne", [("G1", 132660), ("G6", 120460)])
def test_mesh_matches_paper(name, ne):
    """El mallado reproduce el número de elementos informado (h = 0,5 mm)."""
    assert build_mesh(GEOMETRIES[name], 0.5).ne == ne


def test_winkler_numbers():
    arm, M = crown_moment(100.0)
    assert arm == pytest.approx(50.25)
    w = winkler_circular(M, 100.0, 22.5, 6.0)
    assert w["rn"] == pytest.approx(22.093, abs=1e-3)
    assert w["sigma_i"] == pytest.approx(37.9, abs=0.05)
    assert w["Ki"] == pytest.approx(1.25, abs=0.01)


def test_stress_independent_of_E():
    """Teorema de escalado (sección 4.2) en un modelo grueso."""
    from ganchofem.model import SimSpec, run
    r1, _ = run(SimSpec("t1", "G6", 3, "5p25", h=1.0, E=3500.0))
    r2, _ = run(SimSpec("t2", "G6", 3, "5p25", h=1.0, E=1880.0))
    assert r1.s1_max == pytest.approx(r2.s1_max, rel=1e-8)
    assert r2.opening / r1.opening == pytest.approx(3500 / 1880, rel=1e-8)


def test_findley_fit_recovers_parameters():
    from ganchofem.creep import fit_findley
    t = np.array([1, 3, 10, 24, 72, 168.0])
    y = 2.0 * (1 + (t / 500.0) ** 0.25)
    d0, tau, n = fit_findley(t, y, op0=2.0)
    assert tau == pytest.approx(500, rel=1e-3) and n == pytest.approx(0.25, rel=1e-3)


def test_offset_force():
    from ganchofem.nonlinear import offset_force
    o = np.array([1, 2, 3, 4, 5.0])
    f = np.array([10, 20, 28, 30, 31.0])
    k0, F = offset_force(o, f, 1.0)
    assert k0 == 10 and F == pytest.approx(30.0)
