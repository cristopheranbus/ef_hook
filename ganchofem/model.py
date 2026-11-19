"""Simulación completa del gancho: malla → material → cargas → solución → tensiones."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .element import B_matrix, D_matrix, unit_stiffness
from .geometry import A_LEG, GEOMETRIES, YT, Y_TOPE, Geometry
from .material import E_REF, NU_REF, PROFILES, PrintProfile, element_modulus
from .mesh import VoxelMesh, build_mesh
from .solver import assemble, element_dofs, solve

F_REF = 100.0  # fuerza de referencia por pata [N]

CASES = {
    1: "Tiro del tramo superior",
    2: "Apertura a media altura",
    3: "Apertura en los extremos (diseño)",
}


@dataclass
class SimSpec:
    sim_id: str
    geometry: str
    case: int
    profile: str
    h: float = 0.5
    E: float = E_REF
    nu: float = NU_REF
    k: float | None = None   # None = valor adoptado del perfil
    series: str = ""
    note: str = ""


@dataclass
class SimResult:
    spec: dict
    n_elements: int
    n_nodes: int
    n_dof_free: int
    nnz: int
    solver: str
    iterations: int
    residual: float
    t_solve: float
    t_total: float
    volume_cm3: float
    s1_max: float
    s1_p998: float
    svm_max: float
    s1_max_loc: list
    opening: float
    n_load_nodes: int
    extra: dict = field(default_factory=dict)


# ----------------------------------------------------------------------
def boundary_conditions(geom: Geometry, mesh: VoxelMesh, case: int, F: float):
    """Devuelve (dofs fijos, vector de fuerzas, nodos cargados por lado)."""
    X, Y, Z = mesh.nodes.T
    surf = mesh.surface_nodes
    dn, y_axis, _ = geom.project(X, Y)
    A, delta = geom.section(y_axis)
    outer = (dn - delta) > 0   # mitad exterior de la sección
    ndof = 3 * mesh.nn
    Fv = np.zeros(ndof)
    tol = 1e-9

    if case in (1, 2):
        fixed_nodes = np.where(Y <= Y_TOPE + tol)[0]
    else:  # caso 3: plano de simetría x = 0, tres componentes
        fixed_nodes = np.where(np.abs(X) <= tol)[0]
    fixed = (3 * fixed_nodes[:, None] + np.arange(3)).ravel()

    groups = []
    if case == 1:
        sel = surf & (np.abs(X) <= 10 + tol) & (~outer) & (Y > 0)
        groups.append((np.where(sel)[0], np.array([0.0, 1.0, 0.0]), 2 * F))
    elif case == 2:
        for sgn in (+1, -1):
            sel = surf & outer & (sgn * X > A_LEG - 1.0) & (Y >= -8 - tol) & (Y <= 2 + tol)
            groups.append((np.where(sel)[0], np.array([sgn, 0.0, 0.0]), F))
    elif case == 3:
        for sgn in (+1, -1):
            sel = surf & outer & (sgn * X > 0) & (Y <= Y_TOPE + tol)
            groups.append((np.where(sel)[0], np.array([sgn, 0.0, 0.0]), F))
    for nodes, d, Ftot in groups:
        f = Ftot / len(nodes)
        for c in range(3):
            Fv[3 * nodes + c] += f * d[c]
    return fixed, Fv, groups


def element_stress(mesh: VoxelMesh, U: np.ndarray, E: np.ndarray, nu: float):
    """Tensión en el centroide de cada elemento (ξ = 0); Voigt (ne, 6)."""
    B0 = B_matrix(0.0, 0.0, 0.0, mesh.h)
    Dh = D_matrix(1.0, nu)
    ue = U[element_dofs(mesh.conn)]           # (ne, 24)
    eps = ue @ B0.T                            # (ne, 6)
    return (eps @ Dh.T) * E[:, None]


def principal_and_vm(sig: np.ndarray):
    sx, sy, sz, txy, tyz, txz = sig.T
    T = np.empty((len(sig), 3, 3))
    T[:, 0, 0], T[:, 1, 1], T[:, 2, 2] = sx, sy, sz
    T[:, 0, 1] = T[:, 1, 0] = txy
    T[:, 1, 2] = T[:, 2, 1] = tyz
    T[:, 0, 2] = T[:, 2, 0] = txz
    ev = np.linalg.eigvalsh(T)
    s1 = ev[:, -1]
    vm = np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2)
                 + 3 * (txy ** 2 + tyz ** 2 + txz ** 2))
    return s1, vm


def run(spec: SimSpec, keep_fields: bool = False):
    t0 = time.perf_counter()
    geom = GEOMETRIES[spec.geometry]
    prof: PrintProfile = PROFILES[spec.profile]
    if spec.k is not None:
        prof = prof.with_k(spec.k)

    mesh = build_mesh(geom, spec.h)
    depth = geom.depth(*mesh.centroids.T)
    E, shell = element_modulus(depth, prof, spec.E)

    ndof = 3 * mesh.nn
    K = assemble(mesh.conn, E, unit_stiffness(spec.h, spec.nu), ndof)
    fixed, Fv, groups = boundary_conditions(geom, mesh, spec.case, F_REF)
    free = np.setdiff1d(np.arange(ndof), fixed)
    Kff = K[free][:, free].tocsr()
    del K
    Uf, info = solve(Kff, Fv[free])
    U = np.zeros(ndof)
    U[free] = Uf

    sig = element_stress(mesh, U, E, spec.nu)
    s1, vm = principal_and_vm(sig)

    # Indicadores sobre la cáscara; en los casos 1 y 2 se excluye la zona
    # empotrada (y <= -22,5 mm), donde el apoyo rígido es una idealización.
    valid = shell.copy()
    if spec.case in (1, 2):
        valid &= mesh.centroids[:, 1] > Y_TOPE
    s1v, vmv = s1[valid], vm[valid]
    imax = np.argmax(s1v)
    loc = mesh.centroids[valid][imax]

    # Apertura: desplazamiento relativo medio de las caras extremas de las
    # patas (y = y0). En el caso 1, desplazamiento medio en y de la zona cargada.
    Ux = U.reshape(-1, 3)
    X, Y = mesh.nodes[:, 0], mesh.nodes[:, 1]
    if spec.case == 1:
        opening = float(Ux[groups[0][0], 1].mean())
    else:
        tip = np.abs(Y - Y.min()) < 1e-9
        opening = float(Ux[tip & (X > 0), 0].mean() - Ux[tip & (X < 0), 0].mean())

    res = SimResult(
        spec=asdict(spec),
        n_elements=mesh.ne, n_nodes=mesh.nn, n_dof_free=len(free), nnz=int(Kff.nnz),
        solver=info.method, iterations=info.iterations, residual=info.residual,
        t_solve=info.seconds, t_total=time.perf_counter() - t0,
        volume_cm3=mesh.ne * spec.h ** 3 / 1000.0,
        s1_max=float(s1v.max()), s1_p998=float(np.percentile(s1v, 99.8)),
        svm_max=float(vmv.max()), s1_max_loc=[float(v) for v in loc],
        opening=opening, n_load_nodes=int(sum(len(g[0]) for g in groups)),
    )
    # Cociente σ_VM/σ1 en el punto de σ1 máximo
    res.extra["vm_over_s1_at_peak"] = float(vmv[imax] / s1v[imax])
    if not keep_fields:
        return res, None

    # Campos del plano medio: capa de elementos con centroide en z = +h/2
    lay = mesh.ijk[:, 2] == 0
    conn_l = mesh.conn[lay][:, :4]     # cara ζ = -1, z = 0
    fields = dict(
        h=spec.h,
        ij=mesh.ijk[lay][:, :2],
        s1=s1[lay], vm=vm[lay], shell=shell[lay], E=E[lay],
        corners=mesh.nodes[conn_l][:, :, :2],
        ucorners=Ux[conn_l][:, :, :2],
    )
    return res, fields
