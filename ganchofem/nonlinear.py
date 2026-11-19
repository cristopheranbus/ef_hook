"""Análisis no lineal del caso 3: plasticidad J2 + grandes desplazamientos.

Responde a la pregunta práctica "¿con qué carga el gancho cede y se estira?".

Formulación
-----------
* Lagrangiana total: F = I + ∇₀u, deformación de Green–Lagrange
  E = ½(Fᵀ F − I) y segundo tensor de Piola–Kirchhoff S. Incluye grandes
  desplazamientos y rotaciones (deformaciones moderadas).
* Plasticidad J2 (von Mises) con endurecimiento isótropo lineal, aditiva sobre
  E (modelo de Kirchhoff–Saint Venant elastoplástico), con algoritmo de
  retorno radial y tangente consistente.
* Hexaedro de 8 nodos con integración 2 × 2 × 2 y el método Ē (análogo al
  B-bar): la parte volumétrica de E se reemplaza por su media en el elemento,
  para evitar el bloqueo volumétrico del flujo plástico isocórico.
* Cuarto de modelo: x ≥ 0 (simetría en x = 0) y z ≥ 0 (simetría en z = 0),
  más un nodo de la corona con u_y = 0.
* Control por desplazamiento: se impone u_x en la cara exterior de la punta
  (y ≤ −22,5 mm), como un tope rígido que empuja la pata hacia afuera. La
  fuerza por pata es la reacción en esos nodos (× 2 por la simetría en z).

Modelo de material
------------------
Cáscara: E = E_t (ficha), σ_y = σ_t,XY (resistencia a tracción de ficha) y
H = h_frac · E. Núcleo de relleno: E y σ_y escalados por el mismo k (misma
deformación de fluencia). La rotura se estima cuando la deformación plástica
equivalente en la cáscara alcanza ε_b − σ_t/E_t (alargamiento de ficha menos
la parte elástica).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .element import NAT, dN_dxi
from .geometry import GEOMETRIES, Y_TOPE
from .material import PROFILES, element_modulus
from .mesh import VoxelMesh, build_mesh
from .solver import element_dofs

M_V = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
GP = np.array([[a, b, c] for c in (-1, 1) for b in (-1, 1) for a in (-1, 1)]) / np.sqrt(3.0)


@dataclass
class NLMaterial:
    name: str
    E: float        # módulo de tracción [MPa]
    nu: float
    sy: float       # tensión de fluencia [MPa]
    H: float        # módulo de endurecimiento [MPa]
    eps_b: float    # alargamiento a la rotura [-]


@dataclass
class NLResult:
    material: str
    geometry: str
    profile: str
    h: float
    n_elements: int
    n_dof: int
    opening: list = field(default_factory=list)     # apertura en las puntas [mm]
    force: list = field(default_factory=list)       # fuerza por pata [N]
    s1_max: list = field(default_factory=list)      # σ1 (Cauchy aprox.) máx en cáscara
    epbar_max: list = field(default_factory=list)   # deformación plástica eq. máx en cáscara
    plastic_frac: list = field(default_factory=list)
    iters: list = field(default_factory=list)
    F_first_yield: float | None = None
    F_limit: float | None = None
    F_rupture: float | None = None
    F_offset: float | None = None   # fuerza con 1 mm de apertura permanente (offset)
    k0: float | None = None         # rigidez inicial [N/mm]
    open_rupture: float | None = None
    seconds: float = 0.0


OFFSET_MM = 1.0


def offset_force(opening, force, offset=OFFSET_MM):
    """Fuerza a la que la apertura supera en `offset` a la elástica (método del offset).

    Aproxima la carga con la que el gancho queda abierto `offset` mm de forma
    permanente al descargar (la descarga es elástica con la rigidez inicial).
    """
    o, f = np.asarray(opening), np.asarray(force)
    if len(o) < 2:
        return None, None
    k0 = f[0] / o[0]
    perm = o - f / k0
    if perm.max() < offset:
        return float(k0), None
    i = int(np.argmax(perm >= offset))
    t = (offset - perm[i - 1]) / (perm[i] - perm[i - 1])
    return float(k0), float(f[i - 1] + t * (f[i] - f[i - 1]))


def quarter_mesh(geom_name: str, h: float) -> VoxelMesh:
    """Malla del cuarto x ≥ 0, z ≥ 0 (renumerada)."""
    full = build_mesh(GEOMETRIES[geom_name], h)
    keep = (full.ijk[:, 0] >= 0) & (full.ijk[:, 2] >= 0)
    keep[keep] = _face_connected(full.ijk[keep])
    conn = full.conn[keep]
    used, inv = np.unique(conn.ravel(), return_inverse=True)
    conn = inv.reshape(-1, 8)
    counts = np.bincount(conn.ravel(), minlength=len(used))
    return VoxelMesh(h=h, ijk=full.ijk[keep], conn=conn, nodes=full.nodes[used],
                     centroids=full.centroids[keep], node_count=counts)


def _face_connected(ijk: np.ndarray) -> np.ndarray:
    """Máscara de la mayor componente conexa por caras.

    Los vóxeles unidos al resto solo por una arista o un vértice son
    mecanismos (matriz singular), inofensivos para el PCG lineal pero no para
    el método de Newton con factorización directa.
    """
    from scipy.sparse.csgraph import connected_components

    key = {tuple(v): n for n, v in enumerate(ijk)}
    rows, cols = [], []
    for d in np.eye(3, dtype=np.int64):
        for n, v in enumerate(ijk):
            m = key.get(tuple(v + d))
            if m is not None:
                rows.append(n)
                cols.append(m)
    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(ijk),) * 2)
    _, lab = connected_components(A, directed=False)
    return lab == np.argmax(np.bincount(lab))


class TLPlasticModel:
    def __init__(self, geom_name: str, profile: str, mat: NLMaterial, h: float = 1.0):
        self.geom = GEOMETRIES[geom_name]
        self.mesh = quarter_mesh(geom_name, h)
        self.mat = mat
        self.prof = PROFILES[profile]
        depth = self.geom.depth(*self.mesh.centroids.T)
        Ee, self.shell = element_modulus(depth, self.prof, 1.0)  # factor relativo
        self.scale = Ee                         # 1 en cáscara, k en núcleo
        self.E = mat.E * Ee
        self.sy = mat.sy * Ee
        self.H = mat.H * Ee
        nu = mat.nu
        self.G = self.E / (2 * (1 + nu))
        self.K = self.E / (3 * (1 - 2 * nu))

        ne = self.mesh.ne
        self.ndof = 3 * self.mesh.nn
        self.dofs = element_dofs(self.mesh.conn)
        # Derivadas cartesianas en los 8 puntos de Gauss (iguales para todos los cubos)
        self.dN = np.stack([dN_dxi(*g) * (2.0 / h) for g in GP])  # (8gp, 8n, 3)
        self.w = (h / 2) ** 3                                     # peso × det J
        # Variables internas por punto de Gauss
        self.ep = np.zeros((ne, 8, 6))
        self.alpha = np.zeros((ne, 8))
        # Patrón de ensamblaje
        r = np.repeat(self.dofs, 24, axis=1).ravel()
        c = np.tile(self.dofs, (1, 24)).ravel()
        self._rc = (r, c)
        self._bc()

    # ------------------------------------------------------------------
    def _bc(self):
        X, Y, Z = self.mesh.nodes.T
        tol = 1e-9
        fixed = {}
        for n in np.where(np.abs(X) < tol)[0]:
            fixed[3 * n] = 0.0
        for n in np.where(np.abs(Z) < tol)[0]:
            fixed[3 * n + 2] = 0.0
        # u_y = 0 en el nodo exterior de la corona (x = 0, z = 0, y máx)
        cand = np.where((np.abs(X) < tol) & (np.abs(Z) < tol))[0]
        top = cand[np.argmax(Y[cand])]
        fixed[3 * top + 1] = 0.0
        dn, y_axis, _ = self.geom.project(X, Y)
        A, delta = self.geom.section(y_axis)
        load = self.mesh.surface_nodes & ((dn - delta) > 0) & (Y <= Y_TOPE + tol) & (X > 1.0)
        self.load_nodes = np.where(load)[0]
        self.load_dofs = 3 * self.load_nodes
        self.fixed_zero = np.array(sorted(fixed), dtype=np.int64)
        self.presc = self.fixed_zero
        self.free = np.setdiff1d(np.arange(self.ndof), self.presc)
        self.tip_nodes = np.where(np.abs(Y - Y.min()) < tol)[0]

    # ------------------------------------------------------------------
    def _kinematics(self, U):
        ue = U[self.dofs].reshape(-1, 8, 3)                       # (ne, 8n, 3)
        Hg = np.einsum("eai,gaj->egij", ue, self.dN)              # (ne, gp, 3, 3) ∂u_i/∂X_j
        F = Hg + np.eye(3)
        Egl = 0.5 * (Hg + np.swapaxes(Hg, -1, -2) + np.einsum("egki,egkj->egij", Hg, Hg))
        Ev = np.stack([Egl[..., 0, 0], Egl[..., 1, 1], Egl[..., 2, 2],
                       2 * Egl[..., 0, 1], 2 * Egl[..., 1, 2], 2 * Egl[..., 0, 2]], axis=-1)
        # B no lineal (ne, gp, 6, 24)
        dN = self.dN
        ne = ue.shape[0]
        B = np.zeros((ne, 8, 6, 8, 3))
        for i in range(3):
            B[:, :, 0, :, i] = F[:, :, None, i, 0] * dN[None, :, :, 0]
            B[:, :, 1, :, i] = F[:, :, None, i, 1] * dN[None, :, :, 1]
            B[:, :, 2, :, i] = F[:, :, None, i, 2] * dN[None, :, :, 2]
            B[:, :, 3, :, i] = F[:, :, None, i, 0] * dN[None, :, :, 1] + F[:, :, None, i, 1] * dN[None, :, :, 0]
            B[:, :, 4, :, i] = F[:, :, None, i, 1] * dN[None, :, :, 2] + F[:, :, None, i, 2] * dN[None, :, :, 1]
            B[:, :, 5, :, i] = F[:, :, None, i, 0] * dN[None, :, :, 2] + F[:, :, None, i, 2] * dN[None, :, :, 0]
        B = B.reshape(ne, 8, 6, 24)
        # Método Ē: volumétrica media del elemento
        trE = Ev[..., :3].sum(-1)
        Ev_bar = Ev + (trE.mean(1, keepdims=True) - trE)[..., None] * M_V / 3.0
        mB = B[:, :, :3, :].sum(2)                                # (ne, gp, 24)
        corr = (mB.mean(1, keepdims=True) - mB) / 3.0
        Bbar = B + M_V[None, None, :, None] * corr[:, :, None, :]
        return F, Ev_bar, Bbar

    def _return_map(self, Ev, ep_old, a_old):
        E = self.E[:, None, None]
        nu = self.mat.nu
        G = self.G[:, None]
        Kb = self.K[:, None]
        H = self.H[:, None]
        sy = self.sy[:, None]
        ee = Ev - ep_old
        tr = ee[..., :3].sum(-1)
        dev = ee.copy()
        dev[..., :3] -= tr[..., None] / 3.0
        # tensión de prueba (Voigt tensión)
        s_tr = 2 * G[..., None] * np.concatenate([dev[..., :3], 0.5 * dev[..., 3:]], axis=-1)
        p = Kb * tr
        nrm = np.sqrt((s_tr[..., :3] ** 2).sum(-1) + 2 * (s_tr[..., 3:] ** 2).sum(-1))
        q = np.sqrt(1.5) * nrm
        f = q - (sy + H * a_old)
        yld = f > 1e-10
        dg = np.where(yld, f / (3 * G + H), 0.0)
        nvec = s_tr / np.maximum(nrm, 1e-30)[..., None]
        s = s_tr - (2 * G * np.sqrt(1.5) * dg)[..., None] * nvec * yld[..., None]
        S = s.copy()
        S[..., :3] += p[..., None]
        dep = (np.sqrt(1.5) * dg)[..., None] * nvec
        dep_eng = np.concatenate([dep[..., :3], 2 * dep[..., 3:]], axis=-1)
        ep_new = ep_old + dep_eng * yld[..., None]
        a_new = a_old + dg
        # Tangente consistente
        P = np.diag([1, 1, 1, 0.5, 0.5, 0.5]) - np.outer(M_V, M_V) / 3.0
        Cel = Kb[..., None, None] * np.outer(M_V, M_V) + 2 * G[..., None, None] * P
        beta = np.where(yld, 1 - 3 * G * dg / np.maximum(q, 1e-30), 1.0)
        gbar = np.where(yld, 1.0 / (1 + H / (3 * G)) - (1 - beta), 0.0)
        C = Kb[..., None, None] * np.outer(M_V, M_V) + (2 * G * beta)[..., None, None] * P \
            - (2 * G * gbar)[..., None, None] * np.einsum("egi,egj->egij", nvec, nvec)
        C = np.where(yld[..., None, None], C, np.broadcast_to(Cel, C.shape))
        return S, C, ep_new, a_new, yld

    def residual_tangent(self, U, need_K=True):
        F, Ev, B = self._kinematics(U)
        S, C, ep, al, yld = self._return_map(Ev, self.ep, self.alpha)
        w = self.w
        fe = np.einsum("egij,egi->ej", B, S) * w                  # (ne, 24)
        fint = np.bincount(self.dofs.ravel(), weights=fe.ravel(), minlength=self.ndof)
        Kmat = None
        if need_K:
            Ke = np.einsum("egki,egkl,eglj->eij", B, C, B, optimize=True) * w
            Sm = np.empty(S.shape[:-1] + (3, 3))
            Sm[..., 0, 0], Sm[..., 1, 1], Sm[..., 2, 2] = S[..., 0], S[..., 1], S[..., 2]
            Sm[..., 0, 1] = Sm[..., 1, 0] = S[..., 3]
            Sm[..., 1, 2] = Sm[..., 2, 1] = S[..., 4]
            Sm[..., 0, 2] = Sm[..., 2, 0] = S[..., 5]
            Gab = np.einsum("gai,egij,gbj->eab", self.dN, Sm, self.dN) * w  # (ne, 8, 8)
            Kg = np.einsum("eab,ij->eaibj", Gab, np.eye(3)).reshape(-1, 24, 24)
            Ke = Ke + Kg
            Kmat = sp.coo_matrix((Ke.ravel(), self._rc), shape=(self.ndof, self.ndof)).tocsr()
        return fint, Kmat, (S, ep, al, yld, F)

    # ------------------------------------------------------------------
    def run(self, open_max: float = 20.0, n_steps: int = 40, tol: float = 1e-6, max_it: int = 15,
            verbose: bool = False, stop_factor: float = 1.5) -> NLResult:
        """Control por desplazamiento de la apertura con patrón de carga fijo.

        La carga es una fuerza uniforme λ·p en +x sobre la cara exterior de la
        punta (como en el análisis lineal), y λ es incógnita: se impone la
        apertura media de la cara extrema, c·u = d (método de control de
        desplazamiento con sistema orlado). Así se sigue la curva fuerza–
        apertura incluso pasado el punto límite.
        """
        t0 = time.perf_counter()
        m = self.mesh
        res = NLResult(self.mat.name, self.geom.name, self.prof.name, m.h, m.ne, len(self.free))
        nd = self.ndof
        p = np.zeros(nd)
        p[self.load_dofs] = 1.0 / len(self.load_dofs)          # λ = fuerza en el cuarto
        c = np.zeros(nd)
        c[3 * self.tip_nodes] = 2.0 / len(self.tip_nodes)       # apertura = 2·media(u_x)
        fr, pf, cf = self.free, p[self.free], c[self.free]
        U = np.zeros(nd)
        lam = 0.0
        dd = open_max / n_steps
        d = 0.0
        eps_rupt = self.mat.eps_b - self.mat.sy / self.mat.E
        while d < open_max - 1e-12 and dd > 1e-5 * open_max:
            d_new = min(d + dd, open_max)
            Ut, lt = U.copy(), lam
            ok = False
            for it in range(max_it):
                fint, K, st = self.residual_tangent(Ut)
                R = (lt * p - fint)[fr]
                g = d_new - c @ Ut
                ref = max(abs(lt), 1.0) / len(self.load_dofs) * 10
                rn = np.linalg.norm(R) / max(abs(lt), 1.0)
                if verbose:
                    print(f"  d={d_new:.3f} it={it} |R|/λ={rn:.2e} g={g:.1e} λ={lt:.2f}")
                if it > 0 and rn < tol and abs(g) < 1e-9 * open_max:
                    ok = True
                    break
                lu = spla.splu(K[fr][:, fr].tocsc())
                uR = lu.solve(R)
                up = lu.solve(pf)
                dl = (g - cf @ uR) / (cf @ up)
                du = uR + dl * up
                if not (np.all(np.isfinite(du)) and np.isfinite(dl)):
                    break
                Ut[fr] += du
                lt += dl
            if not ok:
                dd /= 2
                continue
            S, ep, al, yld, F = st
            self.ep, self.alpha = ep, al
            U, lam, d = Ut, lt, d_new
            force = 2.0 * lam                                   # por pata (simetría en z)
            sh = self.shell
            res.opening.append(float(d))
            res.force.append(float(force))
            res.s1_max.append(float(self._sigma1_cauchy(S[sh], F[sh], q=99.8)))
            epbar = float(al[sh].max())
            res.epbar_max.append(epbar)
            res.plastic_frac.append(float((al[sh] > 0).mean()))
            res.iters.append(it)
            if res.F_first_yield is None and epbar > 0:
                res.F_first_yield = float(force)
            if res.F_rupture is None and epbar >= eps_rupt:
                res.F_rupture, res.open_rupture = float(force), float(d)
            if epbar > stop_factor * eps_rupt:
                break
            if it <= 4:
                dd = min(dd * 1.5, open_max / 10)
        res.F_limit = float(max(res.force)) if res.force else None
        res.k0, res.F_offset = offset_force(res.opening, res.force, OFFSET_MM)
        res.seconds = time.perf_counter() - t0
        self.U = U
        return res

    @staticmethod
    def _sigma1_cauchy(S, F, q=None):
        Sm = np.empty(S.shape[:-1] + (3, 3))
        Sm[..., 0, 0], Sm[..., 1, 1], Sm[..., 2, 2] = S[..., 0], S[..., 1], S[..., 2]
        Sm[..., 0, 1] = Sm[..., 1, 0] = S[..., 3]
        Sm[..., 1, 2] = Sm[..., 2, 1] = S[..., 4]
        Sm[..., 0, 2] = Sm[..., 2, 0] = S[..., 5]
        J = np.linalg.det(F)
        sig = np.einsum("...ik,...kl,...jl->...ij", F, Sm, F) / J[..., None, None]
        s1 = np.linalg.eigvalsh(sig)[..., -1]
        return s1.max() if q is None else np.percentile(s1, q)
