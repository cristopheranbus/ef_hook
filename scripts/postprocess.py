"""Genera tablas (CSV y LaTeX), figuras y la comparación con el estudio original.

Uso:
    python scripts/postprocess.py

Lee results/sims/*.json y results/fields/*.npz (producidos por run_all.py) y
escribe:
    results/simulaciones.csv          una fila por simulación
    results/resumen.json              cifras derivadas que cita el informe
    results/comparacion.md            reproducción frente a cifras publicadas
    report/tablas/*.tex               tablas que incluye el informe
    report/cifras.tex                 macros con las cifras citadas en el texto
    figures/*.pdf, figures/*.png      figuras
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ganchofem import paper_values as PV  # noqa: E402
from ganchofem.analytic import crown_moment, winkler_circular  # noqa: E402
from ganchofem.geometry import GEOMETRIES, window_area  # noqa: E402
from ganchofem.material import E_REF, FILAMENTS  # noqa: E402

SIMS = ROOT / "results" / "sims"
FIELDS = ROOT / "results" / "fields"
TAB = ROOT / "report" / "tablas"
FIG = ROOT / "figures"
MAT_ORDER = ["PAHT-CF", "PC", "ABS", "PLA", "ASA", "PETG"]
ETA = 0.3
T_SERV = 40.0


def load():
    R = {}
    for p in sorted(SIMS.glob("*.json")):
        R[p.stem] = json.loads(p.read_text())
    return R


def fmt(v, nd=1):
    """Número con coma decimal para LaTeX (siunitx no es necesario)."""
    if v is None:
        return "—"
    s = f"{v:,.{nd}f}".replace(",", "X").replace(".", "{,}").replace("X", "\\,")
    return s


def pct(a, b):
    return 100.0 * (a - b) / b


def write_tex(name, header, rows, colspec, caption, label, note=None):
    TAB.mkdir(parents=True, exist_ok=True)
    lines = [r"\begin{table}[htbp]", r"\centering\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             r"\fitbox{", f"\\begin{{tabular}}{{{colspec}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    for r in rows:
        if r == "mid":
            lines.append(r"\midrule")
        else:
            lines.append(" & ".join(r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}"]
    if note:
        lines.append(f"\\par\\smallskip\\parbox{{0.95\\linewidth}}{{\\footnotesize {note}}}")
    lines.append(r"\end{table}")
    (TAB / f"{name}.tex").write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------
def main():
    R = load()
    print(f"{len(R)} simulaciones cargadas")
    FIG.mkdir(exist_ok=True)
    S = {}  # resumen

    # ---------------- CSV global
    keys = ["sim_id", "series", "geometry", "case", "profile", "h", "E", "nu", "k"]
    cols = ["n_elements", "n_dof_free", "nnz", "solver", "iterations", "residual", "t_solve",
            "volume_cm3", "s1_max", "s1_p998", "svm_max", "opening", "n_load_nodes"]
    with open(ROOT / "results" / "simulaciones.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys + cols)
        for sid, r in R.items():
            w.writerow([r["spec"][k] for k in keys] + [r[c] for c in cols])

    g = lambda sid: R[sid]  # noqa: E731

    # ---------------- Tabla de series (sección 7)
    from collections import Counter
    cnt = Counter(r["spec"]["series"] for r in R.values())
    S["n_sims"] = len(R)
    S["series_count"] = dict(cnt)

    # ---------------- Tamaño de modelos
    size_ids = ["D-G6-macizo-h1.0", "D-G6-macizo-h0.75", "C-G6-5p25", "D-G1-2p15-k0.0225",
                "C-G10-5p25", "D-G6-macizo-h0.4"]
    size_lab = ["G6 macizo", "G6 macizo", "G6, 5 paredes", "G1, 2 paredes, $k=0{,}0225$",
                "G10, 5 paredes", "G6 macizo"]
    rows = []
    for sid, lab in zip(size_ids, size_lab):
        r = g(sid)
        it = "—" if r["solver"].startswith("LU") else f"{r['iterations']}"
        rows.append([lab, fmt(r["spec"]["h"], 2), f"{r['n_elements']:,}".replace(",", r"\,"),
                     f"{r['n_dof_free']:,}".replace(",", r"\,"), f"{r['nnz']/1e6:.1f}".replace(".", "{,}") + r"$\cdot10^6$",
                     r["solver"], it, f"{r['residual']:.1e}".replace(".", "{,}"), fmt(r["t_solve"], 0)])
    write_tex("tamanos", ["Modelo", "$h$ [mm]", "Elementos", "GDL libres", "nnz($K_{ff}$)", "Solver",
                          "Iter.", "Residuo", "$t$ [s]"], rows, "lrrrrlrrr",
              "Tamaño y costo de los modelos (valores registrados por el solver de este repositorio).",
              "tab:tamanos", "Tiempos en un contenedor de 4 núcleos con 3 simulaciones en paralelo.")
    S["max_dof"] = max(r["n_dof_free"] for r in R.values())

    # ---------------- 8.1 Independencia de E
    a, b = g("C-G6-5p25"), g("D-G6-5p25-E1880")
    S["E_test"] = dict(s1_ref=a["s1_max"], s1_abs=b["s1_max"], p_ref=a["s1_p998"], p_abs=b["s1_p998"],
                       op_ref=a["opening"], op_abs=b["opening"], ratio=b["opening"] / a["opening"],
                       rel_diff=abs(a["s1_max"] - b["s1_max"]) / a["s1_max"])
    write_tex("E_indep", ["Corrida (G6, 5 paredes, $h=0{,}5$ mm)", "$E$ [MPa]", r"$\sigma_{1,\max}$ [MPa]",
                          "Percentil 99,8 [MPa]", "Apertura [mm]"],
              [["Referencia", "3500", fmt(a["s1_max"], 2), fmt(a["s1_p998"], 2), fmt(a["opening"], 4)],
               ["Módulo de flexión del ABS", "1880", fmt(b["s1_max"], 2), fmt(b["s1_p998"], 2), fmt(b["opening"], 4)]],
              "lrrrr", "Independencia de la tensión respecto de $E$.", "tab:E")

    # ---------------- 8.2 Poisson
    rows, S["nu"] = [], {}
    for nu, sid in [(0.30, "D-G6-5p25-nu0.30"), (0.35, "C-G6-5p25"), (0.40, "D-G6-5p25-nu0.40")]:
        r = g(sid)
        var = "referencia" if nu == 0.35 else f"{pct(r['s1_max'], a['s1_max']):+.1f} \\%".replace(".", "{,}")
        rows.append([fmt(nu, 2), fmt(r["s1_max"], 2), var, fmt(r["opening"], 3)])
        S["nu"][nu] = r["s1_max"]
    write_tex("nu", [r"$\nu$", r"$\sigma_{1,\max}$ [MPa]", "Variación", "Apertura [mm]"], rows, "rrrr",
              "Sensibilidad al coeficiente de Poisson (G6, 5 paredes).", "tab:nu")

    # ---------------- 8.3 Relleno
    krows = [("G6, 5 paredes", "0,25 (adoptado)", "C-G6-5p25", "ref6"),
             ("G6, 5 paredes", "0,0625 ($n=2$)", "D-G6-5p25-k0.0625", "C-G6-5p25"),
             ("G1, 2 paredes", "0,15 ($n=1$)", "D-G1-2p15-k0.15", "C-G1-2p15"),
             ("G1, 2 paredes", "0,08 (adoptado)", "C-G1-2p15", "ref1"),
             ("G1, 2 paredes", "0,0225 ($n=2$)", "D-G1-2p15-k0.0225", "C-G1-2p15")]
    rows, S["k"] = [], {}
    for lab, k, sid, ref in krows:
        r = g(sid)
        var = "referencia" if ref.startswith("ref") else f"{pct(r['s1_max'], g(ref)['s1_max']):+.1f} \\%".replace(".", "{,}")
        rows.append([lab, k, fmt(r["s1_max"], 1), var, fmt(r["opening"], 2)])
        S["k"][sid] = r["s1_max"]
    write_tex("k", ["Geometría y perfil", "$k$", r"$\sigma_{1,\max}$ [MPa]", "Variación", "Apertura [mm]"], rows,
              "llrrr", "Sensibilidad al módulo del relleno.", "tab:k")

    # ---------------- 8.4 Convergencia
    conv = {}
    for h in (1.0, 0.75, 0.5, 0.4):
        s6 = "C-G6-macizo" if h == 0.5 else f"D-G6-macizo-h{h}"
        s1 = "C-G1-macizo" if h == 0.5 else f"D-G1-macizo-h{h}"
        conv[h] = (g(s6)["s1_max"], g(s6)["opening"], g(s6)["s1_p998"],
                   g(s1)["s1_max"] if s1 in R else None)
    S["conv"] = {str(k): v for k, v in conv.items()}
    rows = [[fmt(h, 2), fmt(v[0], 2), fmt(v[2], 2), fmt(v[1], 3), fmt(v[3], 2)] for h, v in conv.items()]
    write_tex("conv", ["$h$ [mm]", r"G6: $\sigma_{1,\max}$", "G6: P99,8", "G6: apertura [mm]", r"G1: $\sigma_{1,\max}$"],
              rows, "rrrrr", "Convergencia de malla (macizo, caso 3; tensiones en MPa).", "tab:conv")
    S["conv_G6_last"] = pct(conv[0.4][0], conv[0.5][0])
    S["conv_G6_open_last"] = pct(conv[0.4][1], conv[0.5][1])
    S["conv_G1_jump"] = pct(conv[0.5][3], conv[0.75][3])

    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
    hs = np.array(list(conv))
    ax[0].plot(hs, [conv[h][0] for h in hs], "o-", label=r"G6 $\sigma_{1,\max}$")
    ax[0].plot(hs, [conv[h][2] for h in hs], "s--", label="G6 P99,8")
    ax[0].plot(hs[:3], [conv[h][3] for h in hs[:3]], "^-", label=r"G1 $\sigma_{1,\max}$")
    W = winkler_circular(crown_moment()[1], 100.0, 22.5, 6.0)["sigma_i"]
    ax[0].axhline(W, color="k", lw=0.8, ls=":", label="Winkler (G6)")
    ax[0].set_xlabel("h [mm]"); ax[0].set_ylabel("Tensión [MPa]"); ax[0].invert_xaxis(); ax[0].legend(fontsize=7)
    ax[1].plot(hs, [conv[h][1] for h in hs], "o-")
    ax[1].set_xlabel("h [mm]"); ax[1].set_ylabel("Apertura G6 [mm]"); ax[1].invert_xaxis()
    fig.tight_layout(); fig.savefig(FIG / "convergencia.pdf"); fig.savefig(FIG / "convergencia.png", dpi=200)
    plt.close(fig)

    # ---------------- 8.5 Winkler
    arm, M = crown_moment()
    w = winkler_circular(M, 100.0, 22.5, 6.0)
    S["winkler"] = dict(arm=arm, M=M, **{k: float(v) for k, v in w.items()},
                        fem_max=conv[0.4][0], fem_p998=conv[0.4][2])

    # ---------------- 9.1 Casos
    rows, S["cases"] = [], {}
    for gname, lab in [("G1", r"G1 $\cdot$ R6 (v5 actual)"), ("G4", r"G4 $\cdot$ R15 (v8)"), ("G6", r"G6 $\cdot$ arco R22,5 (v7)")]:
        v = [g(f"A-{gname}-5p25")["s1_max"], g(f"B-{gname}-5p25")["s1_max"], g(f"C-{gname}-5p25")["s1_max"]]
        S["cases"][gname] = v
        rows.append([lab] + [fmt(x, 1) for x in v])
    write_tex("casos", ["Geometría (5 paredes, 25 \\%)", "Caso 1: tiro, 200 N", "Caso 2: 100 N/pata", "Caso 3: 100 N/pata"],
              rows, "lrrr", r"$\sigma_{1,\max}$ [MPa] en la cáscara para los tres casos de carga.", "tab:casos",
              "En los casos 1 y 2 se excluye la zona empotrada ($y\\le-22{,}5$ mm).")
    ratios = [S["cases"][k][2] / max(S["cases"][k][:2]) for k in S["cases"]]
    S["case_ratio_min"], S["case_ratio_max"] = min(ratios), max(ratios)

    # ---------------- 9.2 Geometrías (caso 3)
    rows, S["geom"] = [], {}
    for gname in [f"G{i}" for i in range(1, 11)]:
        r = g(f"C-{gname}-5p25")
        win = window_area(GEOMETRIES[gname])
        S["geom"][gname] = dict(s1=r["s1_max"], p=r["s1_p998"], vm=r["svm_max"], op=r["opening"],
                                vol=r["volume_cm3"], win=win)
        rows.append([f"{gname} $\\cdot$ {GEOMETRIES[gname].label}", fmt(r["s1_max"], 1), fmt(r["s1_p998"], 1),
                     fmt(r["svm_max"], 1), fmt(r["opening"], 2), fmt(r["volume_cm3"], 1), fmt(win, 0)])
    write_tex("geometrias", ["Geometría", r"$\sigma_{1,\max}$", "P99,8", r"$\sigma_{VM,\max}$", "Apertura [mm]",
                             "Vol. [cm$^3$]", "Ventana [mm$^2$]"], rows, "lrrrrrr",
              "Exploración geométrica, caso 3, 5 paredes y 25 \\%, 100 N por pata, $E=3500$ MPa (tensiones en MPa).",
              "tab:geom", "Ventana: área libre en el plano medio, en la mitad $x\\ge0$.")
    G = S["geom"]
    S["G10_vs_G1"] = G["G1"]["s1"] / G["G10"]["s1"]
    S["G10_vs_G6"] = G["G6"]["s1"] / G["G10"]["s1"]
    S["G6_vs_G1"] = G["G1"]["s1"] / G["G6"]["s1"]
    S["R6_R9"] = -pct(G["G2"]["s1"], G["G1"]["s1"])
    S["R9_arc"] = -pct(G["G6"]["s1"], G["G2"]["s1"])
    S["vm_over_s1_G6"] = g("C-G6-5p25")["extra"]["vm_over_s1_at_peak"]

    # ---------------- 10 Materiales
    s_v7, s_v8, s_v5 = G["G6"]["s1"], G["G4"]["s1"], g("C-G1-2p15")["s1_max"]
    s_v5p = g("C-G1-2p15")["s1_p998"]
    op_v7 = G["G6"]["op"]
    rows, S["mat"] = [], {}
    for m in MAT_ORDER:
        f = FILAMENTS[m]
        Fu7 = 100 * f.st_xy / s_v7
        Fu8 = 100 * f.st_xy / s_v8
        Fu10 = 100 * f.st_xy / G["G10"]["s1"]
        op50 = op_v7 * E_REF / f.E_f * 0.5
        duct = f.eb_xy / (100 * f.st_xy / f.E_t)
        S["mat"][m] = dict(Fu7=Fu7, Fu8=Fu8, Fu10=Fu10, Fu5_lo=100 * f.st_xy / s_v5, Fu5_hi=100 * f.st_xy / s_v5p,
                           Fsost=ETA * Fu7, op50=op50, duct=duct, hdt_margin=f.hdt - T_SERV,
                           U50_v7=50 * s_v7 / 100 / f.st_xy, U50_v10=50 * G["G10"]["s1"] / 100 / f.st_xy)
        rows.append([f.name, fmt(Fu7, 0), fmt(Fu8, 0), fmt(ETA * Fu7, 0), fmt(op50, 1), fmt(duct, 1),
                     fmt(f.hdt - T_SERV, 0), fmt(f.impact, 1)])
    write_tex("materiales", ["Material", "$F_u$ v7 [N]", "$F_u$ v8 [N]", r"$F_{adm}$ sost. [N]", "Apertura 50 N [mm]",
                             r"$\varepsilon_b/\varepsilon_{pico}$", "Margen HDT [°C]", "Impacto [kJ/m$^2$]"],
              rows, "lrrrrrrr", "Indicadores por material (5 paredes, 25 \\%, caso 3).", "tab:mat",
              f"$F_u = 100\\,\\mathrm{{N}}\\cdot\\sigma_{{t,XY}}/\\sigma_{{1,\\max}}$; sostenida con $\\eta={fmt(ETA,1)}$; "
              f"apertura escalada con el módulo de flexión; margen HDT con {T_SERV:.0f} °C de servicio.")
    S["v5_to_v7_gain"] = s_v5 / s_v7
    S["v5p_to_v7_gain"] = s_v5p / s_v7
    S["paht_vs_pla"] = S["mat"]["PAHT-CF"]["Fu7"] / S["mat"]["PLA"]["Fu7"]
    S["pc_vs_pla"] = S["mat"]["PC"]["Fu7"] / S["mat"]["PLA"]["Fu7"]
    S["s_v5_2p"] = s_v5
    S["s_v5_2p_p998"] = s_v5p

    _figures(R, S)
    _nonlinear(S)
    _creep(S)
    _comparison(R, S)
    _macros(S)
    (ROOT / "results" / "resumen.json").write_text(json.dumps(S, indent=1, ensure_ascii=False))
    print("listo")


# ----------------------------------------------------------------------
def _poly(fields, scale=0.0):
    c = fields["corners"] + scale * fields["ucorners"]
    return c[:, [0, 1, 2, 3], :]


def _figures(R, S):
    # Figura 1: mapas de σ1 de las 10 geometrías
    fig, axs = plt.subplots(2, 5, figsize=(13, 6.2))
    vmax = 60
    for ax, gname in zip(axs.ravel(), [f"G{i}" for i in range(1, 11)]):
        F = np.load(FIELDS / f"C-{gname}-5p25.npz")
        polys = _poly(F)
        col = np.where(F["shell"], F["s1"], np.nan)
        pc = PolyCollection(polys, array=np.ma.masked_invalid(col), cmap="turbo", edgecolors="none")
        pc.set_clim(0, vmax)
        core = PolyCollection(polys[~F["shell"]], facecolors="0.75", edgecolors="none")
        ax.add_collection(core); ax.add_collection(pc)
        ax.set_xlim(-30, 30); ax.set_ylim(-31, 32); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"{gname} · {GEOMETRIES[gname].label}\nσ1,max = {S['geom'][gname]['s1']:.1f} MPa", fontsize=8)
    cb = fig.colorbar(pc, ax=axs, shrink=0.7, label="σ1 [MPa] (escala saturada en 60)")
    cb.ax.tick_params(labelsize=8)
    fig.savefig(FIG / "fig1_sigma1_geometrias.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig1_sigma1_geometrias.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Figuras 2 y 3: utilización por material con deformada ×3, 50 N por pata
    for gname, fname in [("G6", "fig2_utilizacion_v7"), ("G10", "fig3_utilizacion_G10")]:
        F = np.load(FIELDS / f"C-{gname}-5p25.npz")
        fig, axs = plt.subplots(2, 3, figsize=(11, 8))
        for ax, m in zip(axs.ravel(), MAT_ORDER):
            f = FILAMENTS[m]
            # Campos a 100 N y E_ref; se escalan a 50 N y al módulo de flexión
            scale_u = 0.5 * E_REF / f.E_f * 3.0
            U = np.where(F["shell"], 0.5 * F["s1"] / f.st_xy, np.nan)
            base = PolyCollection(_poly(F), facecolors="0.88", edgecolors="none")
            pd = _poly(F, scale_u)
            pc = PolyCollection(pd, array=np.ma.masked_invalid(U), cmap="RdYlGn_r", edgecolors="none")
            pc.set_clim(0, 1)
            core = PolyCollection(pd[~F["shell"]], facecolors="0.6", edgecolors="none")
            ax.add_collection(base); ax.add_collection(core); ax.add_collection(pc)
            ax.set_xlim(-36, 36); ax.set_ylim(-32, 32); ax.set_aspect("equal"); ax.axis("off")
            op = S["geom"][gname]["op"] * E_REF / f.E_f * 0.5
            ax.set_title(f"{f.name}\nU máx = {np.nanmax(U):.2f} · apertura = {op:.1f} mm", fontsize=9)
        fig.colorbar(pc, ax=axs, shrink=0.6, label="Utilización U = σ1 / σt,XY")
        fig.savefig(FIG / f"{fname}.pdf", bbox_inches="tight")
        fig.savefig(FIG / f"{fname}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # Frente de Pareto capacidad–ventana
    G = S["geom"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    base = G["G1"]["s1"]
    pts = sorted(((v["win"], base / v["s1"], k) for k, v in G.items()), reverse=True)
    best, front = 0, []
    for wv, c, k in pts:
        ax.plot(wv, c, "o", color="C3" if k == "G10" else "0.4")
        ax.annotate(k, (wv, c), textcoords="offset points", xytext=(4, 4), fontsize=8)
        if c > best:
            best = c; front.append((wv, c))
    ax.plot(*zip(*front), "--", color="0.5", lw=1, label="Frente de Pareto")
    ax.set_xlabel("Ventana interior (mitad x ≥ 0) [mm²]"); ax.set_ylabel("Capacidad relativa a G1")
    ax.legend(); fig.tight_layout()
    fig.savefig(FIG / "pareto.pdf"); fig.savefig(FIG / "pareto.png", dpi=200); plt.close(fig)

    # Capacidad por material
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    y = np.arange(len(MAT_ORDER))
    v7 = [S["mat"][m]["Fu7"] for m in MAT_ORDER]
    v5 = [S["mat"][m]["Fu5_lo"] for m in MAT_ORDER]
    ax.barh(y - 0.2, v7, 0.4, label="v7 (G6), 5 paredes 25 %", color="C0")
    ax.barh(y + 0.2, v5, 0.4, label="v5 (G1), 2 paredes 15 % (cota inferior)", color="C0", alpha=0.35)
    for yi, a in zip(y, v7):
        ax.text(a + 2, yi - 0.2, f"{a:.0f} N", va="center", fontsize=8)
    ax.axvline(100, color="k", ls=":", lw=0.8)
    ax.set_yticks(y, [FILAMENTS[m].name for m in MAT_ORDER]); ax.invert_yaxis()
    ax.set_xlabel("Fuerza de rotura por pata, caso 3 [N]"); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "capacidad_materiales.pdf"); fig.savefig(FIG / "capacidad_materiales.png", dpi=200)
    plt.close(fig)


NL_GEOMS = [("G1", "2p15", "v5 actual (G1), 2 paredes 15 %"), ("G6", "5p25", "v7 (G6), 5 paredes 25 %"),
            ("G10", "5p25", "G10, 5 paredes 25 %")]


def _nonlinear(S):
    """Curvas fuerza–apertura del análisis no lineal y tabla de cargas características."""
    d = ROOT / "results" / "nonlinear"
    if not d.exists():
        return
    NL = {p.stem: json.loads(p.read_text()) for p in d.glob("*.json")}
    S["nl"] = {}
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    rows = []
    for ax, (gname, prof, lab) in zip(axs, NL_GEOMS):
        for i, m in enumerate(MAT_ORDER):
            r = NL.get(f"{gname}-{prof}-{m}")
            if r is None:
                continue
            o, f = np.r_[0, r["opening"]], np.r_[0, r["force"]]
            ax.plot(o, f, color=f"C{i}", label=FILAMENTS[m].name)
            if r["F_rupture"]:
                ax.plot(r["open_rupture"], r["F_rupture"], "x", color=f"C{i}", ms=8, mew=2)
            if r["F_offset"] and (not r["F_rupture"] or r["F_offset"] <= r["F_rupture"]):
                k = np.interp(r["F_offset"], f, o)
                ax.plot(k, r["F_offset"], "o", color=f"C{i}", mfc="none")
            key = f"{gname}-{prof}"
            lin = S["geom"].get(gname, {}).get("s1") if prof == "5p25" else S["s_v5_2p"]
            Fu_lin = 100 * FILAMENTS[m].st_xy / lin
            S["nl"].setdefault(key, {})[m] = dict(F_offset=r["F_offset"], F_rupture=r["F_rupture"],
                                                  open_rupture=r["open_rupture"], F_yield=r["F_first_yield"],
                                                  F_max=r["F_limit"], k0=r["k0"], Fu_lin=Fu_lin)
            off = r["F_offset"] if r["F_offset"] and (not r["F_rupture"] or r["F_offset"] <= r["F_rupture"]) else None
            rows.append([f"{gname} ({'2p' if prof == '2p15' else '5p'})", FILAMENTS[m].name, fmt(Fu_lin, 0),
                         fmt(r["F_first_yield"], 0), fmt(off, 0), fmt(r["F_rupture"], 0),
                         fmt(r["open_rupture"], 1)])
        ax.set_title(lab, fontsize=10); ax.set_xlabel("Apertura de las puntas [mm]")
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("Fuerza por pata [N]"); axs[0].legend(fontsize=8)
    fig.text(0.5, -0.03, "○: 1 mm de apertura permanente (el gancho queda «estirado»)   ×: rotura estimada "
             "(deformación plástica = alargamiento de ficha)", ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "nl_fuerza_apertura.pdf", bbox_inches="tight")
    fig.savefig(FIG / "nl_fuerza_apertura.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    write_tex("nolineal", ["Geometría", "Material", r"$F_u$ lineal", "1.ª fluencia", r"$F_{1\,mm}$ permanente",
                           r"$F$ rotura", "Apertura en rotura [mm]"], rows, "llrrrrr",
              "Análisis no lineal (plasticidad J2 + grandes desplazamientos), caso 3. Fuerzas por pata en N.",
              "tab:nl", "$F_u$ lineal: criterio de Rankine del análisis elástico (sección 10). 1.ª fluencia: primer "
              "punto de Gauss de la cáscara que plastifica (von Mises). $F_{1\\,mm}$: fuerza con la que, al descargar, "
              "queda 1 mm de apertura permanente. —: el material rompe antes de llegar a ese punto.")
    # controles
    base = NL.get("G6-5p25-PLA")
    for tag in ("G6-5p25-PLA-h0.75", "G6-5p25-PLA-H0"):
        r = NL.get(tag)
        if r and base:
            S["nl"][tag] = dict(F_offset=r["F_offset"], F_rupture=r["F_rupture"], k0=r["k0"],
                                base_offset=base["F_offset"], base_rupture=base["F_rupture"], base_k0=base["k0"])


def _creep(S):
    """Fluencia: apertura en el tiempo bajo la carga sostenida admisible (escenarios de Findley)."""
    from ganchofem.creep import SCENARIOS
    S["creep"] = {}
    t = np.logspace(-1, np.log10(10 * 8760), 200)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    rows = []
    for i, m in enumerate(MAT_ORDER):
        sc, f = SCENARIOS[m], FILAMENTS[m]
        axs[0].semilogx(t, sc.phi(t), color=f"C{i}", label=f.name)
        for gname in ("G6", "G10"):
            op_per_N = S["geom"][gname]["op"] * E_REF / f.E_f / 100
            F_s = ETA * 100 * f.st_xy / S["geom"][gname]["s1"]
            op0 = op_per_N * F_s
            op1y, op10y = op0 * (1 + sc.phi(8760)), op0 * (1 + sc.phi(87600))
            S["creep"].setdefault(gname, {})[m] = dict(F_sost=F_s, op0=op0, op1y=float(op1y), op10y=float(op10y),
                                                       phi1000=float(sc.phi(1000)), phi10y=float(sc.phi(87600)))
            if gname == "G10":
                axs[1].semilogx(t, op0 * (1 + sc.phi(t)), color=f"C{i}")
            rows.append([gname, f.name, fmt(F_s, 0), fmt(op0, 2), fmt(float(sc.phi(1000)), 2),
                         fmt(float(op1y), 2), fmt(float(op10y), 2)])
    axs[0].set_xlabel("Tiempo [h]"); axs[0].set_ylabel("Coeficiente de fluencia φ(t)"); axs[0].legend(fontsize=8)
    axs[0].set_title("Escenarios de Findley (no medidos)", fontsize=10)
    axs[1].set_xlabel("Tiempo [h]"); axs[1].set_ylabel("Apertura [mm]")
    axs[1].set_title("G10 con la carga sostenida admisible (η = 0,3)", fontsize=10)
    for a in axs:
        a.grid(alpha=0.3, which="both"); a.axvline(8760, color="0.6", ls=":"); a.axvline(87600, color="0.6", ls=":")
    fig.tight_layout(); fig.savefig(FIG / "fluencia.pdf"); fig.savefig(FIG / "fluencia.png", dpi=200); plt.close(fig)
    write_tex("fluencia", ["Geom.", "Material", r"$F_{sost}$ [N]", r"$\Delta_0$ [mm]", r"$\varphi$(1000 h)",
                           r"$\Delta$ 1 año [mm]", r"$\Delta$ 10 años [mm]"], rows, "llrrrrr",
              "Apertura bajo la carga sostenida admisible ($\\eta=0{,}3$), por fluencia lenta a 23 °C.", "tab:creep",
              "Los coeficientes de fluencia son escenarios de Findley que respetan el orden cualitativo de [14, 15], "
              "no datos de ensayo. Calibrar con \\texttt{scripts/ajuste\\_fluencia.py}.")


def _comparison(R, S):
    L = ["# Reproducción frente al estudio original", "",
         "Diferencia = (este repositorio − estudio) / estudio.", "",
         "## Exploración geométrica (caso 3, 5 paredes, h = 0,5 mm)", "",
         "| Geometría | σ1,max estudio | σ1,max aquí | Δ | P99,8 estudio | P99,8 aquí | Δ | Apertura estudio | Apertura aquí | Δ |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for k, (s1, p, vm, op, vol, win) in PV.GEOM_C3.items():
        m = S["geom"][k]
        L.append(f"| {k} | {s1} | {m['s1']:.2f} | {pct(m['s1'], s1):+.1f} % | {p} | {m['p']:.2f} | {pct(m['p'], p):+.1f} % "
                 f"| {op} | {m['op']:.3f} | {pct(m['op'], op):+.1f} % |")
    L += ["", "## Casos de carga (σ1,max, 5 paredes)", "", "| Geometría | Caso | Estudio | Aquí | Δ |", "|---|---|---|---|---|"]
    for k, vals in PV.CASES.items():
        for c in range(3):
            L.append(f"| {k} | {c+1} | {vals[c]} | {S['cases'][k][c]:.1f} | {pct(S['cases'][k][c], vals[c]):+.0f} % |")
    L += ["", "## Convergencia (macizo)", "", "| h | G6 σ1 estudio | aquí | G6 apertura estudio | aquí | G1 σ1 estudio | aquí |",
          "|---|---|---|---|---|---|---|"]
    for h, (a, b, c) in PV.CONV.items():
        m = S["conv"][str(h)]
        L.append(f"| {h} | {a} | {m[0]:.2f} | {b} | {m[1]:.3f} | {c if c else '—'} | {m[3]:.2f} |" if m[3] else
                 f"| {h} | {a} | {m[0]:.2f} | {b} | {m[1]:.3f} | — | — |")
    L += ["", "## Sensibilidad a k", "", "| Caso | σ1 estudio | aquí | apertura estudio | aquí |", "|---|---|---|---|---|"]
    kmap = {("G6", "5p25", 0.25): "C-G6-5p25", ("G6", "5p25", 0.0625): "D-G6-5p25-k0.0625",
            ("G1", "2p15", 0.15): "D-G1-2p15-k0.15", ("G1", "2p15", 0.08): "C-G1-2p15",
            ("G1", "2p15", 0.0225): "D-G1-2p15-k0.0225"}
    for key, (s, o) in PV.K_SENS.items():
        r = R[kmap[key]]
        L.append(f"| {key[0]} {key[1]} k={key[2]} | {s} | {r['s1_max']:.2f} | {o} | {r['opening']:.2f} |")
    L += ["", "## Poisson (G6, 5 paredes)", "", "| ν | σ1 estudio | aquí |", "|---|---|---|"]
    for nu, (s, o) in PV.NU.items():
        L.append(f"| {nu} | {s} | {S['nu'][nu]:.2f} |")
    L += ["", "## Tamaño de modelos", "", "| Modelo | Elem. estudio | aquí | GDL estudio | aquí | Iter. estudio | aquí |",
          "|---|---|---|---|---|---|---|"]
    for sid, (ne, nd, it, t) in PV.SIZES.items():
        r = R[sid]
        L.append(f"| {sid} | {ne} | {r['n_elements']} | {nd} | {r['n_dof_free']} | {it or '—'} | "
                 f"{r['iterations'] if r['iterations'] else '—'} |")
    L += ["", "## Capacidad por material, v7 [N por pata]", "", "| Material | Estudio | Aquí |", "|---|---|---|"]
    for m, v in PV.FU_V7.items():
        L.append(f"| {m} | {v} | {S['mat'][m]['Fu7']:.0f} |")
    (ROOT / "results" / "comparacion.md").write_text("\n".join(L) + "\n")


def _macros(S):
    """Macros LaTeX con las cifras citadas en el texto del informe."""
    G, M = S["geom"], S["mat"]
    d = {
        "NSims": f"{S['n_sims']}",
        "MaxDof": fmt(S["max_dof"] / 1e5, 1) + r"\cdot10^5",
        "SGsix": fmt(G["G6"]["s1"], 1), "SGone": fmt(G["G1"]["s1"], 1), "SGten": fmt(G["G10"]["s1"], 1),
        "PGsix": fmt(G["G6"]["p"], 1),
        "OpGsix": fmt(G["G6"]["op"], 2),
        "GtenVsGone": fmt(S["G10_vs_G1"], 1), "GtenVsGsix": fmt(S["G10_vs_G6"], 1), "GsixVsGone": fmt(S["G6_vs_G1"], 2),
        "RsixRnine": fmt(S["R6_R9"], 0), "RnineArc": fmt(S["R9_arc"], 0),
        "WinGone": fmt(G["G1"]["win"], 0), "WinGten": fmt(G["G10"]["win"], 0), "WinGsix": fmt(G["G6"]["win"], 0),
        "WinLossGten": fmt(100 * (1 - G["G10"]["win"] / G["G1"]["win"]), 0),
        "WinLossGsix": fmt(100 * (1 - G["G6"]["win"] / G["G1"]["win"]), 0),
        "VolGone": fmt(G["G1"]["vol"], 1), "VolGten": fmt(G["G10"]["vol"], 1),
        "VolGainGten": fmt(100 * (G["G10"]["vol"] / G["G1"]["vol"] - 1), 0),
        "CaseRatioMin": fmt(S["case_ratio_min"], 0), "CaseRatioMax": fmt(S["case_ratio_max"], 0),
        "ConvLast": fmt(abs(S["conv_G6_last"]), 1), "ConvOpLast": fmt(abs(S["conv_G6_open_last"]), 1),
        "ConvGoneJump": fmt(S["conv_G1_jump"], 0),
        "WinklerSi": fmt(S["winkler"]["sigma_i"], 1), "WinklerKi": fmt(S["winkler"]["Ki"], 2),
        "WinklerStraight": fmt(S["winkler"]["straight"], 1),
        "FemMax": fmt(S["winkler"]["fem_max"], 1), "FemPct": fmt(S["winkler"]["fem_p998"], 1),
        "FemMaxDev": fmt(pct(S["winkler"]["fem_max"], S["winkler"]["sigma_i"]), 0),
        "FemPctDev": fmt(pct(S["winkler"]["fem_p998"], S["winkler"]["sigma_i"]), 0),
        "EratioOpen": fmt(S["E_test"]["ratio"], 4),
        "VmOverSone": fmt(S["vm_over_s1_G6"], 2),
        "FuPAHT": fmt(M["PAHT-CF"]["Fu7"], 0), "FuPC": fmt(M["PC"]["Fu7"], 0), "FuPLA": fmt(M["PLA"]["Fu7"], 0),
        "FuPETG": fmt(M["PETG"]["Fu7"], 0), "FuABS": fmt(M["ABS"]["Fu7"], 0),
        "PAHTvsPLA": fmt(S["paht_vs_pla"], 1), "PCvsPLA": fmt(S["pc_vs_pla"], 1),
        "GainVfiveVseven": fmt(S["v5_to_v7_gain"], 1),
        "GainTotal": fmt(S["v5_to_v7_gain"] * S["paht_vs_pla"], 1),
        "FuTenPAHT": fmt(M["PAHT-CF"]["Fu10"], 0), "FuTenPC": fmt(M["PC"]["Fu10"], 0),
        "FuTenPAHTadm": fmt(M["PAHT-CF"]["Fu10"] / 2, 0), "FuTenPAHTsost": fmt(0.3 * M["PAHT-CF"]["Fu10"], 0),
        "FuFivePLAlo": fmt(M["PLA"]["Fu5_lo"], 0), "FuFivePLAhi": fmt(M["PLA"]["Fu5_hi"], 0),
        "UPLAseven": fmt(M["PLA"]["U50_v7"], 2), "UPAHTseven": fmt(M["PAHT-CF"]["U50_v7"], 2),
        "UPCseven": fmt(M["PC"]["U50_v7"], 2), "UPETGseven": fmt(M["PETG"]["U50_v7"], 2),
        "UPLAten": fmt(M["PLA"]["U50_v10"], 2), "UPAHTten": fmt(M["PAHT-CF"]["U50_v10"], 2),
        "OpPAHTseven": fmt(M["PAHT-CF"]["op50"], 1), "OpPLAseven": fmt(M["PLA"]["op50"], 1),
        "OpPAHTten": fmt(G["G10"]["op"] * E_REF / FILAMENTS["PAHT-CF"].E_f * 0.5, 1),
        "DuctPC": fmt(M["PC"]["duct"], 1),
    }
    if "nl" in S:
        nl6, nl10, nl1 = S["nl"]["G6-5p25"], S["nl"]["G10-5p25"], S["nl"]["G1-2p15"]
        ductile = [m for m in MAT_ORDER if nl6[m]["F_offset"]]
        r = [nl6[m]["F_offset"] / nl6[m]["Fu_lin"] for m in ductile]
        d.update({
            "NLoffPLAsix": fmt(nl6["PLA"]["F_offset"], 0), "NLrupPLAsix": fmt(nl6["PLA"]["F_rupture"], 0),
            "NLoffPAHTsix": fmt(nl6["PAHT-CF"]["F_offset"], 0), "NLrupPAHTsix": fmt(nl6["PAHT-CF"]["F_rupture"], 0),
            "NLrupPCsix": fmt(nl6["PC"]["F_rupture"], 0), "NLoffPETGsix": fmt(nl6["PETG"]["F_offset"], 0),
            "NLoffPLAten": fmt(nl10["PLA"]["F_offset"], 0), "NLoffPAHTten": fmt(nl10["PAHT-CF"]["F_offset"], 0),
            "NLrupPAHTten": fmt(nl10["PAHT-CF"]["F_rupture"], 0), "NLrupPCten": fmt(nl10["PC"]["F_rupture"], 0),
            "NLoffPLAone": fmt(nl1["PLA"]["F_offset"], 0), "NLyPLAone": fmt(nl1["PLA"]["F_yield"], 0),
            "NLrupPLAone": fmt(nl1["PLA"]["F_rupture"], 0),
            "NLratioMin": fmt(min(r), 1), "NLratioMax": fmt(max(r), 1),
            "NLmeshOff": fmt(abs(pct(S["nl"]["G6-5p25-PLA-h0.75"]["F_offset"], nl6["PLA"]["F_offset"])), 0),
            "NLmeshRup": fmt(abs(pct(S["nl"]["G6-5p25-PLA-h0.75"]["F_rupture"], nl6["PLA"]["F_rupture"])), 0),
            "NLhardOff": fmt(abs(pct(S["nl"]["G6-5p25-PLA-H0"]["F_offset"], nl6["PLA"]["F_offset"])), 0),
            "NLhardRup": fmt(abs(pct(S["nl"]["G6-5p25-PLA-H0"]["F_rupture"], nl6["PLA"]["F_rupture"])), 0),
        })
    if "creep" in S:
        c10 = S["creep"]["G10"]
        d.update({"CrPLAzero": fmt(c10["PLA"]["op0"], 1), "CrPLAten": fmt(c10["PLA"]["op10y"], 1),
                  "CrPCzero": fmt(c10["PC"]["op0"], 1), "CrPCten": fmt(c10["PC"]["op10y"], 1),
                  "CrPAHTzero": fmt(c10["PAHT-CF"]["op0"], 1), "CrPAHTten": fmt(c10["PAHT-CF"]["op10y"], 1),
                  "CrPAHTF": fmt(c10["PAHT-CF"]["F_sost"], 0), "CrPLAF": fmt(c10["PLA"]["F_sost"], 0)})
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in d.items()]
    (ROOT / "report" / "cifras.tex").write_text("% Generado por scripts/postprocess.py\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
