# Gancho en U impreso en 3D: selección de material por elementos finitos

Código, resultados e informe del estudio *"Selección de material para un gancho
en U impreso en 3D mediante análisis por elementos finitos"*.

Un gancho en U de PLA (barra Ø12, contorno 57 × 60 mm) se rompe en servicio en
las esquinas. Este repositorio contiene un solver de elementos finitos 3D
escrito desde cero en Python (NumPy + SciPy) que:

- genera la geometría de forma implícita para 10 variantes de esquina y corona,
- malla con vóxeles hexaédricos de 8 nodos (integración completa 2 × 2 × 2),
- representa la pieza impresa con un modelo homogeneizado cáscara–núcleo
  (Gibson–Ashby),
- resuelve con LU dispersa o gradiente conjugado precondicionado con Jacobi,
- y recupera tensiones en el centroide para compararlas con la resistencia de
  seis filamentos (PLA, PETG, ASA, ABS, PC y PAHT-CF).

Además de reproducir el estudio original (todas las cifras del caso de diseño
quedan dentro de +0,4 %; ver [`docs/diferencias.md`](docs/diferencias.md)),
el repositorio agrega:

- **Análisis no lineal** (plasticidad J2 + grandes desplazamientos): la carga
  con la que el gancho "se estira" y queda abierto, y su rotura.
- **Fluencia lenta** bajo carga sostenida, con un script para calibrarla con un
  ensayo casero.
- **Modelos imprimibles**: STL cerrados de las 10 geometrías en
  [`modelos_stl/`](modelos_stl/).

**Resultado principal:** el PAHT-CF es el material más adecuado (≈ 2,3 veces la
capacidad del PLA) y la geometría G10 (arco con la corona engrosada hacia
adentro) multiplica la capacidad por ≈ 3,2 respecto del gancho actual. Ver
[`report/informe.pdf`](report/informe.pdf).

## Estructura

```
ganchofem/            paquete del solver
  geometry.py         eje paramétrico, proyección analítica y función implícita φ
  mesh.py             malla de vóxeles alineada con x = 0
  element.py          hex8: funciones de forma, B, D y K̂_e
  material.py         perfiles de impresión (cáscara–núcleo) y fichas técnicas
  solver.py           ensamblaje CSR, LU dispersa y PCG-Jacobi
  model.py            casos de carga, solución, tensiones e indicadores
  analytic.py         viga curva de Winkler
  nonlinear.py        Lagrangiana total + plasticidad J2 (método Ē), control por desplazamiento
  creep.py            fluencia (principio de correspondencia + Findley)
  series.py           las 63 simulaciones (series A–D)
  paper_values.py     cifras publicadas en el estudio original (para comparar)
scripts/
  run_all.py          corre las simulaciones (paralelo y reanudable)
  run_nonlinear.py    20 análisis no lineales (6 materiales × 3 geometrías + controles)
  postprocess.py      tablas CSV/LaTeX, figuras y comparación
  export_stl.py       modelos imprimibles
  ajuste_fluencia.py  ajusta Findley a un ensayo de carga sostenida
tests/                pruebas del elemento, la malla, Winkler y el escalado
results/              resultados por simulación (JSON), campos del plano medio (NPZ),
                      simulaciones.csv, resumen.json y comparacion.md
figures/              figuras en PDF y PNG
modelos_stl/          STL imprimibles de G1–G10 (mm, U acostada en la cama)
report/               informe en LaTeX y PDF compilado
docs/diferencias.md   diferencias de esta reconstrucción con el estudio original
```

## Cómo reproducir

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                                  # ~20 s
python scripts/run_all.py --jobs 3         # 63 simulaciones, ~40 min con 4 núcleos y 16 GB
python scripts/run_nonlinear.py --jobs 3   # análisis no lineal, ~10 min
python scripts/postprocess.py              # tablas, figuras y report/cifras.tex
python scripts/export_stl.py               # STL imprimibles, ~30 s
cd report && latexmk -pdf informe.tex      # informe.pdf
```

Una simulación suelta:

```python
from ganchofem.model import SimSpec, run
res, _ = run(SimSpec("prueba", geometry="G6", case=3, profile="5p25", h=0.5))
print(res.s1_max, res.opening)   # ≈ 54,5 MPa y 5,32 mm con 100 N por pata
```

`run_all.py` salta las simulaciones cuyo JSON ya existe; usa `--force` para
recalcular o `--only C` para una sola serie. La simulación más grande
(h = 0,4 mm, 7,7·10⁵ GDL) necesita unos 4 GB de RAM.

## Licencia

MIT. Ver [LICENSE](LICENSE).
