# Diferencias de esta reconstrucción con el estudio original

El estudio original (Claude Doc *"Análisis por elementos finitos del gancho en
U (PLA)"*, 19-11-2026) describe la formulación completa pero no incluye el
código. Este repositorio la reimplementa desde cero a partir del texto y las
ecuaciones. La tabla completa de cifras lado a lado está en
[`results/comparacion.md`](../results/comparacion.md); aquí se resume lo que
coincide, lo que no y por qué.

## 1. Lo que coincide

- **Mallas con h = 0,5 mm:** mismo número de elementos y de grados de libertad
  en G1 (132 660 elementos, 442 212 GDL) y G6 (120 460, 403 548). Mismas
  iteraciones de PCG en G6 con 5 paredes (1048) y casi las mismas en G1 con
  k = 0,0225 (1688 frente a 1685). Esto indica que el criterio de mallado, la
  alineación de la grilla y las condiciones de borde son las mismas.
- **Tensiones y aperturas del caso 3 (series C y D, h = 0,5 mm):** todas
  quedan entre −0,1 % y +0,4 % de las publicadas (por ejemplo G6: 54,48 frente
  a 54,34 MPa; G10: 33,42 frente a 33,3 MPa).
- **Casos 1 y 2:** iguales al decimal publicado (diferencias ≤ 1 %).
- **Verificaciones:** el teorema de escalado se cumple hasta el último dígito,
  la sensibilidad a ν y a k reproduce las variaciones publicadas, y la solución
  de Winkler da los mismos 37,9 MPa.
- **Capacidades por material:** idénticas al newton (el PETG da 62 N en vez de
  63 N por el redondeo de σ₁).
- **Volúmenes y ventanas:** mismos volúmenes (16,6 a 18,3 cm³); ventanas
  dentro de 6 mm² (≤ 1 %).

## 2. Lo que difiere y por qué

| Tema | Estudio | Aquí | Causa probable |
|---|---|---|---|
| Sesgo sistemático de +0,2 a +0,4 % en σ₁ | — | — | Detalles no documentados: tolerancias de selección de nodos cargados, o la definición exacta de la cara exterior. No cambia ninguna conclusión. |
| Malla h = 0,75 mm | 35 736 elementos, σ₁ = 46,52 MPa (G6) y 72,31 MPa (G1) | 36 564 elementos, 45,25 y 67,03 MPa | El estudio dice que realineó la grilla con x = 0 para h = 1,0 y 0,75, pero no cuál fue el origen en y y z. Aquí todas las grillas tienen nodos en múltiplos enteros de h en los tres ejes. Con h = 0,75 el escalonado de la superficie cambia y el pico en G1 (singular) es muy sensible a eso. |
| Malla h = 0,4 mm | 234 348 elementos | 237 032 | Mismo motivo (0,4 no es exacto en binario y cambia qué centroides caen justo en φ = 0). La tensión coincide (47,96 frente a 47,79 MPa). |
| Iteraciones de PCG con h = 0,75 | 1128 | 608 | Malla distinta (fila anterior). |
| Tiempos de cálculo | 2 núcleos, 8 GB | 4 núcleos, 3 simulaciones en paralelo | Máquinas distintas; los tiempos no son comparables. |
| Transición de sección en G7–G10 | "transición entre y = −15 e y = 5 mm" (forma no especificada) | smoothstep cúbico sobre la altura del punto del eje | Elección propia; da G10 con 138 620 elementos frente a 138 652 (0,02 %). |
| Definición de "apertura" | no especificada | Desplazamiento relativo medio en x de las caras extremas de las patas (y = −30 mm) | Es la definición que reproduce las aperturas publicadas (5,325 frente a 5,314 mm en G6). |
| Composición exacta de las 63 simulaciones | Totales por serie (16, 16, 17, 14) | Mismos totales; la lista concreta está en `ganchofem/series.py` | El estudio no lista cada corrida. Las series A y B incluyen variantes engrosadas hacia afuera (G4/G6 a 14 y 16 mm), que el estudio menciona sin detallar. |
| Exclusión en los casos 1 y 2 | no especificada | Se excluyen los elementos de la zona empotrada (y ≤ −22,5 mm) | Con esa convención las tensiones coinciden con las publicadas. |

## 3. Lo que se agregó (no estaba en el estudio)

- **Análisis no lineal** (plasticidad J2 + grandes desplazamientos,
  `ganchofem/nonlinear.py`): responde a "¿con qué carga el gancho cede y se
  estira?".
- **Fluencia lenta** (`ganchofem/creep.py`): apertura en el tiempo bajo carga
  sostenida, con escenarios de Findley y un script para calibrarla con un
  ensayo casero (`scripts/ajuste_fluencia.py`).
- **Modelos imprimibles** (`modelos_stl/`): STL cerrados de G1–G10 generados
  desde la misma función implícita del MEF.
- **Pruebas automáticas** (`tests/`).
