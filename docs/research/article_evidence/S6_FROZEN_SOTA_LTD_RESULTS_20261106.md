# LTD-Attack — expediente de resultados S6 para el artículo

**Registro:** 2026-11-06 · Fuente: consola de Perseo (ejecución real) · Estado: resultados S6 generados y revisión read-only realizada · **No es todavía un manuscrito sometible ni un resultado prospectivo ciego.**

## 1. Proveniencia y condiciones

- Repositorio: `a52290451/ltd-attack`, rama `feature/macro-v2-historical-only`; S6 ejecutado en Perseo y finalizó `S6_PAIRED_COMPARISON=PASS`. El revisor `scripts/sota/s6_report_and_s7_input_gate_v1.py` finalizó `S6_REPORT_AND_S7_INPUT_GATE=PASS_READ_ONLY`.
- Resultados originales: `/data/ltd-attack/sota/s6/comparison_v1/` (no modificar):
  - `00_model_metrics.csv`
  - `01_sota_vs_hybrid_paired.csv`
  - `02_daily_metrics.csv`
  - `03_site_metrics.csv`
  - `manifest.json` (SHA256 de los reportes)
  - `README.md` (método y advertencias)
- Las probabilidades SOTA originales están en `/data/ltd-attack/sota/s5/inference/<cohort>/<model>/` y el mapeo de las 65 clases en `/data/ltd-attack/sota/s5/frozen_s5_refits_and_labels_v1.json`. Los resultados LTD 09B y Future-B se obtuvieron de sus NPZ congelados. S6 verificó hashes, UID, fechas, etiquetas, orden de clases y poblaciones.
- Comparación de cinco clasificadores: **LTD-Hybrid, LTD-Micro, LTD-Macro, DF-WFlib, VarCNN-WFlib**, en las mismas capturas dentro de cada cohorte; **09B** = 14.687 capturas, 11 fechas; **Future-B DEV_FROZEN** = 18.543 capturas, 16 fechas. 65 clases, single-tab closed-world, tracks históricos distintos.
- Entrenamiento DF/VarCNN exclusivo en Historical DEV = 57.916 capturas; selección de época sobre DEV_LATE y refit C sobre DEV completo. Checkpoints: DF SHA256 `0bb25cb4aa245af4ac545936b39bef73986cb4c034c9c31e853ca527fc3ceab3` (28 épocas); VarCNN `642a944c8d751d5a9e4f81714cd2cc957dab6a1badfb0ed4e23627199ee8a54f` (30 épocas). Ninguno se reentrenó para S6.
- Diferencia en esta batería: 2 baselines SOTA × 4 métricas vs LTD-Hybrid, familia Holm8 **separada por cohorte**. IC95 percentil de bootstrap pareado por bloques de fecha (2.500 réplicas); pruebas de intercambio condicional por día: enumeración exacta para Accuracy/Top5/MRR, Monte Carlo 9.999 para Macro-F1. Debido a fechas consecutivas y comparaciones poshoc, sus valores p son **exploratorios y condicionados a intercambiabilidad de bloques**, no prueba confirmatoria independiente.
- Observación metodológica: el grupo de 65 sitios podría estar condicionado históricamente por disponibilidad de Future-B; Future-B fue previamente abierto. No afirmar validación prospectiva ciega ni inferir causalidad exclusiva de concept drift a partir de la caída 09B→Future-B. No mezclar Future-B DEV_FROZEN con HISTORICAL_FINAL.

## 2. Tabla de resultados: 09B

Las métricas de Accuracy, Macro-F1 y Top-5 se muestran en porcentaje; MRR en [0,1]. Cifras tal como imprimió el revisor (redondeadas por ese script).

| Modelo | N | Accuracy (%) | Macro-F1 (%) | Top-5 (%) | MRR |
|---|---:|---:|---:|---:|---:|
| LTD-Micro | 14.687 | 95.002 | 94.818 | 99.020 | 0.9685 |
| LTD-Macro | 14.687 | 84.068 | 83.946 | 98.523 | 0.9039 |
| **LTD-Hybrid** | **14.687** | **97.440** | **97.379** | **99.816** | **0.9853** |
| DF | 14.687 | 93.423 | 93.239 | 99.163 | 0.9593 |
| VarCNN | 14.687 | 97.290 | 97.216 | 99.775 | 0.9840 |

### 09B — comparaciones pareadas

**Signo:** Δ = SOTA − LTD-Hybrid. Δ negativo favorece a LTD-Hybrid. Las unidades de Δ e IC son puntos porcentuales para Accuracy/Macro-F1/Top-5; **para MRR se muestra Δ×100** (escalado a puntos de MRR, no accuracy).

| SOTA | Métrica | Δ×100 | IC95 bajo | IC95 alto | p sin ajuste | p Holm8 | Prueba |
|---|---|---:|---:|---:|---:|---:|---|
| DF | Accuracy | -4.017 | -4.596 | -3.550 | 0.0009765625 | 0.0078125 | exact day-swap |
| DF | Macro-F1 | -4.140 | -4.760 | -3.639 | 0.0011 | 0.0078125 | MC day-swap |
| DF | Top-5 | -0.654 | -0.857 | -0.484 | 0.0009765625 | 0.0078125 | exact day-swap |
| DF | MRR×100 | -2.594 | -2.912 | -2.324 | 0.0009765625 | 0.0078125 | exact day-swap |
| VarCNN | Accuracy | -0.150 | -0.384 | 0.088 | 0.291015625 | 0.825 | exact day-swap |
| VarCNN | Macro-F1 | -0.163 | -0.416 | 0.126 | 0.275 | 0.825 | MC day-swap |
| VarCNN | Top-5 | -0.041 | -0.136 | 0.039 | 0.4765625 | 0.825 | exact day-swap |
| VarCNN | MRR×100 | -0.125 | -0.261 | -0.001 | 0.1025390625 | 0.41015625 | exact day-swap |

**Lectura 09B:** LTD-Hybrid supera a DF en las cuatro métricas con IC95 de Δ por debajo de cero; frente a VarCNN, las diferencias de Accuracy, Macro-F1 y Top-5 tienen intervalos que incluyen cero y el p Holm8 no apoya una diferencia detectable. La pequeña diferencia MRR frente a VarCNN tiene IC95 percentil negativo pero el p Holm8 no es significativo; **no presentarla como superioridad inferencial confirmada**.

## 3. Tabla de resultados: Future-B DEV_FROZEN

| Modelo | N | Accuracy (%) | Macro-F1 (%) | Top-5 (%) | MRR |
|---|---:|---:|---:|---:|---:|
| LTD-Micro | 18.543 | 43.116 | 44.134 | 72.399 | 0.5615 |
| LTD-Macro | 18.543 | 22.305 | 18.328 | 57.995 | 0.3865 |
| **LTD-Hybrid** | **18.543** | **49.453** | **49.262** | **80.559** | **0.6306** |
| DF | 18.543 | 37.464 | 36.192 | 75.020 | 0.5358 |
| VarCNN | 18.543 | 33.447 | 31.140 | 67.244 | 0.4880 |

### Future-B — comparaciones pareadas

| SOTA | Métrica | Δ×100 | IC95 bajo | IC95 alto | p sin ajuste | p Holm8 | Prueba |
|---|---|---:|---:|---:|---:|---:|---|
| DF | Accuracy | -11.988 | -13.090 | -10.866 | 0.000030517578125 | 0.000244140625 | exact day-swap |
| DF | Macro-F1 | -13.070 | -14.069 | -11.992 | 0.0001 | 0.000244140625 | MC day-swap |
| DF | Top-5 | -5.538 | -6.450 | -4.517 | 0.00006103515625 | 0.000244140625 | exact day-swap |
| DF | MRR×100 | -9.483 | -10.293 | -8.547 | 0.000030517578125 | 0.000244140625 | exact day-swap |
| VarCNN | Accuracy | -16.006 | -17.084 | -14.971 | 0.000030517578125 | 0.000244140625 | exact day-swap |
| VarCNN | Macro-F1 | -18.122 | -18.975 | -17.240 | 0.0002 | 0.000244140625 | MC day-swap |
| VarCNN | Top-5 | -13.315 | -13.979 | -12.579 | 0.000030517578125 | 0.000244140625 | exact day-swap |
| VarCNN | MRR×100 | -14.260 | -15.022 | -13.492 | 0.000030517578125 | 0.000244140625 | exact day-swap |

**Lectura Future-B:** LTD-Hybrid tiene mejor resultado puntual en las cuatro métricas frente a ambos baselines; los IC95 de Δ SOTA-Hybrid están enteramente por debajo de cero y los p Holm8 post-hoc informados son 0.00024414 para los ocho contrastes. Interpretar como **evidencia exploratoria fuerte bajo el procedimiento condicional de bloques**, no como confirmación causal o test ciego.

## 4. Retención temporal observada (Accuracy)

Las diferencias en puntos porcentuales entre cohortes son descriptivas, no emparejamiento de las mismas capturas entre periodos.

| Modelo | Accuracy 09B | Accuracy Future-B | Δ Future-B − 09B |
|---|---:|---:|---:|
| LTD-Hybrid | 97.440 % | 49.453 % | -47.987 pp |
| LTD-Micro | 95.002 % | 43.116 % | -51.886 pp |
| LTD-Macro | 84.068 % | 22.305 % | -61.763 pp |
| DF | 93.423 % | 37.464 % | -55.959 pp |
| VarCNN | 97.290 % | 33.447 % | -63.843 pp |

LTD-Hybrid tiene la menor caída absoluta de Accuracy de los cinco modelos en estas dos cohortes. Ello **no demuestra por sí solo** que el mecanismo LTD sea causalmente responsable, puesto que intervienen desplazamiento temporal, posibles cambios de condiciones de captura y dificultad de las muestras.

## 5. Hallazgos y limitaciones a conservar en artículo

1. **Conclusión principal provisional:** en 10B DEV_FROZEN, el LTD-Hybrid alcanza 49.453 % Accuracy y 49.262 % Macro-F1, frente a DF 37.464/36.192 y VarCNN 33.447/31.140; delta de Accuracy LTD-Hybrid − DF = +11.988 pp; LTD-Hybrid − VarCNN = +16.006 pp. Se acompañan de IC y valores p exploratorios detallados arriba.
2. **09B:** LTD-Hybrid vs VarCNN muestra rendimiento prácticamente equivalente por Accuracy/Macro-F1, sin evidencia de una diferencia significativa bajo Holm8. No reportar victoria estadística donde no existe.
3. **LTD-Micro y LTD-Macro** están incluidos descriptivamente en la comparación común S6. **Este script S6 no generó contrastes pareados inferenciales Hybrid vs Micro/Macro.** Para ello existe un expediente congelado separado; no atribuirles los Holm8 SOTA.
4. **VarCNN** presentó oscilaciones extremas en DEV_LATE a lo largo de las épocas; conservar curva completa como limitación. DF/VarCNN implementados con WFlib fijado, pero no garantizar reproducción exacta del paper original.
5. **SOTA aún incompleto:** faltan CUMUL/SVM, Tik-Tok, NetCLR y métodos adicionales. No afirmar superioridad frente a todo el estado del arte; esta conclusión es limitada a DF y VarCNN evaluados.
6. **Paquetes y CUMUL:** el nuevo gate encontró columnas Historical `direction_vector`, `time_vector` y `size_vector`; por tanto, la entrada candidata existe. Antes de presentar CUMUL como fiel, verificar semántica original del tamaño (signo/dirección, overhead, orden), procedencia y extracción. El viejo `src/models/sota/SOTA_01_CUMUL_SVM.py` usa 94 características históricamente informadas por futuro y **no es admisible como baseline canónico**.
7. **Dataset C:** no se utiliza en S6. Continúa como cohorte futura en preparación, sin estado validado de reloj NTP y cobertura de capturas. No mezclar con 09B/10B ni editar su cron.
8. **Auditoría estadística aún recomendada antes de manuscrito:** revisar manualmente `01_sota_vs_hybrid_paired.csv`, comprobar familia Holm8 y su implementación, valores p exactos vs Monte Carlo y dependencia temporal. Los decimales aquí reproducen el revisor de Perseo; no es una auditoría externa independiente.

## 6. Texto candidato de resultados para el artículo (borrador)

> En una evaluación retrospectiva de 65 sitios web y dos cohortes temporales distintas, se comparó LTD-Hybrid con dos baselines de aprendizaje profundo, DF y VarCNN, utilizando capturas y etiquetas emparejadas por identificador dentro de cada cohorte. En la cohorte interna 09B (n = 14.687), LTD-Hybrid obtuvo 97,44 % de exactitud, frente a 93,42 % de DF y 97,29 % de VarCNN. En la cohorte Future-B (n = 18.543), LTD-Hybrid obtuvo 49,45 % de exactitud y 49,26 % de macro-F1, superando los valores de DF (37,46 % y 36,19 %) y VarCNN (33,45 % y 31,14 %). Las diferencias de exactitud en Future-B favorecieron a LTD-Hybrid en 11,99 y 16,01 puntos porcentuales, respectivamente; los intervalos de confianza al 95 % obtenidos mediante bootstrap pareado por bloques de fecha no incluyeron cero. Dado que los conjuntos de evaluación habían sido inspeccionados previamente y los días consecutivos pueden presentar dependencia temporal, los análisis inferenciales se interpretan como exploratorios y retrospectivos, no como confirmación prospectiva independiente.

**Nota de publicación:** completar con fuente bibliográfica exacta de cada arquitectura, tabla de presupuesto, semilla, definición de entorno Tor, procesamiento de los paquetes, auditoría cronológica y comparaciones adicionales antes de enviar el manuscrito.
