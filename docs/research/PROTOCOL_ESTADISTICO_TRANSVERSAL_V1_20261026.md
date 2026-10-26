# LTD-Attack — Protocolo estadístico transversal V1
**Fecha:** 2026-10-26. **Estado:** auditoría retrospectiva de resultados congelados; propuesta de reglas prospectivas para próximos datos independientes. **No autoriza entrenamiento, búsqueda de modelos, ni selección con Future-B.**

## A. Preguntas científicas (y afirmaciones permitidas)
- RQ1: ¿El **LTD-HYBRID-FINAL** congelado mejora frente a **MICRO-FINAL** y **MACRO-LTD-FINAL** sobre **las mismas capturas**, en INTERNAL_TEST 09B y Future-B 10B, usando el checkpoint DEV_FROZEN?
- RQ2: ¿Es mejor Hybrid que XGB o LTD individuales? ¿Macro (XGB+LTD) supera a XGB? ¿Qué ocurre al retirar LTD o XGB de Hybrid con pesos renormalizados? Las dos últimas comparaciones son **diagnósticos de componente**, y las variantes 10C de Future-B son estrictamente **post-hoc**.
- RQ3: ¿La mejora y los efectos adversos se mantienen por fecha, sitio y horizonte temporal, con incertidumbre y presupuesto computacional explícitos?
- RQ4: ¿Qué resultados de Macro P11/Hybrid P12/Micro P13 y P15 no pasan pruebas de consistencia, significancia o relevancia práctica?
- No afirmar universalmente que Hybrid «es el mejor»: es el **mejor resultado observado de la arquitectura congelada entre los controles canónicos evaluados** en los conjuntos indicados. No confundir rendimiento de la fusión con prueba causal del componente LTD.

## B. Matriz común por conjunto y checkpoint
*09B — INTERNAL_TEST*: 14,687 capturas, 65 sitios, 11 fechas. Checkpoint reentrenado por 09A con 57,916 capturas DEV (41 fechas), parámetros de arquitectura ya seleccionados; Macro pasa de 79.80% Fold B a 84.07% 09B porque **modelo y población de evaluación cambiaron**. Esta diferencia no mide por sí sola el efecto causal del refit.

*10B — Future-B principal*: 18,543 capturas, 65 sitios, 16 fechas, **DEV_FROZEN**, el **mismo checkpoint 09A** evaluado en 09B. *HISTORICAL_FINAL* es un checkpoint distinto que se reporta aparte. Nunca mezclar observaciones de 09B/10B como si fueran pares individuales.

Columna de cada reporte: dataset/track/model/checkpoint/fecha/pcap_uid/clase verdadera, n capturas, n fechas, n sitios, Top-1, Macro-F1, Top-5, MRR. Todos los contrastes de un dataset utilizan los mismos UIDs, etiquetas y fechas, alineados antes de medir.

Familia de comparación transversal principal **por dataset**: (1) Hybrid − Micro, (2) Hybrid − Macro, (3) Hybrid − XGB, en Top-1, Macro-F1, Top-5, MRR (3×4 = 12 contrastes). Estos contrastes forman una **familia de auditoría declarada ahora, posteriormente a la apertura de los datos**, y NO se describirán como pre-registrados ni como doce nuevas confirmaciones independientes.

Comparaciones secundarias: Hybrid − LTD, Macro − XGB, Macro − LTD, XGB − LTD. Las ablaciones 10C incluyen Hybrid − (Hybrid sin LTD, renormalizado), Hybrid − (Hybrid sin XGB, renormalizado), Micro+XGB y Micro+LTD por sustitución. La sustitución de la rama y la eliminación renormalizada NO tienen los mismos pesos.

## C. Cálculos obligatorios
1. **Tamaño del efecto**: Δ = métrica_candidato − métrica_referencia (puntos porcentuales para Top-1/F1/Top-5, unidades absolutas para MRR). Reportar denominadores, captura por fecha, n sitios y diferencias por clase.
2. **Tabla pareada** Top-1 por captura: ambos aciertan, solo referencia acierta (**perjuicios**), solo candidato acierta (**rescates**), ambos fallan. ΔTop1=(rescates−perjuicios)/n. Reportar matriz por sitio y fecha además del total.
3. **IC 95%** de Δ mediante bootstrap pareado por fecha, idénticos bloques para los modelos, recalculando Macro-F1 con clases originales. Semilla y 2,500+ extracciones registradas. Percentil simple es un análisis inicial: con **11/16 días** la cobertura puede ser inexacta; comparar sensibilidad por fechas y bloques contiguos.
4. **Valores p exploratorios post-hoc** mediante intercambio de las etiquetas de modelo para días completos (misma permutación para todas las capturas del día), contraste bilateral; supone intercambiabilidad de los bloques bajo la nula y puede fallar con dependencia entre días. Reportar metodología, número de intercambios, hipótesis, valor exacto/Monte Carlo y limitaciones. **Holm** para la familia principal de 12 tests por dataset; comparaciones secundarias etiquetadas exploratorias sin promoción. El porcentaje de bootstrap Δ>0 **NO** es un p-value.
5. **Robustez/heterogeneidad**: Δ por fecha y por sitio, porcentaje de sitios/días con mejora, posibles grandes perjuicios. Análisis adicional de sensibilidad frente a autocorrelación y escaso número de bloques; si se requieren p-values confirmatorios, considerar métodos inferenciales para resultados binarios emparejados con clústeres (McNemar generalizado/GEE o wild bootstrap adecuado) y sus supuestos, contrastados con experto estadístico.
6. **Reproducibilidad**: leer exclusivamente NPZ y CSV congelados de 09B/10B/10C, validar SHA256 con manifiestos, verificar conteos y conciliación de accuracy/F1/Top5/MRR archivados, crear carpeta de salida nueva (no sobrescribir); incluir script/version/seed/SHA.

## D. Criterios estándar: distinción entre evidencia y promoción
**Reglas obligatorias de evaluación para TODOS los experimentos**: comparación sobre igual población y fecha, mismo presupuesto cuando se alega superioridad de un mecanismo, hipótesis nula clara, Top1/F1/Top5/MRR, incertidumbre pareada, corrección por comparaciones múltiples si aplica, ausencia de data leakage, resultados por fecha/sitio y costo. Clasificar como: *mejora estadística descriptivamente sustentada*, *indeterminado*, *perjudica*, *evidencia mecanística*, *promoción no autorizada*.

**Propuesta PROSPECTIVA de gate técnico común para una misma familia de modelos que compita por sustituir al modelo congelado** (ratificar ANTES de disponer de un nuevo conjunto ciego): ΔMacro-F1 medio FAR ≥ +0.50 pp, positivo en los dos orígenes FAR, IC95 inferior >0 por origen, ΔTop1 ≥0 por origen, caída Top5 no mayor de 0.50 pp por origen, respeto presupuesto, y p ajustado Holm <0.05 en hipótesis primarias preespecificadas. El umbral práctico es una **propuesta a discutir**, NO el criterio previamente aplicado en todas las fases históricas. Un gate de 0.50 pp no debe confundirse con un test de equivalencia/no-inferioridad, que requiere margen propio y procedimiento estadístico adecuado.

**Retroactividad prohibida**: conservar las reglas históricas originales en 07/08/11/12/13/15; un análisis posterior más homogéneo puede reforzar o limitar su interpretación, pero no atribuirles pre-registro inexistente. Resultados de ORIGIN14/ORIGIN28 y P11–P15 reutilizan Historical; su multiplicidad real debe declararse y ningún p-value aislado los convierte en confirmación externa.

## E. Conclusiones pendientes, no completar por inferencia
- Hybrid vs Micro en 09B y Future-B: Top1 +2.44 pp y +6.3366 pp respectivamente; Future-B ΔMacro-F1 +5.1259 pp, ΔTop5 +8.1702 pp, ΔMRR +0.069147. El historial indica IC por días positivos en Future-B, **pero los límites numéricos/p-values de todos los contrastes deben extraerse, no inventarse**.
- Macro vs XGB en 09B y 10B, Hybrid vs Macro/XGB, LTD ablation 10C: calcular intervalos y, donde sea justificable, p-values sobre scores archivados alineados. 10C es post-hoc, no una nueva validación.
- 11L pasó gate **en Macro** (+0.579 pp media FAR F1, CIs positivos); 12A no lo hizo **en Hybrid** (+0.2095 pp frente a umbral original +0.25 pp, segundo IC cruza cero). 13B T3 vs U3 tiene efectos opuestos por origen; 15D M1 vs M0 Hybrid tiene efectos opuestos por origen.
- Cohorte de 65 seleccionada tras problemas de cobertura en Future-B según investigador: **B-conditioned site inclusion**. A documentar para validez externa; 120 inicial según investigador vs 118 Historical-wide en documentos requiere reconciliación.
- Dataset C piloto obfs4 fuera de comparaciones A/B.

## F. Procedimiento operativo — GitHub como fuente única (sin nuevos entrenamientos)
El auditor **ya está versionado** en [`scripts/diagnostics/audit_frozen_pairwise_stats.py`](../../scripts/diagnostics/audit_frozen_pairwise_stats.py) de `feature/macro-v2-historical-only`. Commit de incorporación: `b9f8e2f59f8fe8e8a6fe507aeb46c62286b4964a`; SHA del contenido GitHub `4b48bf4edc1725b70e01f8126edc19231403f0a5`. **No copiar scripts por /tmp ni alterar resultados congelados.**

En Perseo, desde un checkout limpio del repositorio, actualizar solo por *fast-forward* y comprobar que existe el script:
```bash
cd ~/ltd-attack
git status --short
git fetch origin feature/macro-v2-historical-only
git switch feature/macro-v2-historical-only
git merge --ff-only FETCH_HEAD
git log -1 --oneline
git ls-files scripts/diagnostics/audit_frozen_pairwise_stats.py
python3 -m py_compile scripts/diagnostics/audit_frozen_pairwise_stats.py
```
**Si hay modificaciones locales o no es posible un fast-forward, NO forzar `reset --hard` ni `git clean`**: inspeccionar antes de sincronizar.

Ejecución de solo lectura para fuentes (necesita `numpy` y `pandas` del entorno adecuado):
```bash
LTD_ENV=perseo PYTHONPATH=. python3 scripts/diagnostics/audit_frozen_pairwise_stats.py
```
Si `python3` no tiene las dependencias, activar el entorno virtual habitual de LTD-Attack y repetir el mismo comando sin cambiar argumentos de evaluación.

El programa lee `result_path("final","INTERNAL-TEST")` (09B), `result_path("final","FUTURE-B")` (10B DEV_FROZEN + 10C) y **crea únicamente** `result_path("reports","FROZEN-PAIRWISE-EVIDENCE-V1")`. Rechaza sobreescribir el directorio si existe. Comprueba SHA256 archivados (09B `outputs.scores`, 10B scores SHA avalado por 10C, outputs de 10C), población, número de fechas, coherencia de etiquetas y conciliación numérica con métricas originales. **No** abre PCAP ni CSV de captura, reentrena ni reajusta modelos/umbrales.

Evidencia generada: `00_model_metrics.csv` (5 modelos×2 periodos), `01_paired_comparisons.csv` (7 parejas×4 métricas×2 periodos, 56 filas), `02_per_day_metrics.csv`, `03_per_site_metrics.csv`, `04_existing_10C_ablation_summary.csv`, `05_existing_10C_ablation_bootstrap.csv`, `06_existing_10C_rescues.csv`, `README.md`, `manifest.json` con SHA256 de entradas/salidas. Bootstrap pareado 2,500 réplicas y permutaciones por día 9,999 por pareja, configurables; los valores p son **exploratorios posteriores**, condicionados a la intercambiabilidad de días, no pruebas independientes confirmatorias. No convertir la fracción positiva bootstrap en p-value.

**Estado:** script publicado y sintaxis validada, pero sin ejecutar sobre Perseo en este punto; solo cuando el usuario entregue `EVIDENCE_AUDIT_COMPLETE` y archivos reales se podrán completar IC/p-values y conclusiones. Tras ejecución, revisar heterogeneidad, pocos bloques y limitaciones antes de calificar evidencia.

## G. Principales referencias metodológicas externas
- MacKinnon, Nielsen & Webb, *Cluster-robust inference: A guide to empirical practice* (Journal of Econometrics, 2023), DOI 10.1016/j.jeconom.2022.04.001.
- Estudios de McNemar generalizado para resultados binarios emparejados y agrupados (GEE) como análisis de sensibilidad.
- Welch es para dos medias de grupos **independientes** con varianzas desiguales, no para predicciones emparejadas sobre los mismos PCAP.
