# Reutilización del cruce canónico para comparar LTD con siete SOTA (13-11-2026)

## Evidencia previa localizada en el repositorio

1. `src/models/micro/08A_micro_temporal_alignment_audit.py` y `docs/evidence/MACRO-V2-HIST/08A_micro_alignment_manifest.json`: correspondencia `Micro↔Macro` **dentro del conjunto Historical**, con llave `pcap_uid`, 72.603 registros canónicos (57.916 DEV y 14.687 INTERNAL_TEST), 65 sitios y cobertura de 100% de los metadatos. La llave no constituye una identidad global entre campañas.
2. `src/models/final/10B_evaluate_future_concept_drift.py`: correspondencia `Micro↔Macro` **dentro de Future-B**, recuperando las 18.543 filas en el orden de Macro, con cotejo de `site_label`, fechas derivadas y 65 etiquetas.
3. `src/analysis/future_rank_gap_battery.py` y `.agent_results/P14A-RECOVERY-001-safe-future-uid.md`: la reconstrucción segura de `pcap_uid` de un NPZ de Future-B con `dtype=object` se hace desde el CSV Macro congelado. Nunca se resuelve reordenando arbitrariamente por UID ni recurriendo a una deserialización insegura de datos no verificados. Se coteja `query_date`, `y_true`, `candidate_labels` y población en el orden exacto. El NPZ histórico se preserva por SHA.
4. `scripts/sota/s5_freeze_and_label_alignment_v1.py`: el manifiesto S5 congela los NPZ originales de 09B/Future-B con SHA y la correspondencia exacta entre 65 clases de S1 y el orden de `candidate_labels` de cada cohorte. No se debe presuponer `identity_order`; hay permutaciones explícitas.
5. `scripts/sota/s6_compare_frozen_sota_ltd_v1.py`: ya define el contraste emparejado `LTD_MICRO` / `LTD_MACRO` / `LTD_HYBRID` frente a SOTA, con igualdad **fila por fila dentro de cada cohorte** de `pcap_uid`, `query_date`, `y_true`, `candidate_labels`, cuatro métricas y 2.500 remuestreos pareados por día.
6. `docs/evidence/FINAL/FUTURE-B/10B_manifest_future_b.json`: 18.543 capturas, 65 sitios, 2026-03-23 a 2026-04-07. Su advertencia científica es vinculante: la selección histórica de la cohorte de 65 sitios pudo estar condicionada por disponibilidad en Future-B; por ello la selección de cohortes NO es completamente ciega a Future-B. Los valores futuros no se emplearon en la selección actual del modelo, pero no se puede caracterizar como holdout prospectivo completamente virgen.

## Resultado de auditoría global ya obtenido

El cotejo de `pcap_uid` encontró 9.003 reutilizaciones DEV↔Future-B y 2.557 09B↔Future-B. El cotejo global de vectores de dirección y tiempo (después de normalizar numéricamente a µs relativos y usando SHA por secuencia completa, independiente de UID) encontró **cero** coincidencias exactas, incluso al buscar entre UIDs distintos, y ninguna fecha común. Eso corrige el error previo que exigía UID globalmente únicos. No prueba ausencia de duplicados cercanos, campañas de adquisición relacionadas ni dependencia en la definición de los 65 sitios.

## Implementación reutilizable para siete SOTA

`scripts/sota/compare_all_seven_with_original_ltd_s5_s6_v1.py` automatiza la misma comparación documentada, extendida a los siete autores:

- Lee de `S5.label_alignment[cohort].source` los **NPZ originales congelados** de 09B y Future-B, verifica SHA y extrae únicamente matrices de probabilidades seguras, fechas, verdades y etiquetas; **no accede al UID object** del NPZ.
- Contrasta `query_date`, `y_true` y `candidate_labels` fila por fila contra los siete NPZ SOTA ya alineados con el orden canónico de 09B/Future-B en la batería (y verifica sus SHA); la reconstrucción original de UID se apoya en S5/P14A y debe conservarse como evidencia de procedencia.
- Calcula Accuracy, Macro-F1, Top-5 y MRR y 2.500 remuestreos pareados por fecha para Micro/Macro/Hybrid y los siete SOTA. La Macro-F1 es reagrupada en cada réplica, no promediada incorrectamente.
- Genera un JSON independiente sin tocar checkpoints, datasets, los dos NPZ originales ni predicciones SOTA. Swallow queda marcado `DIAGNOSTIC_COLLAPSED_BASELINE`.
- No se compara directamente contra la cifra publicada de cada artículo ni se eleva una implementación adaptada a equivalencia paper-exact.

## Límites científicos vigentes

- Validación estadística emparejada de modelos propios vs adaptaciones SOTA: posible y respaldada por la población congelada si el script emite PASS.
- Certificación de fidelidad de las implementaciones a cada método fuente: pendiente de la auditoría completa de arquitectura, augmentación, entrenamiento y paridad de inferencia, en especial Swallow; no inferirla de métricas altas.
- Future-B está congelado para evaluación y no debe emplearse para seleccionar nuevos hiperparámetros o adaptaciones.
