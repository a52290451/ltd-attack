# Figura 2 — LTDPairScorer dentro de LTD-Macro (borrador v1)
**Fecha:** 2026-11-16. **Estado visual:** construida en SVG editable y PDF vectorial, pendiente de aprobación visual del investigador. **Figura 1:** permanece bloqueada; no se modifica.

## Título
**LTD-MACRO — Candidate-conditioned longitudinal scoring**
Objetivo: ampliar exclusivamente la ruta longitudinal `LTDPairScorer` dentro de `LTD-Macro`, sin replicar XGBoost ni Hybrid ya mostrados en la Figura 1.

## Bloques de la figura y su fuente
- **A — Query representation:** `x(t) ∈ R^128`, `QueryEncoder` con `Linear(128,64)+GELU+LayerNorm+Dropout`, y vector `q∈R^64`.
- **B — Candidate historical context:** 65 candidatos; perfiles diarios por etiqueta, solo fecha histórica `< t`, últimos `W≤5` días observados, left padding y máscara de ausencia; `CandidateTemporalEncoder` proyecta 128→64, suma posiciones sinusoidales, aplica Transformer 2 capas/4 cabezas y `masked mean` de días válidos. Produce `c_j∈R^64` para cada candidato.
- **C — Pairwise comparison:** concatenación `[q, c_j, abs(q-c_j), q*c_j]∈R^256`; `pair_head: 256→128→1`, un logit por candidato.
- **D — 65-way scoring:** logits de los 65 candidatos → softmax en cada modelo entrenado con semilla 11, 42, 73 → promedio de **probabilidades** (no de logits) → `p_LTD∈R^65`. Esta salida alimenta la fusión geométrica con XGBoost ya mostrada en Figura 1.

## Fuentes verificadas en GitHub
- `docs/research/article_evidence/FIGURE1_ARCHITECTURE_VALIDATION_20261108.md` (reconciliación de arquitectura original).
- `src/features/macro_v2/07A_candidate_conditioned_temporal_encoder.py`: `build_context_for_date`, `QueryEncoder`, `CandidateTemporalEncoder`, `LTDPairScorer.forward`. Nota: el módulo de desarrollo tiene por defecto 7 días, pero `09A_refit_final_dev_models.py` fija `CONTEXT_DAYS` con el valor final 5.
- `configs/models/MACRO-LTD-FINAL.yaml` y `configs/models/MACRO-LTD-V2.yaml`: W=5, semillas 11/42/73, ensemble `mean_softmax_probability`.
- `src/models/final/10B_evaluate_future_concept_drift.py`: comprobación de que el banco histórico congelado no se actualiza con Future-B.

## Guardarraíles
- No hacer depender la selección de historial del **label verdadero** de la consulta.
- No incluir observaciones del mismo día o posteriores a la fecha de consulta.
- No inferir que existe un `candidate identity embedding`, que la implementación descarta.
- No confundir `mean(softmax(logits_s))` con `softmax(mean(logits_s))`.
- Las celdas visuales son simbólicas, no datos reales de un sitio particular.
- Los histogramas finales de 65 sitios de la Figura 1 no se repiten.

## Artefactos
La fuente SVG y las conversiones PDF/PNG se generan en la conversación y se entregan para inspección; este registro documenta su contrato en el repositorio. Al aprobarse visualmente, incorporar el PDF mediante `figure2_longitudinal_scoring.tex` y congelar la revisión usada.

## Pendiente
Aprobación visual (espaciado, conexiones, etiquetas, terminología y densidad). No declarar la figura bloqueada antes de recibirla.
