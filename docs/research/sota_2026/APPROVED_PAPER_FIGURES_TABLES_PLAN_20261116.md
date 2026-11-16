# LTD-ATTACK — Plan editorial y visual aprobado (2026-11-16)

**Estado:** aprobado por el responsable del artículo el 16 de noviembre de 2026.
**Regla:** no alterar la Figura 1 aprobada; continuar con Figura 2 y validar cada figura contra la implementación congelada. Esta planificación organiza el trabajo, no promueve por sí sola nuevas conclusiones científicas.

## Secciones del manuscrito
Abstract;
1 Introduction;
2 Related Work;
3 Problem Setting and Threat Model;
4 LTD-Attack Methodology (arquitectura global y mecanismo longitudinal);
5 Datasets and Chronological Protocol;
6 Experimental Design;
7 Results and Temporal Generalization;
8 Statistical Analysis;
9 Discussion and Ablation Analysis;
10 Threats to Validity and Ethics;
11 Conclusion;
Appendices.

## Figuras planificadas
1. **LTD-Attack Overall Architecture** — aprobada, bloqueada (SVG/PDF fuente original).
2. **Longitudinal Profile Construction and Candidate-Conditioned Scoring** — nueva figura; detalla exclusivamente LTDPairScorer dentro de LTD-Macro: query BASE-128, banco histórico con 65 candidatos y hasta cinco días estrictamente anteriores, left padding y máscara, query encoder 128→64, temporal transformer 128→64 (2 capas, 4 cabezas, posiciones sinusoidales), pooling válido, features de pares [q,c,|q−c|,q⊙c], MLP 256→128→1, logits 65, softmax por seed y media probabilística seeds 11/42/73; entrega p_LTD a fusión Macro de la Figura 1. No incluir XGB ni Hybrid internamente ni un embedding de identidad candidata.
3. **Chronological Training and Evaluation Protocol** — Historical DEV, EARLY/MIDDLE/LATE, selección B, refit C, 09B, Future-B.
4. **Historical and Future Performance across LTD and SOTA** — comparación de 10 modelos, 2 cohortes, métricas y pérdida descriptiva.
5. **Paired Accuracy Differences and Statistical Significance** — forest plot de 14 diferencias Hybrid−SOTA, IC 95%, p exacto Holm-14 post hoc y aviso Holm-168.
6. **Per-Date and Per-Site Generalization Analysis** — evidencia desagregada, heterogeneidad, separación por sitios/días.
7. **Component Ablation and Error-Rescue Analysis** — sin LTD, sin XGB, controles de sustitución, rescates/perjuicios con citas a evidencias de 10C.

## Tablas principales
1 Dataset and Cohort Summary;
2 Baseline Implementation and Adaptation Matrix;
3 Main Performance Comparison (10×2×4);
4 Temporal Generalization Summary;
5 Paired Statistical Inference (14);
6 Component Ablation Results.

## Suplementos
S1: 168 contrastes con p, IC95 y ajustes, procedentes de los certificados de Zeus.
S2: resultados por fecha/sitio y rescates/perjuicios.

## Reglas de trazabilidad
- No confundir comparación entre SOTA **adaptados** y reproducción paper-exact.
- No llamar prerregistrada a la familia estadística focalizada Holm-14, definida post hoc.
- No afirmar superioridad global en todas las métricas ni causalidad del drift.
- Longitudinal context estrictamente histórico con 65 candidatos; no usar etiqueta verdadera de la consulta para escoger contexto.
- XGBoost y la fusión geométrica están explicados en la Figura 1, no repetirlos en Figura 2.
- Figuras con fuentes vectoriales editables SVG/PDF y PNG para revisión; numeraciones finales sujetas a la maqueta editorial.
