# LTD-Attack — Las cuatro preguntas originales: cierre de evidencia congelada

**Fecha:** 2026-10-26. **Objetivo:** discusión científica y presentación 2026-10-27.  
**Alcance:** mismo checkpoint 09A/DEV_FROZEN, 65 sitios, 09B INTERNAL_TEST (14.687 capturas / 11 fechas) y 10B FUTURE-B (18.543 / 16). Comparación por las mismas `pcap_uid` **dentro** de cada conjunto. No emparejar 09B contra 10B.

**Origen:** `FROZEN-PAIRWISE-EVIDENCE-V1` (fuentes 09B/10B SHA verificadas por Perseo, 10C original, manifiesto SHA256 `5d38ffb7d3a4ec2d6a01913019141250786e1f9af9f2a2d310e3748054f968c8`), suplemento `FROZEN-PAIRWISE-SENSITIVITY-V2`, y revisión V3 de CSV agregados (sin nuevos modelos). `scripts/diagnostics/audit_frozen_pairwise_stats.py`, `scripts/diagnostics/closeout_frozen_evidence_v2.py`.

## Matriz de respuestas

| Pregunta de investigación | Respuesta defendible |
|---|---|
| **1. ¿Hybrid supera a Micro sobre las mismas capturas?** | **Sí, en 09B y 10B DEV_FROZEN**. Top-1 +2,4375 pp (IC95 [+2,0786,+2,8104]) y +6,3366 pp (IC95 [+5,8663,+6,7497]); Macro-F1 +2,5610 / +5,1259 pp. |
| **2. ¿Hybrid supera a Macro sobre las mismas capturas?** | **Sí, en 09B y 10B DEV_FROZEN**. Top-1 +13,3724 pp (IC95 [+12,5832,+14,0975]) y +27,1477 pp (IC95 [+26,2219,+28,0356]); Macro-F1 +13,4334 / +30,9343 pp. No atribuir el efecto exclusivamente a Micro. |
| **3. ¿XGBoost mejora Macro/Hybrid frente a no utilizarlo?** | En Hybrid Future-B 10C, **sí en la ablación renormalizada**: Hybrid original vs `HYBRID_NO_XGB_RENORM` Top-1 +4,7943 pp (IC95 [+4,2763,+5,2801]), Macro-F1 +3,4715 pp ([+2,8854,+4,0168]), Top-5 +8,3158 pp ([+7,8053,+8,8937]). Macro completo supera a LTD aislado en 09B (+8,0343 pp Top-1) y 10B (+11,1363 pp), pero **Macro vs LTD es comparación de variantes, NO una ablación igualada de XGB dentro de Macro**. |
| **4. ¿LTD aporta una mejora adicional real?** | **No se demuestra beneficio incremental robusto universal.** Macro vs XGB solo Top-1 +1,5864 pp Internal 09B (IC95 [+1,0471,+2,1228]) y −1,0300 pp Future-B 10B (IC95 [−1,5447,−0,5148]); cambia de signo. En 10C Future, Hybrid original vs sin LTD renormalizado Top-1 +0,2912 pp, IC95 [−0,1569,+0,7114] y Macro-F1 +0,1538 pp, IC95 [−0,3547,+0,6445], **ambos cruzan 0**; Top-5 −0,7011 pp, IC95 [−0,9636,−0,4813]. Describir LTD como contribución condicionada al régimen, no «inútil» ni causa del +6,34 pp. |

## Evidencia por misma captura

| Conjunto y contraste | Rescatadas | Perjudicadas | Ganancia neta | Días con ΔTop1 positivo |
|---|---:|---:|---:|---:|
| 09B Hybrid−Micro | 405 | 47 | +358 | 11/11 |
| 10B Hybrid−Micro | 1.865 | 690 | +1.175 | 16/16 |
| 09B Hybrid−Macro | 2.067 | 103 | +1.964 | 11/11 |
| 10B Hybrid−Macro | 5.821 | 787 | +5.034 | 16/16 |
| 09B Macro−XGB | 599 | 366 | +233 | 11/11 |
| 10B Macro−XGB | 932 | 1.123 | −191 | 4/16 |

**Sensibilidad retrospectiva, eliminando cualquier ventana de cuatro días:** Hybrid−Micro Internal [+2,1869,+2,7262] pp, Future [+6,0709,+6,6336]; Hybrid−Macro Internal [+12,8116,+13,9383], Future [+26,6471,+27,6417]; Macro−XGB Internal [+1,0169,+2,1688], Future [−1,4032,−0,6553]. Respalda estabilidad descriptiva, **no** garantiza inferencia con serialidad.

## Comparaciones de variantes 10C — no confundir diseños

- `HYBRID_NO_XGB_RENORM` y `HYBRID_NO_LTD_RENORM`: ablaciones con eliminación + **renormalización de pesos**. Son la evidencia pertinente para preguntar el papel incremental dentro de la combinación.
- `MICRO_XGB_OUTER55` y `MICRO_LTD_OUTER55`: sustituciones de rama, **no mismos pesos/composición** que las eliminaciones renormalizadas.
- Micro+XGB sustitución Top-1 49,6953% vs Hybrid original 49,4526% (Hybrid − sustitución = −0,2427 pp, IC95 [−0,8225,+0,3321] **cruza cero**); Hybrid Macro-F1 +0,7018 pp IC95 [+0,1213,+1,2394] respecto a Micro+XGB.
- No tratar el checkpoint `HISTORICAL_FINAL` como si fuese `DEV_FROZEN`; el 10C original incluye pistas descriptivas para ambos, pero los IC V1 importados aquí son de DEV_FROZEN.

## Puntos de control y decisiones

| Checkpoint | Resultado documentado | Alcance/decisión |
|---|---|---|
| 08B | Micro Fold A 77,54% vs Fold B 86,32% Top-1 | Cambian checkpoint, período de test y entrenamiento; **no** contraste pareado de beneficio |
| 09A | Entrenamiento de referencia sobre 57.916 DEV Historical / 41 fechas | Checkpoint congelado antes del futuro |
| 09B | Internal Top-1: Micro95,00%, Macro84,07%, Hybrid97,44% | Resultado directo primario interno, n=14.687 |
| 10B | Future Top-1: Micro43,12%, XGB23,34%, LTD11,17%, Macro22,30%, Hybrid49,45% | Resultado primario futuro, n=18.543 |
| 10C | Ablaciones de XGB/LTD con pesos renormalizados, sustituciones | Exploratorio POST-HOC; no ajustar modelo con Future |
| 11L | Gate Macro FAR +0,579 pp media Macro-F1 | Promoción histórica de **Macro**, no de Hybrid |
| 12A | Hybrid media +0,2095 pp < gate +0,25 pp; un IC cruza 0 | Sin promoción |
| 13B | T3 vs U3 mismo presupuesto, FAR resultados de signos opuestos por origen | Sin beneficio consistente controlado |
| 13C | T9 usa 9 expertos vs 3 | Descriptivo; cómputo no equiparado |
| 14A/B | Rank1 9.170, ranks2–5 5.768, >Top5 3.605 en Future | Potencial oracular, no accuracy alcanzada |
| 15A-MECH | Hybrid Top-1 aumenta 18/18 comparaciones históricas; O14 Top-5 cae 9/9 | Tradeoffs explícitos |
| 15B | Arithmetic −1,1285 pp y RRF −3,2588 pp media FAR Macro-F1 vs geométrica | Ninguna promoción |
| 15D | M1−M0 Hybrid FAR Macro-F1 O14 +0,3815 pp, O28 −0,2359 pp, IC con signos opuestos | Sin promoción |
| Dataset C | bridge/obfs4 no comparable con protocolo A/B sin validar | Excluir de estas inferencias |

Los resultados históricos 11–15 proceden de `docs/research/P15_SCIENTIFIC_MILESTONE_20261012.md`, no de una ejecución nueva de sus pipelines.

## Estado de validación y límite de publicación

**Mostrable el 27-10-2026:** efectos directos RQ1/2 con n, fechas, IC y rescates-perjuicios; evidencia posterior de ablaciones RQ3/4 con etiqueta POST-HOC; inversión Macro-XGB por período, estabilidad por fechas y heterogeneidad (38 sitios ganan y 27 pierden recall en 10B, Hybrid−Micro). Tabla de decisiones negativas de las fases 11–15.

**Apta para manuscrito con reservas:** estimaciones reproducibles y discusión de sesgo. **No afirmar aptitud para someter el manuscrito** sin verificar contrato original de adquisición A/B, codificador exacto de IDs de sitios, revisión bibliográfica/novelty matrix, reproducibilidad ética y validación independiente prospectiva.

**P-values:** familia Holm12 V1 son pruebas retrospectivas después de abrir Future-B y condicionadas a intercambiabilidad de días; bootstrap de 11/16 fechas y autocorrelación posible. No son pruebas confirmatorias pre-registradas. Las pruebas exactas por intercambio V2 son exactas **solo** bajo ese mecanismo de aleatorización. `p<0.05` no prueba causalidad.

**Contexto de literatura (verificado en páginas de los editores):** Proteus (NDSS 2026, https://www.ndss-symposium.org/ndss-paper/enhancing-website-fingerprinting-attacks-against-traffic-drift/) adapta con datos de despliegue no etiquetados; Shusterman et al. (Computer Networks 2026, DOI 10.1016/j.comnet.2025.111811) estudian actualizaciones incrementales; Aion (IEEE TDSC 2026 DOI 10.1109/TDSC.2026.3723526) incluye adaptación con supervisión limitada. LTD-Attack `DEV_FROZEN` **no usa adaptación futura**. No comparar porcentajes entre estudios sin igualar entorno, fechas, threat model y presupuesto.
