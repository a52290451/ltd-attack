# LTD-Attack — Cierre de evidencia estadística y respuestas RQ1–RQ4
**Fecha:** 26-10-2026 · **Estado:** suplemento científico retrospectivo reproducido sobre salidas congeladas V1 · **Separación estricta:** modelos y scores permanecen inmutables.

## 0. Cadena de custodia y límites
- Fuentes: `09B_internal_scores.npz` (14.687 capturas, 11 fechas, 65 sitios) y `10B_scores_DEV_FROZEN.npz` (18.543 capturas, 16 fechas, 65 sitios) verificadas por SHA en Perseo en auditoría V1. Mismo checkpoint 09A; poblaciones distintas entre conjuntos.
- Manifiesto V1 SHA256: `5d38ffb7d3a4ec2d6a01913019141250786e1f9af9f2a2d310e3748054f968c8`. Script V2 SHA256: `326292eafa284fae8a45d6155e5651160e926aba5719dce28f87163654977dc0`.
- Este suplemento lee solo **CSV agregados V1**. Verifica 8 archivos contra manifiesto, 56 comparaciones rescate/perjuicio, conciliación de fechas, sitios, Top-1, Top-5 y Macro-F1.
- No accede a PCAP ni NPZ, no entrena, calibra ni elige modelos. `FUTURE_10B_DEV_FROZEN` está abierto: hipótesis/cálculos posteriores son exploratorios.

## 1. Respuestas científicas defendibles

### RQ1. ¿Hybrid congelado supera a Micro en las mismas capturas bajo desplazamiento temporal?
**Sí, en los dos conjuntos evaluados, con mejoras pareadas positivas en las cuatro métricas.**

| Conjunto | Micro Top-1 | Hybrid Top-1 | Δ Top-1 | IC95 V1 por día | Rescates | Perjuicios |
|---|---:|---:|---:|---|---:|---:|
| 09B Internal | 95.0024% | 97.4399% | +2.4375 pp | [+2.0786,+2.8104] pp | 405 | 47 |
| 10B Future-B | 43.1160% | 49.4526% | +6.3366 pp | [+5.8663,+6.7497] pp | 1,865 | 690 |

En Future-B rescata 1.865 de los 10.548 errores iniciales de Micro (17,68%), pero perjudica 690 aciertos: ganancia neta +1.175. **No significa que el drift esté resuelto**: Hybrid pasa de 97,44% (interno) a 49,45% (futuro); esa caída no es comparación pareada entre fechas/cohortes.

| Métrica de mejora Hybrid−Micro | 09B | Future-B |
|---|---:|---:|
| accuracy | +2.4375 pp | +6.3366 pp |
| macro_f1 | +2.5610 pp | +5.1259 pp |
| top5 | +0.7966 pp | +8.1702 pp |
| mrr | +0.016750 unidades | +0.069147 unidades |

### RQ2. ¿Qué componentes se asocian con la mejora y cuál aporta LTD?
**La complementariedad XGB es más fuerte en Future-B; la aportación adicional de LTD es dependiente del régimen y la métrica. Atribución causal NO identificada.**

- **09B Macro−XGB Top-1:** +1.5864 pp; IC95 V1 [+1.0471,+2.1228] pp.
- **Future-B Macro−XGB Top-1:** -1.0300 pp; IC95 V1 [-1.5447,-0.5148] pp.
- **10C post-hoc, Future-B, Hybrid original vs sin LTD (pesos renormalizados):** Top-1 +0,2912 pp, IC95 [−0,1569,+0,7114] (cruza 0); Top-5 −0,7011 pp, IC95 [−0,9636,−0,4813].
- **10C post-hoc, Hybrid original vs sin XGB (pesos renormalizados):** Top-1 +4,7943 pp, IC95 [+4,2763,+5,2801]; Top-5 +8,3158 pp, IC95 [+7,8053,+8,8937].
- `Micro+XGB` mediante **sustitución** alcanza Top-1 49,70%, mientras Hybrid original 49,45%; diferencia −0,2427 pp (CI cruza cero). Hybrid supera la Macro-F1 de esa sustitución +0,7018 pp, CI [+0,1213,+1,2394]. **No equiparar sustitución con eliminación renormalizada.**
- En el checkpoint distinto `HISTORICAL_FINAL`, los rescates archivados implican Top-1 original 30.243% y sin LTD 27.913%, Δ descriptivo +2.330 pp. **No tenemos IC para esta comparación en los CSV V1 proporcionados.**

### RQ3. ¿La mejora se mantiene por fecha y por sitio?
**Por fecha sí dentro de los periodos observados; por sitio no es uniforme.**

| Cohorte y contraste | Fechas positivas | Sitios mejoran / empatan / empeoran Top-1 | Rango Δ al retirar 4 fechas seguidas |
|---|---:|---:|---:|
| 09B Hybrid−Micro | 11/11 | 46 / 10 / 9 | [+2.1869,+2.7262] pp |
| Future Hybrid−Micro | 16/16 | 38 / 0 / 27 | [+6.0709,+6.6336] pp |
| 09B Macro−XGB | 11/11 | 52 / 2 / 11 | [+1.0169,+2.1688] pp |
| Future Macro−XGB | 4/16 | 23 / 6 / 36 | [-1.4032,-0.6553] pp |

**Future-B por sitio:** 38 sitios ganan recall y 27 lo pierden. La suma de ganancias positivas por sitio es +1.561 aciertos frente a −386 pérdidas (balance +1.175). Son NETOS por sitio, distintos de los 1.865 rescates y 690 daños por captura.

| Etiqueta de sitio (sin dominio autenticado) | n | Δ Recall Hybrid−Micro | Cambio neto de aciertos | Δ Top-5 |
|---|---:|---:|---:|---:|
| `90` | 178 | -25.84 pp | -46 | -54.49 pp |
| `92` | 286 | -12.24 pp | -35 | -10.14 pp |
| `105` | 276 | -11.96 pp | -33 | -6.16 pp |
| `89` | 273 | -10.62 pp | -29 | -4.03 pp |
| `63` | 274 | -8.39 pp | -23 | +4.38 pp |

**NO asignar dominios a las etiquetas** sin verificar correspondencia con el `candidate_labels` y el codificador efectivo congelado; un diccionario legacy no basta.

### RQ4. ¿Qué alternativas históricas y afirmaciones deben rechazarse?
- 11L promovió su rama Macro bajo el gate histórico (+0,579 pp media FAR Macro-F1); 12A NO promovió Hybrid (+0,2095 pp frente al gate +0,25 pp y un IC que cruza cero). Un éxito parcial Macro no garantiza mejora end-to-end.
- 13B T3 frente a U3 igualando presupuesto tuvo signos opuestos por origen; 13C T9 no iguala presupuesto de expertos. No concluir que temporalidad de Micro ganó controladamente.
- 15D M1 frente M0 en Hybrid tiene signos opuestos de Macro-F1 entre ORIGIN14 y ORIGIN28. **No promoción.**
- 14A Future Top-5 rank gap (5.768 capturas en posiciones 2–5) es oportunidad de un ORACLE, no rendimiento alcanzable por un algoritmo evaluado.
- 65 sitios condicionados por cobertura en B, observación closed-world. Dataset C bridge/obfs4 es incomparble en protocolo actual y queda fuera. No extrapolar a open-world ni afirmar robustez universal.

## 2. Nuevos cálculos reproducibles V2

### 2.1 Prueba exacta condicional por día (no equivale a evidencia confirmatoria)
La auditoría V1 aproximó 9.999 intercambios mediante Monte Carlo. Ahora enumeramos exactamente todos los **2^11=2.048** patrones de signos de día para 09B y los **2^16=65.536** de Future-B, únicamente para las tres métricas aditivas (Top-1, Top-5 y MRR).

| Cohorte | Contraste | Top-1 p exacto condicional | Top-5 | MRR |
|---|---|---:|---:|---:|
| 09B | HYBRID_vs_MICRO | 0.0009766 | 0.0009766 | 0.0009766 |
| 09B | MACRO_vs_XGB | 0.0009766 | 0.0048828 | 0.0009766 |
| Future-B | HYBRID_vs_MICRO | 0.0000305 | 0.0000305 | 0.0000305 |
| Future-B | MACRO_vs_XGB | 0.0046997 | 0.0000916 | 0.0000916 |

**No confundir exactitud combinatoria con calibración estadística:** estos valores son exactos SOLAMENTE bajo intercambiabilidad de etiquetas de modelo por días. Hay dependencia serial posible. La corrección `Holm9` en `01_exact_day_label_swaps.csv` cubre 3 contrastes × 3 métricas aditivas, NO sustituye `Holm12` V1 (que contiene Macro-F1). Las comparaciones secundarias tienen p descriptivos sin ajuste confirmatorio.

### 2.2 Bootstrap por bloques consecutivos circulares — sensibilidad post-hoc
20,000 réplicas por longitud (1–4 días), muestra idéntica de días para cada pareja, bootstrap circular con reemplazo; solo métricas aditivas. IC percentil 95% EXPLORATORIO.

| Cohorte | Contraste | Longitud de bloque | IC95 Top-1 Δ (pp) |
|---|---|---:|---:|
| 09B | HYBRID_vs_MICRO | 1 día(s) | [+2.0843,+2.8028] |
| 09B | HYBRID_vs_MICRO | 2 día(s) | [+2.0668,+2.8446] |
| 09B | HYBRID_vs_MICRO | 3 día(s) | [+2.1048,+2.8742] |
| 09B | HYBRID_vs_MICRO | 4 día(s) | [+2.1038,+2.8604] |
| 09B | MACRO_vs_XGB | 1 día(s) | [+1.0422,+2.1228] |
| 09B | MACRO_vs_XGB | 2 día(s) | [+0.9057,+2.2350] |
| 09B | MACRO_vs_XGB | 3 día(s) | [+0.8324,+2.3020] |
| 09B | MACRO_vs_XGB | 4 día(s) | [+0.7876,+2.3294] |
| Future-B | HYBRID_vs_MICRO | 1 día(s) | [+5.8688,+6.7415] |
| Future-B | HYBRID_vs_MICRO | 2 día(s) | [+5.8867,+6.7581] |
| Future-B | HYBRID_vs_MICRO | 3 día(s) | [+5.8640,+6.7884] |
| Future-B | HYBRID_vs_MICRO | 4 día(s) | [+5.8797,+6.7557] |
| Future-B | MACRO_vs_XGB | 1 día(s) | [-1.5462,-0.5000] |
| Future-B | MACRO_vs_XGB | 2 día(s) | [-1.5797,-0.3552] |
| Future-B | MACRO_vs_XGB | 3 día(s) | [-1.5460,-0.3056] |
| Future-B | MACRO_vs_XGB | 4 día(s) | [-1.5232,-0.2845] |

Esos IC siguen excluyendo cero con signo coherente en los 4 largos de bloque examinados. El bootstrap circular no garantiza cobertura nominal con pocas fechas ni resuelve sesgos de recolección o selección de clases.

### 2.3 Partición cronológica descriptiva
| Cohorte | Comparación | Fechas iniciales Δ Top-1 | Fechas finales Δ Top-1 |
|---|---|---:|---:|
| 09B | HYBRID_vs_MICRO | +2.1844 pp (5 días) | +2.6712 pp (6 días) |
| 09B | MACRO_vs_XGB | +2.2411 pp (5 días) | +0.9821 pp (6 días) |
| Future | HYBRID_vs_MICRO | +5.8400 pp (8 días) | +6.7425 pp (8 días) |
| Future | MACRO_vs_XGB | -0.6236 pp (8 días) | -1.3622 pp (8 días) |

No se realizó test entre mitades: comparar dos ventanas con composiciones posiblemente distintas puede confundir tiempo, composición de sitios y variaciones de adquisición.

## 3. Preguntas todavía no identificables con el paquete V1
1. **¿Qué mecanismo causa el deterioro?** Necesita comparación del contrato original de capturas A/B (Tor, vantage, clock, filtrado, perfil, carga, sesión) y/o diseños controlados; los scores solos no identifican causas.
2. **¿LTD añade una mejora confirmatoria en un test verdaderamente nuevo?** No; Future-B ya fue abierto y 10C es post-hoc. Requiere cohorte independiente y protocolo congelado previamente.
3. **¿Qué sitios reales corresponden a IDs como `90`?** Faltan el mapeo y el SHA del codificador canónico. No asumir identificación con catálogo legacy.
4. **¿Cómo varían los rescates/perjuicios conjuntamente por sitio×día, o CIs sitio×fecha para Macro-F1?** Estos agregados marginales no incluyen la contingencia ni TP/predicción por clase×fecha; necesitaría extractor nuevo *read-only* sobre scores NPZ ya congelados, validado SHA.
5. **¿Cómo se compara exactamente con Proteus/otros métodos?** La comparación longitudinal requiere matching de amenaza, adaptación, trazas y presupuestos; los porcentajes de papers diferentes no son pruebas directas.

## 4. Recomendación de narrativa para presentación (sin construir slides todavía)
**Mensaje principal:** «Una fusión congelada puede recuperar parte de la exactitud perdida bajo drift, pero la complementariedad depende del régimen, no es uniforme entre sitios y no demuestra que la memoria LTD sea la causa de la ganancia».

Estructura de defensa: (1) problema y amenaza; (2) Historical vs Future y congelación previa; (3) arquitectura y comparadores; (4) tamaños de efecto e IC 09B/10B; (5) rescatados/perjudicados; (6) sensibilidad por fechas y sitios; (7) 10C y reversión Macro/XGB; (8) limitaciones y trabajo futuro independiente. Evitar lenguaje de «drift resuelto», «LTD aporta causalmente +6,34 pp» o «superioridad universal».

## 5. Archivos y trazabilidad
- `01_exact_day_label_swaps.csv` SHA256 `c33adb26e55105ab20c0a9b461ff47934ac6189c03d54ad8be75e1b6b5bb4528`
- `02_temporal_sensitivity.csv` SHA256 `a03298453d4b23b687c7a44d9146aa4ec022cfcd5dd86fb528a2688e58cf891f`
- `03_daily_paired_differences.csv` SHA256 `9924fbd2ee735cbf4211d947a6a7c94b973c7080fc98fce2197e4f02339dea67`
- `04_site_paired_differences.csv` SHA256 `406b8e5e1a4564a7cbf6b45ca521090889eb6fd068dce710aa8909d45235862b`
- `05_day_site_signs.csv` SHA256 `96ba7dbf62d116b752f517830c173dc0a567e89d39a3c722362ea458af09143e`
- `06_early_late_descriptive.csv` SHA256 `58884c98fa1308dd7f66720a438605ce1e212203ad0b8e557521b198399a4e61`
- Script `closeout_frozen_evidence_v2.py` SHA256 `326292eafa284fae8a45d6155e5651160e926aba5719dce28f87163654977dc0`
- Manifiesto V2 almacena parent V1 SHA y fuentes 09B/10B. Publicar scripts/tablas/reportes derivados en Git, **nunca NPZ originales, PCAP ni secretos**.
- Auditoría V1 inmutable. Toda fe de erratas de metadatos o nuevo cálculo se hace en V2/V3 versionados, no editando evidencia V1.