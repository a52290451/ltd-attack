# P13-B1 — Experimental Battery Checkpoint

Estado del documento: protocolo y batería predeclarados; pendiente de
ejecución en Zeus. Este documento no contiene resultados nuevos de 13B, 13C
ni 14A.

## 1. Protocolo congelado

La batería contiene 13B, 13C y 14A y ejecuta todos los candidatos fijados
antes de producir cualquier informe de decisión. Historical y ORIGIN14/28
son desarrollo repetido. Future-B es OPEN/FROZEN y se usa solo como análisis
diagnóstico post-hoc.

No hay continual learning, adaptación en inferencia, actualizaciones de
memoria, pseudo-label updates, selección adaptativa ni búsqueda de pesos.
Accuracy y Macro-F1 son métricas co-principales.

## 2. Integridad / hashes

La configuración, el código, los inputs y los outputs se registran en los
manifests JSON mediante SHA256. Cada cache de entrenamiento exige coincidencia
de configuración, código e inputs antes de reutilizarse.

## 3. 13B — ensemble genérico frente a ensemble temporal

La matriz fija contiene ORIGIN14/ORIGIN28 × NEAR/MID/FAR y 18 entrenamientos:
EARLY/UNIFORM/LATE × seeds 11/42/73 por origin. Se derivan U1, U3, T3_S11,
T3_S42, T3_S73 y T9 mediante media geométrica de probabilidades.

## 4. Accuracy y Macro-F1 lado a lado

El runner escribirá `13B_model_summary.csv` con Accuracy, Macro-F1, Top-5,
MRR y mean true rank para cada origin, ventana y candidato. No hay valores
numéricos que reportar hasta la ejecución Zeus.

## 5. Replicación seeds 11/42/73

T3_S11, T3_S42 y T3_S73 se reportan separadamente. La comparación primaria
predeclarada es T3_S42 frente a U3; las otras dos replicaciones son
descriptivas y no crean un umbral posterior.

## 6. T9 como exploración secundaria

T9 combina los nueve expertos. Se reportará como exploratorio y no se usará
para afirmar causalidad temporal, porque tiene más modelos que U3.

## 7. 13C — factorial Hybrid completo

Se evaluarán U1, U3, T3_S42 y T9 contra M0 y M1 en los ocho productos fijos.
M0 es UNIFORM XGB + RECENT5 LTD; M1 es TEMPORAL_SYMMETRIC3 XGB +
MULTISCALE5 LTD. La fusión es siempre Micro=0.45 y Macro=0.55.

## 8. Historical rank gap

Para U1, U3, T3_S42, T9, el Hybrid canónico y las combinaciones robustas se
reportarán TOP1_CORRECT, TOP5_RECOVERABLE y TOP5_MISSED, junto con count,
fraction, confianza top-1, margen top1-top2, entropía y probabilidad de la
clase verdadera.

## 9. 14A — Future-B rank gap

14A leerá exclusivamente `10B_scores_DEV_FROZEN.npz`. No entrena, no cambia
modelos, no selecciona candidatos y no reejecuta 10B. Medirá la distribución
exacta de ranks para Micro, Macro XGB, Macro LTD, Macro final e Hybrid final.

## 10. Diagnósticos por sitio

Se producirán support, Accuracy, Top-5, Top5-minus-Top1, MRR, mean rank y
mediana de rank para los 65 sitios. Las categorías descriptivas se fijan
antes del cálculo: `accuracy >= 0.75` implica `high_top1`; si no, una brecha
Top5-minus-Top1 de al menos 0.25 implica `recoverable_gap`; el resto es
`lost_beyond_top5`.

## 11. Diagnósticos de tamaño y truncación

Se analizará la longitud original de direction_vector/size_vector, bytes
totales cuando sean seguros, mediana, p90, p95, fracción >=3000 y fracción
>3000, cruzadas con Top-1 correcto, recuperable y perdido. MAX_LEN permanece
en 3000.

## 12. Similitud de drift / grafo de confusión

Se calcularán centroides BASE-128 Historical/Future escalados solo con
Historical, desplazamientos estandarizados, similitud coseno y vecinos de
drift. El grafo Hybrid final agregará aristas true_site → predicted_site con
count, fracción, rank verdadero y presencia en Top-5.

## 13. Component rescue

Para errores del Hybrid se conservará si la etiqueta verdadera estaba en
Top-1, Top-3 o Top-5 de Micro, XGB, LTD y Macro final. No se entrenará un
reranker con Future-B.

## 14. Gates predeclarados

El gate primario es T3_S42 frente a U3: Macro-F1 FAR positivo en ambos
origins, gain medio FAR >= +0.005, gain medio FAR de Accuracy > 0 y fracción
bootstrap positiva de Macro-F1 >= 0.90 en ambos origins. Las replicaciones
S11/S73 se reportan descriptivamente. `fraction_delta_gt_0` nunca se llama
valor p.

## 15. Ramas que pasan / fallan descriptivamente

13A ya está archivado con `promotion_gate passed` y `Future-B unused`; su
Accuracy FAR documentada es 72.79% → 75.97% en ORIGIN14 y 78.97% → 87.76%
en ORIGIN28, con gain medio aproximado de +5.99 pp. 13B, 13C y 14A quedan
sin clasificación hasta ejecutarse en Zeus.

## 16. Preguntas abiertas para el siguiente checkpoint

- ¿El beneficio de T3_S42 sobre U3 persiste en Accuracy y Macro-F1?
- ¿Las replicaciones S11/S73 mantienen la dirección del efecto?
- ¿Cómo cambia el gap de rank al integrar M0/M1?
- ¿Los errores recuperables se concentran en capturas largas o en sitios con
  mayor desplazamiento de drift?
- ¿Qué evidencia adicional, si alguna, justifica la siguiente fase sin abrir
  Future-B para selección ni abrir Dataset C?
