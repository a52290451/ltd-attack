# Auditoría metodológica de implementaciones SOTA adaptadas a LTD

Estado: PRELIMINAR — no certificadas como reproducciones exactas. Base: adaptadores de feature/macro-v2-historical-only.

## Alcance
Un commit original fijado y un checkpoint SHA válido prueban trazabilidad técnica, NO equivalencia de método publicado. Los experimentos son comparaciones de implementaciones derivadas del código de los autores, adaptadas a 65 clases, Historical DEV cronológico y evaluación 09B/Future-B.

## Desviaciones materiales registradas
| Método | Fuente utilizada | Desviación metodológica |
|---|---|---|
| NetCLR | Backbone, entrenamiento contrastivo y aumentos de notebook fijado | 100 épocas/batch 64; CDF de bursts corregida; cabeza de 65 clases y C refit |
| Swallow | CIF, ResNet18, MLPHead y NetAugment originales | Pretrain 100 frente a 200, batch 32 frente a 512, patch al bucle de aggregation, configuración de fine tune |
| WFCAT | feature_transform y WFCAT upstream | IAT histograma 9 bins, ventana 0.044, secuencia 1800, 65 clases, protocolo DEV B/C |
| DF | DFNet fuente original AST | Keras actual, entrada dirección 5000, 65 clases, hiperparámetros adaptados |
| VarCNN | ResNet18 rama dirección fuente original | No es el ensemble completo direction/timing/metadata |
| TikTok | ConvNet original | Keras actual, representación DT y entrenamiento 65 clases adaptados |
| RoFiRe | DFVarNet, RMSNorm, DyWin fuente original | Aumentos DyWin x10 frente a x200 original, representación/selección adaptadas |

## Diagnósticos prioritarios
1. Swallow: verificar DEV_LATE B y sus 30 épocas, número de clases predichas, entropía, distribución top-1, resultados 09B/Future-B; si B también está próximo a 1/65, drift externo no explica el fallo de aprendizaje histórico.
2. WFCAT: con 94% en 09B y 3.7% en Future-B, revisar paridad de unidades time_vector, monotonía de timestamps, duración, ventana IAT, suma/saturación histograma, fracción de ceros y cambios de población en ambos períodos. Nunca seleccionar correcciones con Future-B congelado.
3. Independencia: recalcular métricas desde NPZ, verificar UID, fechas, etiquetas, class order e integridad SHA entre métodos sin importar su evaluador.
4. Validar fuente/equivalencia: arquitectura, entrada, pérdida, optimizador, scheduler, batch, aumentos, init, fine tuning, refit y checkpoints efectivos, por contraste con el código de autores.
5. Leakage: demostrar mediante trazas de acceso y flujo de código que Future-B/09B no se usaron para seleccionar modelos. Booleans de manifests son evidencia de una aserción, no prueba forense independiente.

## Criterio de comparabilidad
Una comparación de estas adaptaciones es válida como benchmark bajo protocolo fijo si se verifican filas UID, clases, fechas, particiones, métricas, anti-leakage y desviaciones de cada método; no equivale a reproducir las cifras de los artículos. Especificar input modality direction-only vs timing, y clasificar explícitamente el alcance. Para inferencia, bootstrap pareado por fecha (2500 muestras) conforme a protocolo LTD, tras pasar integrity/parity gates.

## Procedimiento seguro durante GPU
El script de auditoría en scripts/sota/audit_existing_original_sota_readonly_v1.py solo lee artefactos, no usa GPU. No hacer merge/pull sobre working tree de Zeus si la batería sigue activa; hacer fetch y extraer la versión con git show a una ubicación temporal /data, y ejecutarla allí. Nunca alterar los artefactos originales ni retocar parámetros según Future-B.