# LTD-Attack — programa SOTA ampliado (2026-11-03)

DF y VarCNN ya están entrenados sobre 57.916 Historical DEV y evaluados sobre 09B (14.687) y Future-B DEV_FROZEN (18.543). S6 cuantificará su comparación con LTD-Hybrid/Micro/Macro; esto NO cierra la batería SOTA.

## Dos tracks separados

**FROZEN / closed-world single-tab:** entrenar y seleccionar solo en Historical DEV, congelar antes de 09B y Future-B. Exportar mismo UID y orden de 65 clases, probabilidades completas, Accuracy, Macro-F1, Top-5 y MRR. Preservar condiciones y reportar diferencias de modalidad (dirección, tiempo, tamaño), GPU y presupuesto. Future-B ya observado: análisis retrospectivo, no confirmación ciega.

**ADAPTATIVE / drift-aware:** tráfico futuro no etiquetado en Proteus; etiquetas target limitadas en Aion. Estos métodos tienen información distinta; reportarlos en otra tabla con cronología, cantidad de muestras target y etiquetas, no mezclarlos como controles frozen.

## Modelos pendientes y decisión ejecutiva

| Prioridad | Modelo | Estado actual | Próxima implementación | Precaución |
|---|---|---|---|---|
| P1 | CUMUL/SVM-RBF (2016) | NO realizado de forma fiel | Reproducir features CUMUL originales desde dirección/tamaño histórico, escalar y ajustar C/gamma solo en DEV cronológico | Prohibido el antiguo vector 94F seleccionado con futuro |
| P1 | Tik-Tok (2020) | NO ejecutado | Código y commit originales, directional timing DT de 5000 paquetes y entrenamiento DEV | DT no es DT2: no reciclar el adaptador VarCNN sin adaptación explícita |
| P2 | NetCLR (2023) | NO ejecutado | Verificar fuente, augmentations, pares contrastivos y presupuesto pretraining solo DEV | Sin unlabeled future ni test para seleccionar |
| P2 | AWF/DLWF | NO ejecutado | Auditar arquitectura PyTorch WFlib y presupuesto | Evitar comparaciones redundantes |
| P2 | Triplet Fingerprinting | NO ejecutado | Protocolizar few-shot N-shot y etiquetas 65 | Diferente objetivo: comparación específica |
| P2 | Robust Fingerprinting (RF) | NO ejecutado | Auditar representación y robustez por defensa | Declarar si usa aumentación o datos externos |
| P2 | Holmes | NO ejecutado | Auditar enfoque early-stage/tiempos | Si usa primeras celdas, distinta tarea y presupuesto |
| P3 | Rimmer LSTM heredado | NO validado | Corregir split aleatorio, longitud de secuencia y fidelidad | No usar los resultados legacy como SOTA canónico |
| P3 (adaptativo) | Proteus NDSS 2026 | Código público localizado | Track con adaptación mediante target no etiquetado | No comparable directamente con frozen |
| P3 (adaptativo) | Aion | Fuente implementación pendiente | Comprobar acceso a código y presupuesto de etiquetas | Target etiquetado requiere track específico |

Modelos multi-tab y open-world como BAPM, ARES y TMWF se dejan fuera del cuadro closed-world single-tab salvo implementación equiparable a la misma amenaza.

## Ejecución secuencial

1. **S6**: ejecución real de scripts/sota/s6_compare_frozen_sota_ltd_v1.py. Verifica hashes S5, empareja exactamente UID, fecha y clase, calcula 5 modelos x 2 cohortes y contraste post hoc Holm8 (DF y VarCNN × cuatro métricas) vs Hybrid, bootstrap pareado de 2500 días remuestreados e intercambio condicional de bloques. No entrena.
2. **S7 CUMUL**: implementar extractor fiel, SVM-RBF con selección exclusiva DEV Early/Middle/Late, refit, checkpoint hash y exportación 09B/10B.
3. **S8 Tik-Tok**: fijar código fuente y representación DT, congelar semillas/presupuesto, entrenar/evaluar.
4. **S9 NetCLR**: pretraining estricto Historical y seguimiento de augmentations/coste; integrar al reporte extensible.
5. **S10 Proteus/Aion**: evaluación adaptación separada, cuando estén terminados los controles frozen.

La familia de inferencia Holm8 actual no se debe redefinir después de observar resultados de métodos nuevos. Para nuevos baselines, declarar una familia exploratoria adicional con sus propias hipótesis y corrección.

## Fuentes técnicas

- CUMUL: https://www.cs.sfu.ca/~taowang/wf/attacks/ ; reimplementación https://github.com/lsvih/CUMUL
- Tik-Tok: https://github.com/msrocean/Tik_Tok
- WFlib PyTorch (catálogo): https://github.com/FIND-Lab/Website-Fingerprinting-Library ; commit ya fijado para DF/VarCNN: 537ec8e03c805540b04b95b144fa158ce50aa255
- Proteus: https://github.com/Xinhao-Deng/Adaptive-WF-Attack

No copiar accuracy de artículos ajenos sobre datasets distintos. No instalar nuevos entornos ni modificar Dataset C. La degradación 09B→Future-B de DF/VarCNN es descriptiva y puede involucrar covariables de captura: S6 mide diferencias pareadas pero no identifica por sí solo la causa.
