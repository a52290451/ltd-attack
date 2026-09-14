# LTD-Attack Repository Catalog

Catálogo formal de datasets, pipelines, experimentos y artefactos identificados en el repositorio LTD-Attack.

## Alcance

Este catálogo se construyó mediante inspección estática del repositorio. No se ejecutaron experimentos, no se accedió a Zeus y no se modificó código experimental. Las rutas que apuntan a datasets completos, checkpoints o resultados externos se conservan como aparecen en los scripts.

Los datasets completos no están disponibles localmente. Por ello, `exists=UNKNOWN` y `environment=zeus` indican una dependencia esperada en el servidor remoto, no una comprobación de existencia.

## Nomenclatura

| Prefijo | Familia |
|---|---|
| `DATA-*` | Pipelines de datos |
| `FEAT-*` | Feature engineering |
| `MICRO-*` | Modelos Micro |
| `MACRO-*` | Modelos Macro |
| `HYB-*` | Modelos híbridos |
| `DML-*` | Deep Metric Learning |
| `SOTA-*` | Estado del arte |
| `DIAG-*` | Análisis y diagnósticos |
| `PROTO-*` | Prototipos |
| `LEGACY-*` | Código histórico |

## Estados de reproducibilidad

| Estado | Significado |
|---|---|
| `R0` | UNTRACEABLE: no hay asociación reproducible entre código, datos y resultado |
| `R1` | CODE_ONLY: existe código, pero faltan artefactos esenciales |
| `R2` | PARTIAL: existe parte de la cadena, sin evidencia completa |
| `R3` | TRACEABLE: código y artefactos principales identificables |
| `R4` | REPRODUCED: ejecución reproducida |
| `R5` | VALIDATED: resultado reproducido y validado independientemente |

Ningún experimento se marca por encima de `R2` en ausencia de checkpoint, configuración, datasets y resultado asociables. Los resultados históricos hardcoded tienen `verified=false` y `result_source=hardcoded`.

## Archivos del catálogo

- `DATASETS.csv`: fuentes de datos y representaciones.
- `PIPELINES.csv`: etapas de procesamiento y evaluación.
- `EXPERIMENTS.csv`: inventario normalizado de experimentos.
- `ARTIFACTS.csv`: checkpoints, scalers, encoders, resultados y figuras, incluidos los esperados en Zeus.
- `LEGACY.md`: generaciones históricas, duplicaciones y candidatos a código obsoleto.

## Catalog integrity rules

- experiment status usa únicamente R0-R5
- referencias a datasets usan exclusivamente IDs DS-xxx
- múltiples datasets se separan con ;
- UNKNOWN significa información todavía no determinada
- NONE significa que conceptualmente no existe ese elemento
- UNASSIGNED solo se permite para artefactos cuyo productor aún no ha sido identificado
- ningún artefacto pasa a verified=true hasta comprobar físicamente su existencia
- ningún experimento pasa a R3 hasta identificar como mínimo código, datasets, configuración relevante y artefactos necesarios para reproducirlo
