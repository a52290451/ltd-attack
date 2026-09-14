# Legacy and historical candidates

## Criteria

Un elemento se marca como histórico cuando pertenece a una generación anterior, duplica una implementación posterior, usa nombres de artefactos incompatibles o no tiene una ruta de resultados asociable.

Esta clasificación no implica que el código deba eliminarse. Es únicamente una etiqueta de catalogación.

## Generaciones identificadas

### Legacy-001: pipeline inicial de features

Incluye 'features_old/', el modelo Macro de 45 features, los diagnósticos con 230 features y los artefactos 'features_invariantes_seguras.txt'. Es la principal candidata para la transición aproximada 230 → 45.

### Legacy-002: primeros Transformers Micro

Incluye 'VP1_Transformers.py' y 'VP2_Transformers_directions.py', con longitudes máximas de 5.000 y 3.000, respectivamente. Utilizan nombres 'd_*' y 'ds_*' que no coinciden completamente con el pipeline actual 'ds3_*'.

### Legacy-003: primeros híbridos

Incluye 'EXP2_Train_45F3000V_Neutro.py', 'EXP3_Train_45F3000V_InclinadoF.py' y 'VF1_Trans_dir_size_feat.py'. Son precursores de 'vectores_features_v2/' y presentan diferencias en escalado, listas de features y rutas de salida.

### Legacy-004: diagnósticos duplicados

'features_old/', 'concept_drift_old/', 'VF3_*', 'GEN_*', 'VF5_*' y 'VF6_*' contienen análisis que se solapan parcialmente. Algunos resultados están embebidos en títulos o arrays hardcoded.

### Legacy-005: DML por fases

'deep_metric_learning/' constituye una línea separada de DML. Tiene figuras locales de fases anteriores, pero faltan los arrays, pesos y reportes. Además, debe resolverse la discrepancia entre la entrada de tres canales del preprocesamiento y la arquitectura documentada con dos canales.

## Duplicación técnica

- Las arquitecturas se redefinen dentro de scripts de entrenamiento y evaluación.
- Los datasets y funciones de padding se implementan varias veces.
- Hay múltiples convenciones para '45F', '94F', '230F', 'd', 'ds2' y 'ds3'.
- Las rutas relativas dependen del directorio desde el que se ejecuta cada script.
- Los resultados finales no están descritos mediante una configuración única.

## Candidatos a revisión manual

- 'Vectores-24h/'
- 'features_old/'
- 'concept_drift_old/'
- 'VP1_Transformers.py'
- 'VP2_Transformers_directions.py'
- 'VF1_Trans_dir_size_feat.py'
- 'P2_MLP_prueba.py'
- 'P3_TabNET_prueba.py'
- 'P4_Transformers_prueba.py'
- 'deep_metric_learning/faseL_ConcepDrift.py'
- 'Graficas/GEN_Resultados_Finales.py'

## Regla de conservación

No se ha eliminado ni modificado ningún elemento histórico. La reorganización física conserva cada copia mediante `git mv`; el origen y destino de cada movimiento están en `docs/migration/LOCAL_MIGRATION.csv`. Antes de reorganizar resultados o checkpoints debe localizarse en Zeus la pareja código–dataset–checkpoint–log correspondiente y confirmar si el resultado tiene valor científico o únicamente valor exploratorio.
