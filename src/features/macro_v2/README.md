# MACRO-V2-HIST

Pipeline de ingeniería de características longitudinales de LTD-Attack.

## Regla fundamental

Todo el desarrollo utiliza exclusivamente Historical.

Future no puede participar en:

- generación de candidatos;
- selección de features;
- cálculo de estabilidad;
- definición de thresholds;
- ranking;
- normalización;
- arquitectura;
- hyperparameter tuning;
- checkpoint selection.

## Pipeline

00_audit_dataset.py
    Auditoría del dataset Historical y universo inicial de features.

01_temporal_split.py
    Congela el split cronológico:
    DEV_EARLY / DEV_MIDDLE / DEV_LATE / INTERNAL_TEST.

02_discriminability.py
    Mide poder discriminativo entre sitios usando únicamente DEV.

03_temporal_stability.py
    Mide estabilidad entre periodos temporales internos.

04_site_persistence.py
    Mide si cada sitio conserva individualmente su fingerprint.

05_redundancy.py
    Detecta dimensiones redundantes.

06_select_features.py
    Combina los criterios anteriores y produce MACRO-V2-HIST.

07_validate_features.py
    Evalúa por primera vez el feature set congelado sobre INTERNAL_TEST.

## External Future

El Future longitudinal real NO pertenece a esta pipeline.
Solo puede utilizarse después de congelar MACRO-V2-HIST.
