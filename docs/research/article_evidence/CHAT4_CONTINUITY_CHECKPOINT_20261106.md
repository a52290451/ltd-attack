# LTD-Attack — CHAT 4 · punto de continuidad (2026-11-06)

## Alcance
Resumen operativo de continuidad reconstruido de las conversaciones de investigación, la consola proporcionada por el investigador y los expedientes versionados de GitHub. No sustituye los artefactos experimentales originales, los manifiestos ni los hashes de Perseo.

## Objetivo científico
Evaluar website fingerprinting sobre Tor ante concept drift. Distinguir señales micro (estructura intravisita) y macro (evolución longitudinal). Comparar LTD-Micro, LTD-Macro y LTD-Hybrid con baselines SOTA usando cohortes, métricas y políticas temporales comunes; publicar resultados con incertidumbre y limitaciones explícitas.

## Contrato inmutable de evaluación
- Entrenamiento/selección: Historical DEV exclusivamente (57.916 capturas; DEV_EARLY=19.098, DEV_MIDDLE=20.160, DEV_LATE=18.658); B usa 39.258 para fit y 18.658 para validación, C refit 57.916.
- 09B: cohorte interna 14.687 capturas, 11 fechas; Future-B DEV_FROZEN: 18.543 capturas, 16 fechas; 65 clases. No mezclar Future-B DEV_FROZEN con HISTORICAL_FINAL.
- Jamás ajustar arquitecturas, pesos, umbrales o hiperparámetros mirando resultados 09B o Future-B, ni preentrenar sin etiquetas en ellos.
- Dataset C independiente, adquisición directa sin bridges en revisión; preservar cron/watchdog, raw PCAP, noise y evidencia, y vigilar sincronización NTP. Ninguna fase S7–S9 debe tocarlo.
- Conservar evaluaciones congeladas, UID, mapeos de clases, SHA256, manifests, seeds y métodos estadísticos. Ninguna ejecución debe sobrescribir artefactos anteriores.
- Inferencias S6: bootstrap pareado por bloques de fecha (2.500 remuestreos), day-swap condicional y Holm8 por cohorte; son análisis exploratorios retrospectivos, no ensayo prospectivo ciego.

## Resultado S6 ya congelado
Fuente canónica: `docs/research/article_evidence/S6_FROZEN_SOTA_LTD_RESULTS_20261106.md`.
- 09B, Accuracy %: Hybrid 97.440, Micro 95.002, Macro 84.068, DF 93.423, VarCNN 97.290.
- Future-B, Accuracy %: Hybrid 49.453, Micro 43.116, Macro 22.305, DF 37.464, VarCNN 33.447.
- Future-B: Δ(Hybrid−DF)=+11.988 pp; Δ(Hybrid−VarCNN)=+16.006 pp; Holm8 exploratorio p=0.00024414 en las ocho comparaciones.
- 09B VarCNN no muestra superioridad significativa de Hybrid bajo Holm8.
- Contrastes Hybrid vs Micro/Macro y contribución XGBoost/LTD deben provenir de sus expedientes separados (no de Holm8 S6). En auditorías previas, XGBoost resultó significativo; LTD por sí solo no fue concluyente en Top-1/MRR y perjudicó Top-5: evitar atribución causal exclusiva al LTD.

## Batería S7–S9
Fuente canónica: `docs/research/article_evidence/S7_S8_S9_EXECUTION_PLAN_20261106.md`.
- S7 CUMUL-style: 4 estadísticas + 100 puntos interpolados de bytes firmados; SVM RBF, C=1, gamma=scale, probabilidad activada, scaler fit solo train, seed42. Se llama CUMUL-style porque `len(pkt)` incluye overhead y no hay equivalencia demostrada con el paper. No usar baseline antiguo de 94 features contaminadas.
- S8 Tik-Tok: adapter DT=direction × tiempo relativo, hasta 5.000 posiciones. Arquitectura y entrenamiento pendientes.
- S9 NetCLR: exige augmentación y preentrenamiento contrastivo, seguido de fine-tuning supervisado; no sustituir por CNN/DF sin contrastive pretraining. Fuentes WFlib fijadas.

## Resultado de consola reportado en Perseo, 2026-11-06
- `git fetch/merge --ff-only` actualizado de 20f5b62 a a002f02, rama `feature/macro-v2-historical-only`.
- `test_sota_s7_s8_features_v1.py`: 6 tests OK.
- WFlib commit `537ec8e03c805540b04b95b144fa158ce50aa255`.
- TikTok: `WFlib/models/TikTok.py`, `scripts/Tik-Tok.sh`. NetCLR: `WFlib/models/NetCLR.py`, `WFlib/tools/netclr_augmentor.py`, `WFlib/tools/netclr_pretrain.py`, `scripts/NetCLR.sh`.
- PyTorch `2.11.0+cu128` operativo; faltan `sklearn` y `joblib` en `/data/ltd-sota-bsierra/venv-cu128`.
- S7 falló inmediatamente: `S7_CUMUL_ERROR ModuleNotFoundError No module named 'joblib'`. No se ha generado resultado B ni iniciado entrenamiento.
- Almacenamiento reportado: raíz 110 GB disponibles (87% uso); /data 330 GB (88% uso). Vigilar concurrencia y no borrar datos.

## Próximo procedimiento seguro
1. Instalar solo dependencias ausentes dentro de la venv explícita: `python -m pip install scikit-learn joblib`. Verificar numpy/scipy/sklearn/joblib, pip check y PyTorch GPU sin alterar el entorno CUDA.
2. Ejecutar nuevamente tests y gate S8–S9; comprobar que la carpeta S7/B no existe antes de iniciar (el script se niega a sobrescribir).
3. Ejecutar `scripts/sota/s7_cumul_frozen_v1.py --stage B` y recoger `S7_CUMUL_DEV_LATE`, `S7_CUMUL_REAL_TRAIN_PASS B`, manifest/hash. No crear fase C ni evaluar externos hasta inspección B.
4. Auditar semántica CUMUL respecto al original antes de reclamar fidelidad. Implementar S8/S9 con arquitectura y preentrenamiento genuinos y fuentes pinned; después refit congelado y evaluación pareada.

## Infraestructura
Perseo: ejecución y Future-B/C; Zeus: contraste independiente; almacenamiento no compartido. Evitar cambios a adquisición C o permisos de contenedores, reinicios y tareas cron por trabajos SOTA.

## Reproducibilidad, ética y artículo
Mantener evidencia cuantitativa favorable y desfavorable, tests, hashes, clases, fechas, exactitud, Macro-F1, Top5 y MRR. Reportar diferencias pareadas e intervalos, evitar post-hoc no declarado, documentar la dependencia de días, condiciones de captura y selección de 65 sitios, y no inferir causalidad de la deriva exclusivamente por caída entre cohortes.
