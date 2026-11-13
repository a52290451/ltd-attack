# LTD-ATTACK — puerta funcional de fidelidad por autor (13-11-2026)

## Alcance y objetivo

Tras demostrar el alineamiento canónico **S5/S6** entre nuestros modelos `LTD_MICRO`, `LTD_MACRO`, `LTD_HYBRID` y los siete SOTA, completados en las mismas 14.687 capturas de 09B y 18.543 de Future-B, esta batería comprueba que los **adaptadores de código original** conservan integridad de checkpoints, ejecutan sus redes, y que sus predicciones no están trivialmente colapsadas.

Script: `scripts/sota/audit_seven_author_functional_fidelity_v1.py`. No escribe datasets, checkpoints ni matrices NPZ originales. Genera su evidencia en `artifacts/sota/audits/` externa al árbol de entrenamientos.

### Operaciones incluidas

- Lanza en **procesos CPU aislados** las pruebas reales `integration-smoke` existentes para DF, VarCNN-direction, TikTok, RoFiRe, Swallow y WFCAT. Comprueba cálculo de forward/backprop, dos mini-actualizaciones, salida 65-way y roundtrip temporal de checkpoint; para NetCLR la prueba oficial de autor es smoke de forward/backward del encoder y las dos vistas.
- Verifica B/C, épocas seleccionadas, hashes exactos de checkpoints y puntuaciones para cada cohorte; inspecciona diversidad de la distribución `argmax` en 09B.
- Prueba equivalencia numérica, a igualdad del extractor oficial de WFCAT, de la transformación IAT adaptada en tres entradas deterministas con monotonía, regresiones y tiempos empatados.
- Documenta explícitamente las desviaciones de arquitectura, representación y entrenamiento. Se reportan fallos por modelo sin abortar las demás comprobaciones.

### Veredictos

- `PASS_AUTHOR_SOURCE_ADAPTED_FUNCTIONAL_SMOKE`: fuente del autor verificada y comprobaciones funcionales CPU, artefactos de entrenamiento y predicciones que distinguen clases. **No equivale a reproducción fiel del artículo**.
- `FAIL_COLLAPSED_BYOL_REQUIRES_NEW_RUN`: Swallow actual, cuya pérdida de diversidad ocurrió entre épocas BYOL 20 y 40; no usar sus métricas para juzgar el algoritmo del autor.
- `FAIL_OR_PENDING_FUNCTIONAL_INTEGRITY`: alguna prueba CPU, procedencia, checkpoint o diversidad ha fallado; inspeccionar error específico.
- `NOT_PAPER_EXACT_CERTIFIED`: alcance científico general mientras no se complete una paridad end-to-end de código original, configuración y entrenamiento.

### Correcciones pendientes antes de comparaciones fuertes con los artículos originales

1. Swallow: implementar un nuevo experimento versionado para comparar `target.train()` original vs el `target.eval()` adaptado, multiplicación del dataset y batch, guardrails de varianza de embeddings; corregir colapso solo usando Historical DEV. No reemplazar o editar la ejecución congelada.
2. VarCNN: se evaluó solo la **rama de dirección**, no el conjunto completo publicado. No presentarlo como VarCNN full.
3. RoFiRe: multiplicador DyWin ×10 adaptado vs ×200 del original. Evaluar cuantitativamente el impacto si se afirma fidelidad paper-exact.
4. NetCLR: diferencias declaradas de augmentación CDF, batch y pretraining (100 vs 401 parámetros del notebook). La validez como baseline adaptado no requiere fingir equivalencia literal.
5. WFCAT: probar unidades temporales desde el recolector PCAP, invariancia del adaptador frente al extractor upstream y diferencias temporales 09B↔Future-B; diferencia original 50 épocas/batch64 frente a 30/batch16.
6. DF y TikTok: comprobar que compatibilidad Keras y los hiperparámetros adaptados no cambian esenciales del modelo; reportar número de semillas.

### Evidencia ya certificada por una auditoría diferente

El cierre S5/S6 generó `CLOSEOUT_PASS POSTHOC` con 168 IC de las 21 comparaciones LTD×SOTA por cohorte y cuatro métricas. Sobre las evaluaciones retrospectivas, LTD Hybrid superó en accuracy a RoFiRe en 09B por **3,234 pp**, IC95 **[2,826, 3,569] pp**, y a VarCNN-dirección en Future-B por **4,902 pp**, IC95 **[3,469, 6,237] pp**. Estos intervalos con 11 y 16 días son condicionales, no prueba de superioridad universal.

La selección de los 65 sitios puede haber estado condicionada por Future-B históricamente según `10B_manifest_future_b.json`; no presentar Future-B como completamente ciego. No usar Future-B para nuevas decisiones de ajuste.

**No se certificará el título “supera el estado del arte original” hasta tener evidencia de métodos suficientemente equivalentes y sin colapso.**
