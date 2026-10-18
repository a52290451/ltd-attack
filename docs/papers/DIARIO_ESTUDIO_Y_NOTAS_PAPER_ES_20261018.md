# LTD-Attack — Diario de estudio y notas para el artículo
**Inicio:** 2026-10-18. **Modo:** reconstrucción científica desde el origen. Congelados nuevos entrenamientos y selección de modelos. Dataset C bridge/obfs4 excluido de comparación temporal A/B; detener/cuarentenar, conservar solo QA de adquisición una vez consolidado.
**Regla:** cada afirmación anotada para el paper llevará técnica, hipótesis, fuente/commit, datos, unidad estadística, estimador, IC, limitaciones, estado de validación.

## Orden de estudio
1. Pregunta de investigación original y definición del website fingerprinting/closed world, threat model, Tor y observación/captura; origen y protocolo A, cambio a B.
2. Micro base: transformación PCAP→características y tokens; dirección y magnitud/longitud; 65 clases; truncamiento; arquitectura, loss/optimizer/seeds, entrenamiento y validez de su baseline. Reconstruir selección sin hindsight.
3. Macro: 128 características agregadas XGBoost; representación LTD, procedencia de los perfiles y memoria temporal; anti-leakage; pesos y selección de fusión.
4. 10B congelado: datos, split, resultados internos vs Future-B y diferencias pareadas Micro/Hybrid con IC originales.
5. 10C diagnóstico post-hoc por componentes XGB/LTD con pesos efectivos y limites de atribución; no selección futura.
6. Fases 11–12 memoria y controles, criterio de promoción y negativos; documentar **por qué** cada hipótesis se rechazó.
7. Fases 13–15 control temporal U3/T3/T9, gaps Top5, rescates/daños, Macro M0/M1 y contraste de métricas; resultados verificados vs exploratorios.
8. Evidencia estadística completa, límites, literatura y manuscrito para PETS/PoPETs / EuroS&P / AsiaCCS.

## Notas relevantes para el paper
- **[RELEVANTE-PAPER 001 | CONFIRMATORIO PENDIENTE DE IC EXACTO]** Future-B `DEV_FROZEN`, 18,543 capturas/65 sitios: `Micro Top1=0.431160`, `Hybrid Top1=0.494526`, `Δ=+0.063366` (+6.3366 pp), `ΔMacroF1=+0.051259`, `ΔTop5=+0.081702`, `ΔMRR=+0.069147`. Comparación PAREADA sobre las mismas capturas. Intervalos day-block bootstrap informados positivos; extraer extremos originales del manifiesto antes de escribir conclusión inferencial. Fuente `docs/RESULTS_HISTORY.md`, frozen 10B.
- **[RELEVANTE-PAPER 002 | EXPLORATORIO]** 10C, ablations post-hoc Future-B: Micro+XGB 49.70%, Micro+LTD 36.23%, Hybrid sin LTD 49.16%, Hybrid sin XGB 44.66%, Hybrid original 49.45%. Sustituciones de rama y eliminaciones renormalizadas tienen coeficientes distintos; NO equivalen a un experimento causal aleatorizado. Fuente `src/models/final/10C_future_mechanistic_ablation.py`.
- **[RELEVANTE-PAPER 003 | DISEÑO ESTADÍSTICO]** El consejo del profesor sobre Welch `ttest_ind(equal_var=False)` asume GRUPOS INDEPENDIENTES. Comparar Micro vs Hybrid en las mismas trazas crea muestras apareadas, y observaciones agrupadas por fecha no son independientes. Para diferencias por captura, incluir tabla 2x2 de concordancias/discordancias y considerar McNemar (sensibilidad a dependencias); para inferencia longitudinal usar bloques por fecha y diferencias apareadas por día, con supuestos explicitados. NO aplicar Welch sobre 18,543 flags binarios como si fueran dos muestras independientes. Reportar tamaño de efecto, IC y método, no solo `p<.05`. Revisar número real de fechas y naturaleza de dependencia. Una prueba no significativa NO equivale a evidencia de igualdad.
- **[RELEVANTE-PAPER 004 | ALCANCE]** Mejora de score predictivo ≠ explicación causal. La memoria LTD NO ha sido probada responsable exclusiva de la ganancia +6.34 pp. Futuro B ya abierto: no elegir nuevos modelos/pesos/thresholds usando Future-B.
- **[RELEVANTE-PAPER 005 | DATASET C]** C bridge+obfs4 y desplazamiento temporal artificial de Tor en tres contenedores pueden crear confusión de infraestructura. Capturas C no son replicación directa de A/B. Mantener una pequeña muestra QA con procedencia y retirar campaña masiva luego de documentar conservación y consentimiento de borrado. Reinicio C Direct exige verificar contrato original A/B.

## Lección 1 prevista: Micro desde cero
**Qué buscamos en el código original:** proyecto/commit inicial, fuente de las capturas, definición exacta de `direction_vector` y `size_vector`, limpieza, escala/muestreo, max_len, perfiles de clase, arquitectura efectiva, función de pérdida, partición por fecha y por sitio, baseline de comparación, seeds, elección de hiperparámetros, curva train/val y posible leakage.
**Preguntas que responderemos:** ¿qué identifica realmente una traza? ¿Por qué Micro obtiene buen accuracy interno y se degrada en Future-B? ¿Qué cambios temporales están demostrados vs hipotéticos? ¿Qué prueba estadística corresponde a cada comparación?
**Salida por lección:** explicaciones en español, código/fragmento verificado, fórmula de métrica, ejemplo numérico, fuente exacta, `[RELEVANTE-PAPER NNN]`, afirmación permitida y afirmación no permitida.
