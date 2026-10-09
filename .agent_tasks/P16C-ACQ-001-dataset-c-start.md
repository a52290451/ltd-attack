# P16C-ACQ-001 — Arranque de Dataset C (LTD-Attack)

**Fecha de autorización de la tarea:** 2026-10-09  
**Tipo:** tarea MACRO operacional de adquisición y verificación, no de entrenamiento.  
**Prioridad:** ALTA: iniciar nuevas capturas Tor en cuanto se verifique el colector original y su autorización.  
**Fuente de verdad:** repositorio `a52290451/ltd-attack`, rama `feature/macro-v2-historical-only`, almacenamiento real del host de capturas.

## Contexto y límites

- Dataset C **aún no existe ni se ha iniciado**. Es una **nueva campaña** externa, independiente de Historical e Future-B; **no reutilizar ni renombrar** sus capturas.
- Ya se completó el análisis Top-1 frente a Top-5 (P15). En Future-B, Hybrid congelado: Top-1 49,45%, Top-5 80,56% sobre 18.543 capturas de 65 sitios. No intentar arreglar Top-1 seleccionando en Future-B o C.
- Las cohortes de trabajo canónicas contienen 65 sitios, pero la **lista exacta de sitios, las URL y su correspondencia** deben extraerse del inventario histórico congelado; nunca adivinarlas.
- **No tenemos documentado un comando válido de captura Dataset C ni un calendario C preacordado.** No asumir automáticamente el calendario ni cardinalidades de campañas A/B.
- En GitHub predominan scripts de análisis, entrenamiento y preprocesamiento; la ubicación del recolector PCAP/Tor de A/B todavía debe auditarse en los hosts de investigación. No asumir que está en el repo.
- El protocolo científico P16A sigue siendo borrador. **Se permite iniciar la adquisición de datos brutos tras congelar el contrato de adquisición, pero no inspeccionar C como conjunto de evaluación, puntuar modelos, usar C para tuning ni abrirlo para seleccionar hipótesis** hasta que el protocolo de evaluación externa quede preregistrado y fechado.
- Trabajar sobre máquinas y sitios autorizados por la investigación. No romper jobs ni servicios existentes. Cualquier fallo deja evidencia; prohibido borrar datos ajenos.

## Objetivo operacional de esta tarea

**Terminar con Dataset C realmente CAPTURANDO**, si hay un recolector reutilizable/validado y entorno habilitado, no solo con documentación. Una ejecución mínima (smoke) no debe confundirse con captura de producción. Si existe bloqueo real, identificarlo con evidencia y dar el único comando mínimo para resolverlo, sin informar que C ha arrancado.

## Checkpoint 0 — Auditar host y colector (solo lectura)

1. Determinar host actual (`hostname`), cuenta, ruta del repositorio, rama, espacio libre, procesos Tor existentes, utilidades instaladas (Tor/Tor Browser, tshark/tcpdump, browser/automation), permisos de captura y almacenamiento. Consultar primero Perseo y, si el flujo original vive en Zeus u otro equipo, localizar allí la captura.
2. Buscar **sin modificar** scripts, configs, unit files, crontabs, logs y manifiestos de las campañas originales A/B de recolección PCAP; extraer comando exacto de invocación, versión, flujo URL→Tor→PCAP→parsing, duración, frecuencia, retry y esquema de metadatos. Evitar recorridos de disco masivos o leer datasets completos para identificar scripts.
3. Recuperar lista congelada de los 65 sitios y sus URL de fuentes históricas preexistentes; validar que el mapping es unívoco. Dejar hashes de los ficheros usados.
4. Reportar en `.agent_results/P16C-ACQ-001-inventory.md` el inventario y una decisión reproducible sobre **qué recolector** utilizar. Si no hay acceso al host de captura, preparar un bloque de comandos read-only para que el usuario los ejecute allí; no inventar resultados.

## Checkpoint 1 — Congelar contrato de adquisición ANTES del primer PCAP

Crear `docs/protocols/P16C_DATASET_C_ACQUISITION_V1.md` (y config de colector si procede) con:
- `campaign_id` inmutable, fecha/hora UTC inicial, máquina, ruta de almacenamiento NUEVA Y EXCLUSIVA, responsable, versión Tor/Tor Browser, versión exacta del recolector, lista/URL/orden de 65 sitios y SHA256.
- Frecuencia y duración propuestas **justificadas por capacidad real y el recolector auditado**, volumen/disk budget calculados y esquema horario UTC. No copiar por inercia el número de días/horas/capturas de A/B; si no consta la decisión, documentar valor propuesto antes de iniciar y declararlo provisional para extensión sin cambiar datos pasados.
- Reglas de timeouts/retries, fallo por sitio, deduplicación, UID únicos, nombre de fichero, fecha UTC, logs de fallos, checksums y QA, sin eliminar errores fallidos del denominador de calidad.
- Política de datos: crudos PCAP inmutables, procesados derivados separados, ningún checkpoint/score de C durante captura; no sobrescribir Historical/Future-B.
- No exponer datos o capturas sensibles en GitHub; subir solo código, contrato y resúmenes no sensibles. Calcular SHA256 y versionar el contrato antes de comenzar.
- Estado del protocolo de EVALUACIÓN EXTERNA: `DRAFT_NOT_OPENED`, preregistrarlo antes de usar C para análisis.

## Checkpoint 2 — Smoke técnico separado

1. Verificar el flujo completo con pocas capturas en un namespace `dataset_c_smoke` distinto de Dataset C de producción.
2. Exigir PCAP no vacío, UID único, hora UTC, sitio válido, URL correctamente enlazada, tráfico Tor (sin bypass), logs, y preprocesamiento compatible con el esquema sin consultar etiquetas/accuracy.
3. Hacer QA automático, generar `.agent_results/P16C-ACQ-001-smoke.md`. Si falla, corregir lo mínimo y repetir sin reescribir otros datasets.

## Checkpoint 3 — Poner a correr la campaña real

Con checkpoint 0-2 en verde:
- Inicializar ruta exclusiva `~/ltd-storage/datasets/dataset_c/` (o raíz documentada equivalente si ya existe otra ubicación canónica), respetando permisos y libre espacio.
- Lanzar recolector en servicio persistente `systemd --user`, `tmux` o mecanismo canónico verificado, con reanudación segura, rotación de logs, control de tasas/carga y observabilidad.
- **Mostrar evidencia verificable del arranque**: comando de inicio realmente ejecutado, hostname, PID o unidad, archivo de configuración congelado, hora UTC, primeros PCAP válidos producidos, tasa por sitio, ruta de logs y comando de estado/parada.
- Si el agente no puede acceder al host, entregar **un único bloque listo para ejecutar** con preflight, smoke, start, status y logs; no afirmar que ya corre.
- No iniciar entrenamientos, evaluaciones ni reoptimización. No tocar la evidencia P13/P14/P15.

## Checkpoint 4 — Monitorización y entrega

- QA operativo: muestra de integridad, número de capturas completadas/intentos fallidos, 65 sitios esperados vs observados, distribución de horas/días, almacenamiento y continuidad temporal. No mirar predicciones de modelos.
- Elaborar `.agent_results/P16C-ACQ-001-final.md` con estado `RUNNING_CONFIRMED` o `BLOCKED`, pruebas, paths, riesgos y siguientes 24h.
- Evitar `git push --force`, borrados, renombrado de A/B, cambios a pesos/modelos/flags `Future-B` o cualquier selección retrospectiva usando C.

## Prioridades y forma de respuesta

EJECUTAR checkpoints encadenados sin pedir aprobaciones triviales. Resolver detalles técnicos leyendo evidencia; pedir decisión solo ante un bloqueo genuino (sitios sin URL, permisos Tor/PCAP, ausencia de colector, necesidad de privilegios, presupuesto/disco). No fabricar tiempos ni métricas de capturas. Confirmar siempre estado final real. Todos los prompts e informes en español.
