# P16C-ACQ-001 — Arranque de Dataset C (LTD-Attack)

**Fecha de autorización de la tarea:** 2026-10-09  
**Tipo:** tarea MACRO operacional de adquisición y verificación, no de entrenamiento.  
**Prioridad:** ALTA: iniciar nuevas capturas Tor en cuanto se verifique el colector original y su autorización.  
**Ejecución en Perseo:** EXCLUSIVAMENTE manual por bsierra; ChatGPT/Codex no tienen autorización de ejecución remota en Perseo. El asistente edita código pequeño únicamente en GitHub y proporciona instrucciones revisables para la ejecución manual.  
**Fuente de verdad:** repositorio `a52290451/ltd-attack`, rama `feature/macro-v2-historical-only`, almacenamiento real del host de capturas.

## Contexto y límites

- Dataset C **aún no existe ni se ha iniciado**. Es una **nueva campaña** externa, independiente de Historical e Future-B; **no reutilizar ni renombrar** sus capturas.
- Ya se completó el análisis Top-1 frente a Top-5 (P15). En Future-B, Hybrid congelado: Top-1 49,45%, Top-5 80,56% sobre 18.543 capturas de 65 sitios. No intentar arreglar Top-1 seleccionando en Future-B o C.
- **Universo de recolección confirmado por el investigador:** 120 sitios web distribuidos en seis categorías de 20. Existen tres contenedores de captura, con 40 sitios / dos categorías cada uno. **No confundir los 120 sitios de recolección con la cohorte congelada de evaluación de 65 sitios.** Recuperar lista exacta, URLs, categorías y asignación contenedor-sitio de los contenedores originales; preservar trazabilidad de 120→65 sin usar Dataset C para escoger clases.
- **Calendario operativo confirmado:** en cada uno de los tres contenedores, una categoría de 20 sitios comienza cada hora a minuto `:00` y la otra a `:30`, las 24 horas. Confirmar zona horaria, orden de sitios, duración de cada visita y que 20 sitios se completen dentro de sus 30 minutos; si no, impedir solapamientos. El número de días de C no está aún confirmado. La tasa teórica es 120 capturas/hora y 2.880/día **solo si se produce exactamente una captura válida por sitio en cada turno**.
- **Contenedores fuente identificados en Perseo:** `inst1_1site` (`06c7ded30d35`), `inst2_1site` (`13a2a63e885b`), `inst3_1site` (`dc4bfda535ff`), todos basados en `tor_1site_noise:final` (`02cc4a1fe754`) y todos detenidos con exit 137. Montaje existente compartido `/data:/data`, **no montar esta ruta en escritura para Dataset C** antes de auditarla. Recuperar código con `docker inspect` y `docker cp` de contenedores detenidos; no arrancarlos a ciegas ni inferir que 137 significa OOM sin comprobar `State.OOMKilled`.
- El protocolo científico P16A sigue siendo borrador. **Se permite iniciar la adquisición de datos brutos tras congelar el contrato de adquisición, pero no inspeccionar C como conjunto de evaluación, puntuar modelos, usar C para tuning ni abrirlo para seleccionar hipótesis** hasta que el protocolo de evaluación externa quede preregistrado y fechado.
- Trabajar sobre máquinas y sitios autorizados por la investigación. No romper jobs ni servicios existentes. Cualquier fallo deja evidencia; prohibido borrar datos ajenos.

## Objetivo operacional de esta tarea

**Terminar con Dataset C realmente CAPTURANDO**, si hay un recolector reutilizable/validado y entorno habilitado, no solo con documentación. Una ejecución mínima (smoke) no debe confundirse con captura de producción. Si existe bloqueo real, identificarlo con evidencia y dar el único comando mínimo para resolverlo, sin informar que C ha arrancado.

## Checkpoint 0 — Auditar host y colector (solo lectura)

1. Determinar host actual (`hostname`), cuenta, ruta del repositorio, rama, espacio libre, procesos Tor existentes, utilidades instaladas (Tor/Tor Browser, tshark/tcpdump, browser/automation), permisos de captura y almacenamiento. Consultar primero Perseo y, si el flujo original vive en Zeus u otro equipo, localizar allí la captura.
2. Inspeccionar **sin iniciar** los tres contenedores `inst1_1site`, `inst2_1site`, `inst3_1site`: `docker inspect` (sin credenciales), `docker logs`, rutas de programa y `docker diff`. Usar `docker cp` sobre contenedores detenidos para recuperar código de aplicación, dependencias, Dockerfile si existe y config sanitizada. Buscar scripts/configs/logs originales de A/B y extraer flujo URL→Tor→PCAP→parsing, duración de 20 sitios por turno, timeouts, retry, circuitos, contaminación por tráfico ajeno, esquema de metadatos y causa real de exit 137. Evitar recorridos de disco masivos o leer datasets completos para identificar scripts.
3. Recuperar lista de **120 sitios y seis categorías** con asignación exacta de 40 sitios por contenedor de los inventarios/configs existentes; validar URL y mapping únicos. Recuperar aparte la cohorte congelada de **65 sitios usada solo en evaluación** y documentar la relación 120→65 sin cambiar decisiones congeladas. Dejar SHA256 de las fuentes.
4. Reportar en `.agent_results/P16C-ACQ-001-inventory.md` el inventario y una decisión reproducible sobre **qué recolector** utilizar. La ejecución y obtención de evidencias de Perseo la realizará el usuario manualmente; el asistente no debe asumir acceso SSH, docker exec ni ejecución de agentes en ese host.

## Checkpoint 1 — Congelar contrato de adquisición ANTES del primer PCAP

Crear `docs/protocols/P16C_DATASET_C_ACQUISITION_V1.md` (y config de colector si procede) con:
- `campaign_id` inmutable, fecha/hora UTC inicial, máquina, ruta de almacenamiento NUEVA Y EXCLUSIVA, responsable, versión Tor/Tor Browser, versión exacta del recolector, **120 sitios, seis categorías, asignación 40 sitios/2 categorías por cada uno de los tres contenedores**, lista/URL/orden de ejecución y SHA256. Guardar mapping independiente de la cohorte evaluada de 65.
- Horarios obligatorios **cada hora: categoría A a `:00`, categoría B a `:30` por contenedor**, de forma 24/7. Verificar capacidad real de 20 visitas/30 minutos, con separación de circuitos/identidad Tor si corresponde y política de no solapamiento. Calcular presupuesto de disco, CPU/memoria, cuotas y carga. La **duración en días** requiere decisión previa y no se hereda sin validación de A/B; registrar propuesta y congelarla antes del primer PCAP de producción.
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
- El asistente **no puede acceder remotamente a Perseo**: entregar bloques de comandos manuales por checkpoints (inventario, extracción de código, smoke, start, status y logs) y validar cada salida antes de continuar; no afirmar que ya corre.
- No iniciar entrenamientos, evaluaciones ni reoptimización. No tocar la evidencia P13/P14/P15.

## Checkpoint 4 — Monitorización y entrega

- QA operativo: muestra de integridad, intentos completados/fallidos, **120 sitios esperados agrupados en 6 categorías y 3 contenedores**, cobertura de turnos `:00`/`:30`, distribución de horas/días, almacenamiento y continuidad temporal. Reconciliar aparte la cohorte futura de evaluación de 65 sin examinar predicciones de modelos.
- Elaborar `.agent_results/P16C-ACQ-001-final.md` con estado `RUNNING_CONFIRMED` o `BLOCKED`, pruebas, paths, riesgos y siguientes 24h.
- Evitar `git push --force`, borrados, renombrado de A/B, cambios a pesos/modelos/flags `Future-B` o cualquier selección retrospectiva usando C.

## Prioridades y forma de respuesta

EL USUARIO EJECUTA los checkpoints de Perseo manualmente; el asistente analiza los resultados y hace ajustes pequeños directamente en GitHub. Resolver detalles técnicos leyendo evidencia; pedir decisión solo ante un bloqueo genuino (sitios sin URL, permisos Tor/PCAP, ausencia de colector, necesidad de privilegios, presupuesto/disco). No fabricar tiempos ni métricas de capturas. Confirmar siempre estado final real. Todos los prompts e informes en español.
