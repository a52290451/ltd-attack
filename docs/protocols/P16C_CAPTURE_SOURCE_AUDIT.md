# P16C — Auditoría del recolector original (09-10-2026)

**Estado:** auditoría de código terminada; Dataset C NO iniciado. Ejecución operativa en Perseo exclusivamente manual por el investigador. Referencia: TAR original `LTD_C_CAPTURE_SOURCE_REVIEW.tar.gz` facilitado por el usuario.

## Confirmaciones técnicas

- Los tres `capture_dom.js` son **idénticos byte a byte**: SHA256 `41a512e034bc79c40cdc10c397e269a373d7e418af06f0e475f40b7d978e3055`.
- `package.json` y `package-lock.json` son también idénticos en las tres instancias; el código nuevo debe ser UNO solo, parametrizado por categorías y listas de URL.
- Root Cron está configurado **dentro de cada contenedor**, no en el crontab de bsierra del host: minuto 0 y minuto 30 cada hora.
- Instancia 1: `belleza` (:00), `turismo` (:30); instancia 2: `bancos` (:00), `salud` (:30); instancia 3: `deportes` (:00), `politica` (:30).
- Cada categoría contiene 20 posiciones, pero hay **2 URL repetidas**: `www.cosmeticos24h.com` en belleza y `www.estrategiasdeinversion.com` en bancos. Total: **120 posiciones, 118 sitios/hosts únicos**. No sustituirlas unilateralmente.

## Defectos que pueden explicar las capturas inválidas de Dataset B

1. `page.goto(... waitUntil:'networkidle2', timeout:65 s)` y `process.exit(0)` sin verificar HTTP, redirección, autenticación, CAPTCHA, consentimiento ni contenido útil. La automatización puede aceptar como "correcta" una página de bloqueo.
2. `capture_dom.js` nunca pulsa opciones estándar de consentimiento de cookies ni ofrece una sesión persistente controlada.
3. Las envolturas crean `/tmp/chrome_profile_*_persistent` pero **nunca lo pasan a Chrome**, por lo que la persistencia comentada no está implementada.
4. Incrementan `last_day.txt` cada vez que Cron ejecuta un grupo, no por fecha real.
5. Eliminan mensajes de Puppeteer con `>/dev/null 2>&1`; validan superficialmente PCAP y en algunas variantes descartan registros de tcpdump.
6. `pkill -9 -f google-chrome` puede finalizar navegadores ajenos; hay riesgo de contaminación por capturas no aisladas.
7. El valor llamado `GUARD_IP` en realidad corresponde a la IP pública de **salida Tor**, no al nodo guard.

## V2 de adquisición propuesto (en revisión, no desplegado)

- Un `capture_dom.js` compartido y un `run_category.sh` parametrizado por listas externas.
- Navegación regular con `domcontentloaded` y espera acotada; metadatos completos y clasificación explícita `OK`, `BLOCKED`, `CHALLENGE`, `AUTH_REQUIRED`, `CONSENT_UNRESOLVED`, `LOW_CONTENT`, `TIMEOUT`, etc.
- `CONSENT_POLICY=observe|accept_all`: la opción de consentimiento solo pulsa controles identificados de aceptar cookies; no evita sistemas de autenticación ni protección de acceso.
- `SESSION_POLICY=fresh|per_site`: la segunda opción conserva cookies legítimas por sitio como cohorte de visitante recurrente. **No mezclar estos regímenes sin preregistro**, ya que las cookies afectan el patrón de tráfico.
- UUID por visita, día por fecha UTC, registro de fallo e integridad PCAP, `flock` para evitar solapamientos, aislamiento de procesos y almacenamiento nuevo.
- Un archivo TAR/ZIP de trabajo con código revisado y pruebas locales está disponible en la conversación de ChatGPT; **no se ha incorporado ni desplegado todavía en Perseo**.

## Decisiones del investigador (09-10-2026)

1. **No reemplazar las dos URL duplicadas.** Reproducir exactamente las 120 posiciones de las campañas A/B: 118 URL distintas y dos posiciones repetidas. Cada réplica se registra con `slot_index` y UID propios; NUNCA contar las réplicas como 120 clases/sitios únicos ni usarlas para cambiar los 65 sitios congelados de evaluación.
2. **Almacenamiento histórico confirmado en el host**, montado con bind `/data:/data` en Docker, por ejemplo `/data/bs_1site/inst2/bancos` y `/data/bs_1site/inst2/salud`. No es un almacenamiento interno recuperable borrando el contenedor: `/data` es la partición del host llena al 100%. Se requiere un volumen NUEVO y aislado en otro filesystem o ampliar capacidad; `~/ltd-storage` comparte la partición raíz con ~692 GB libres, cuya suficiencia para 60 días NO está demostrada.
3. El investigador autorizó continuar con diagnóstico de **NTP**. Se necesitan UTC y reloj sincronizado antes del primer registro científico. El smoke técnico puede ejecutarse sin ese requisito, registrando expresamente esa limitación.
4. **Control de calidad de las visitas**: cada sitio debe generar navegación real y tráfico coherente. Consentimiento de cookies estándar es admisible mediante política declarada; no eludir CAPTCHA, inicio de sesión ni controles de acceso. Medir proporción OK/BLOCKED/AUTH/TIMEOUT, PCAP legible, tamaño y número de paquetes, y revisar casos problemáticos.
5. Se preparó un **paquete local V2.1 de piloto** que permite duplicados históricos exactos, añade `capture_slot`, revisa texto de bloqueos en la respuesta DOM y limita visitas con `MAX_URLS=1` para smoke. No está aún desplegado en Perseo ni incorporado al repositorio como código de producción.

## Bloqueos obligatorios para arrancar C

- `/data` saturado al 100 %. No escribir ni borrar allí.
- `timedatectl`: `System clock synchronized: no`; confirmar NTP/UTC antes de generar series temporales.
- Duplicados resueltos metodológicamente: preservar 120 posiciones históricas (118 URL distintas), con identificador independiente de réplica; validar listas sin alterar A/B.
- Establecer destino de volumen nuevo y prueba real de Tor/Chrome/Cron por separado. Aún no se ha realizado smoke en Perseo.
- Dataset C es holdout externo virgen; nunca escoger modelos/hiperparámetros usando sus resultados.

## Auditoría del almacenamiento de Perseo (10-10-2026)

Verificada por el investigador, **sin modificación ni borrado de archivos**:
- `/data/bs_1site`: 198.840 PCAP y 484,31 GiB. Se detectaron **80.660 PCAP de 0 bytes** modificados desde 2026-05-31, todos los PCAP de ese intervalo; quedan **118.180 PCAP con contenido** por diferencia. Cualquier métrica de días útiles debe basarse en PCAP no vacíos, no simplemente en conteos de ficheros. Registrar fechas de nombre y metadatos para confirmar procedencia; los `mtime` pueden cambiar por copias.
- `/data/bs_1site`: 155.272 HTML (71,27 GiB), 155.345 JSON (~0,08 GiB) y 10.383 CSV (~0,03 GiB), a conservar durante la auditoría de errores de captura.
- `/data/bs_1site_noise`: 150.668 PCAP en 48 días de modificación, 1473,09 GiB de PCAP. **Prohibido eliminar o modificar**; investigación separada pendiente.
- `/data/tor_debug.log`: **661.703.057.408 bytes** (~616,26 GiB), propietario `messagebus`, mtime `2026-05-30T18:00:42Z`, proceso propietario no identificado en salida `fuser` sin privilegios. **Candidato de limpieza** solo tras verificar con privilegios que no esté abierto, conservar `stat` y muestras, y ejecutar `sudo truncate -s0` manualmente con aceptación explícita del carácter irreversible. No asumir que causó los PCAP vacíos sin investigación adicional.
- El usuario quiere priorizar Dataset C. El piloto V2.2 ya apunta a `~/ltd-storage/dataset-c-smoke` fuera de `/data`, por lo que puede ejecutarse ANTES de limpiar log, si Docker/Tor se comportan correctamente. **No iniciar producción** antes de NTP sincronizado y definición de filesystem aislado.

## Operación sin sudo confirmada (10-10-2026)

- En Perseo, `bsierra` no dispone de acceso `sudo` operativo (el intento de `sudo fuser` falló por autenticación); **no volver a pedir `sudo` ni cambiar propietarios/permisos** del archivo `/data/tor_debug.log` (`messagebus`). No se ejecutó `truncate` y el archivo permanece íntegro.
- Se conservaron `stat`, primera muestra de 1 MiB y última muestra de 1 MiB del log en `~/ltd-storage/audits/tor-log-20261010`. Son muestras, no copia íntegra.
- El piloto V2.2 se ejecutará sin sudo: Docker accesible al usuario y datos exclusivamente en `~/ltd-storage/dataset-c-smoke` en la partición raíz con ~692 GiB libres. Requiere solo carga del ZIP y ejecución manual. Este smoke NO requiere liberar `/data` ni NTP sincronizado; producción sí requiere diseño persistente, NTP y espacio suficiente.
- La gestión del log de 616,26 GiB se delega a administración con aprobación, verificación privilegiada de procesos activos y política de retención; no borrar o truncar por vías alternativas ni tocar `bs_1site_noise`.
