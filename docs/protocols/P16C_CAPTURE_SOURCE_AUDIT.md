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

## Bloqueos obligatorios para arrancar C

- `/data` saturado al 100 %. No escribir ni borrar allí.
- `timedatectl`: `System clock synchronized: no`; confirmar NTP/UTC antes de generar series temporales.
- Resolver/documentar dos duplicados y cerrar contrato de 120 sitios o 118 únicos.
- Establecer destino de volumen nuevo y prueba real de Tor/Chrome/Cron por separado. Aún no se ha realizado smoke en Perseo.
- Dataset C es holdout externo virgen; nunca escoger modelos/hiperparámetros usando sus resultados.
