# LTD-Attack — Plan de ejecución A/B y primer manuscrito (30-10-2026)

**Referencia de preservación:** `checkpoint/frozen-evidence-2026-10-30`, anchored at `da646e9eceeeaa036004dfc124c9e8b3c556d375`.

## A. SOTA — comparación fiel, sin desventajas artificiales

**Estado:** diseño y auditoría de origen; ningún baseline moderno nuevo ha sido entrenado ni evaluado dentro de este checkpoint.

### Selección priorizada

| Prioridad | Modelo / familia | Motivo | Protocolo |
|---|---|---|---|
| P0 | CUMUL-SVM reconstruido | Control clásico de características agregadas | **Congelado**, quitar antigua selección de 94 características informada por Future-B |
| P0 | DF original (CCS 2018) | CNN de secuencia de direcciones, código autor | **Congelado**, 5.000 posiciones si datos de origen lo permiten; no disfrazar la variante de 3.000 como original |
| P0 | Var-CNN (PETS 2019) | Red convolucional competitiva de diferente familia | **Congelado**, verificar implementación/versión |
| P1 | Tik-Tok (PETS 2020) | Control con tiempos y direcciones | **Congelado**, solo si tiempos de paquete auténticos están disponibles; de otro modo registrar inelegibilidad |
| P1 | LSTM Rimmer (2018) | Control histórico secuencial, no confundirlo con SOTA actual | **Congelado**, fidelidad y condiciones marcadas |
| Track separado | Proteus (NDSS 2026) | Adaptación al drift sin etiquetas futuras | **Adaptativo**; condiciones y tráfico objetivo explícitos |
| Track separado | Aion y otros con adaptación | Política de actualizaciones/etiquetas distinta | **Adaptativo**, sujeto a código y método verificados |

No ejecutar `scripts/train/run_estado_del_arte.sh` como evidencia canónica: entrenamiento/particiones no corresponden al protocolo actual; la SVM usa selección informada por futuro.

### Reglas congelables antes de iniciar cualquier training

1. Mismas 65 clases y mapeo de identificadores; DEV 57.916/41 fechas; INTERNAL 14.687/11; FUTURE 18.543/16. Auditoría de `pcap_uid` y sha; revisar original de adquisición A/B. Sin seleccionar subconjuntos por resultado.
2. Solo DEV para scaler, características, early stopping y selección de hiperparámetros. Cronológico, **nunca Future-B**. No premiar o castigar intencionadamente representaciones: DF 5.000 direcciones, Tik-Tok tiempos auténticos, CUMUL features originales reproducidas desde train.
3. Congelar commit upstream, dependencias, seed, presupuesto de GPU/CPU/memoria y criterio de selección; informar tanto resultados bajo presupuesto comparable como ajustes fieles a cada artículo si el cómputo difiere.
4. En cada prueba exportar 65 scores ordenados, `pcap_uid`, clase, fecha, split, modelo, SHA de checkpoint, versión del código, tiempo/costo y fallos por sitio/día. Cualquier exclusión requiere recalcular TODOS sobre el mismo conjunto intersección.
5. Reportar Top1, MacroF1, Top5, MRR y sus IC pareados; Δ contra Hybrid, p-values exactos/Monte Carlo bajo supuestos temporales y Holm predefinido. Future-B abierto -> resultados comparativos **retrospectivos**. Reservar C nuevo como test externo ciego solo si el contrato A/B resulta compatible.
6. Evaluación adaptativa por separado, documentando qué datos de C/B se usan sin etiqueta/con etiqueta, cuánto tiempo de adaptación y qué presupuesto. Nunca comparar la cifra publicada en otro dataset con nuestro test directamente.

**Salida A:** `SOTA-FAIR-COMPARISON-V1` — manifiesto, registros de configuración/datos, score files por UID, tabla completa de métricas, intervalos, pruebas, controles de coste y limitaciones.

## B. Dataset C — volver al transporte Tor directo sin destruir piloto obfs4

**Estado observado hasta ahora:** piloto histórico bridge/obfs4 en cuarentena. El 18 de octubre se informó detención de cron/watchdog y cuatro contenedores sin reinicio automático. Las 166 rondas / ~87 GB del piloto se conservan; no borrar, no reutilizar como prueba comparable. La habilitación del transporte directo fue comunicada por terceros, **no está aún verificada en Perseo**.

### B0. Inventario de solo lectura, desde Perseo

```bash
cd ~/ltd-attack
hostname; date -u '+UTC=%Y-%m-%dT%H:%M:%SZ'
timedatectl show -p NTPSynchronized -p NTP 2>/dev/null || true
df -h /data 2>/dev/null || df -h ~/ltd-storage
for n in ltd-c-tor-inst1 ltd-c-tor-inst2 ltd-c-tor-inst3 ltd-c-manual-tor; do
  docker inspect -f '{{.Name}} status={{.State.Status}} restart={{.HostConfig.RestartPolicy.Name}}' "$n" 2>/dev/null || echo "NOT_FOUND $n"
done
docker image ls --format '{{.Repository}}:{{.Tag}}' | grep -E '^ltd-c-tor-nyx:' || true
(crontab -l 2>/dev/null || true) | grep -Ei 'LTD_C_|dataset.c|run_hourly_qa|three_hourly_qa|watchdog' || true
```

**Gate B0:** NTP sincronizado, `/data` identificado, cron antiguo no activo, contenedores obfs4 sin ejecutar ni autorreiniciar, imagen disponible.

### B1. Test de transporte directo descartable (sin capturas)

El script `direct_tor_preflight_v1.sh` adjunto a la entrega permite `--inventory` (sin efectos) y `--probe` (crea **un contenedor efímero** con Tor propio en red bridge Docker y almacenamiento /tmp efímero, puerto SOCKS solo local al contenedor, `UseBridges 0`, sin `LD_PRELOAD` ni desfase de reloj, y consulta a `check.torproject.org/api/ip`). No toca los cuatro contenedores antiguos ni `/data`, no inicia cron ni guarda capturas. Requiere NTP sincronizado. Solo declara PASS si observa `Bootstrapped 100%` e `IsTor:true`. El simple hecho de iniciar Tor no basta. Este test **no demuestra equivalencia A/B**, solo transporte funcional desde ese entorno Docker.

### B2. QA científico separado

Solo después de B1 PASS: validar el colector/browser/direct Tor y contrato A/B (punto de observación de captura, BPF, versión Tor, navegador, tiempos, sesiones, políticas de circuitos y listas de URLs). Ejecutar **12 visitas QA nuevas** (2 URLs conocidas por cada una de 6 categorías de inst1 belleza/turismo, inst2 bancos/salud, inst3 deportes/política), en directorio **nuevo** `C_DIRECT_V1` sin reutilizar `dataset-c-smoke-obfs4`, guardando estado de salida/PCAP/UTC/checksums e incidencias. No ajustar modelos observando este piloto.

### B3. Captura posterior

Solo si B2 cumple QA y contrato comparable y se fija un protocolo prospectivo: 65/65 sitios, cobertura por día, tasas de fallos, SHA, límites de almacenamiento, reloj, logística y ética. Después habilitar scheduler controlado con control de duplicados; **nunca** reactivar los scripts obfs4 ni la antigua interposición temporal.

**Salida B:** `C-DIRECT-CONNECTIVITY-V1` primero, luego `C-DIRECT-SMOKE-V1` y, tras aprobación, manifiestos `C_DIRECT_V1` bajo almacenamiento científico separado.

## C. Primer artículo

Después de evidencia SOTA comparable y una verificación de transportes/contrato C: primera redacción centrada en *Frozen Hybrid WF under Temporal Drift: Paired Evaluation, Component Ablations and Failure Analysis*. Contribuciones: mejores resultados de fusión que los componentes en las mismas capturas, aporte XGB, resultado negativo/inconcluso de LTD, límites por clases/fechas, evaluación relativa a baselines fieles. Si SOTA aventaja Hybrid, **reportarlo sin ocultarlo y reencuadrar la contribución**. No proclamar novedad general ni eliminar drift. Esta es la primera publicación; la línea de memoria LTD seguirá como investigación posterior autónoma.

**Conservar documentos y artefactos V1-V4 sin modificación**, incluida presentación de 10 slides aprobada.