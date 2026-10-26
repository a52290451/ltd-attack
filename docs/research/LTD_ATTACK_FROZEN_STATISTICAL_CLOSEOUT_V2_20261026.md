# LTD-Attack — Cierre estadístico de evidencia congelada (26-10-2026)

**Proveniencia:** Perseo `FROZEN-PAIRWISE-EVIDENCE-V1`, verificación de SHA de fuentes 09B y 10B; V1 manifest SHA256 `5d38ffb7d3a4ec2d6a01913019141250786e1f9af9f2a2d310e3748054f968c8`.

**Estado:** análisis retrospectivo de scores congelados; ninguna reestimación o selección sobre Future-B.

## Preguntas y hallazgos

- **RQ1, Hybrid vs Micro:** Internal 09B 97,4399% frente a 95,0024% (Δ +2,4375 pp, IC95 día [+2,0786,+2,8104]); Future-B DEV_FROZEN 49,4526% frente a 43,1160% (Δ +6,3366 pp, IC95 [+5,8663,+6,7497]). Sobre 18.543 trazas futuras: 1.865 rescates, 690 perjuicios, +1.175 aciertos netos.
- **RQ2, mecanismo XGB/LTD:** Macro−XGB Top1 +1,5864 pp Internal, pero −1,0300 pp Future-B; 10C post-hoc Hybrid original vs sin LTD renormalizado +0,2912 pp Top1, IC [−0,1569,+0,7114] (no concluyente), y −0,7011 pp Top5. Original vs sin XGB renormalizado +4,7943 pp Top1, IC [+4,2763,+5,2801]. No atribuir causalmente +6,34 pp a LTD.
- **RQ3, heterogeneidad:** Hybrid−Micro positivo 11/11 fechas 09B y 16/16 fechas Future-B, pero en Future-B mejora recall 38/65 sitios y lo reduce 27/65. Etiqueta de sitio 90: −25,84 pp recall, −54,49 pp Top5; dominio pendiente de mapeo archivado.
- **RQ4, promoción:** 11L sí pasó gate histórico Macro; 12A, 13B y 15D no justifican promoción Hybrid. Dataset C bridge/obfs4 excluido, Future-B ya abierto. No hay demostración de eliminación de concept drift.

## Sensibilidad V2 (tablas y script local, pendientes de adjuntar aquí)

A partir exclusivamente de los CSV agregados V1 (sin leer PCAP/NPZ): al excluir cualquier tramo consecutivo de cuatro fechas, ΔTop1 Hybrid−Micro queda en [+2,1869,+2,7262] pp Internal y [+6,0709,+6,6336] pp Future-B. Exact conditional day-label swaps enumerados 2^11 y 2^16 para métricas aditivas: Future-B Hybrid−Micro p=2/65536=0,00003052 para Top1, Top5 y MRR, condicionado a intercambiabilidad de días y NO confirmatorio. Bootstrap circular en bloques de 1–4 días, 20.000 réplicas, mantiene el signo de los IC de Top1.

**Pendiente para conclusiones confirmatorias:** equivalencia de contrato adquisición A/B, mapeo canónico de sitios, sensibilidad alternativa por datos NPZ si se necesita estimar MacroF1 por sitio×fecha y nueva cohorte ciega preespecificada.

**Advertencia:** los p-values exactos son *combinatoriamente exactos bajo intercambio de etiquetas por fecha*, NO están calibrados si hay dependencia serial; los bootstrap son exploratorios con 11/16 fechas. No reinterpretar Holm9 (tres métricas aditivas) como Holm12 original con MacroF1.

Fuentes: `docs/research/PROTOCOL_ESTADISTICO_TRANSVERSAL_V1_20261026.md`, `scripts/diagnostics/audit_frozen_pairwise_stats.py`. Suplemento V2 local SHA script `326292eafa284fae8a45d6155e5651160e926aba5719dce28f87163654977dc0`.