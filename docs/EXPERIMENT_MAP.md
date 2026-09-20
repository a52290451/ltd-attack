# LTD-Attack — Experimental Roadmap

> Documento vivo. Actualizar al cierre de cada etapa.

## Estados

| Estado | Significado |
|---|---|
| CLOSED | Ejecutado, validado y documentado |
| FROZEN | Artefacto congelado |
| ACTIVE | Linea actualmente en desarrollo |
| NEXT | Proximo experimento |
| PARKED | Rama prometedora aplazada |
| CLOSED DATA | Holdout aun no abierto |

## Flujo experimental

```mermaid
flowchart TD
A["Historical 65 sites"] --> B["DEV: EARLY / MIDDLE / LATE"]
B --> C["320 structural"]
C --> D["311 DEV-eligible"]
D --> E["D/T/P"]
E --> F["Redundancy"]
F --> G["148 representatives"]
G --> H["06B + 06B-R1"]
H --> I["FROZEN: BASE-128"]
I --> J["06C: DAY-128"]
J --> K["06D: Historical prototypes"]
K --> L["06E: Recency"]
L --> M["07A: Temporal Encoder"]
M --> N["NEXT: 07B-1 Complementarity"]
N --> O{"Complementarity?"}
O -- yes --> P["07B-2 Fusion / Reranking"]
O -- weak --> Q["Close / targeted redesign"]
P --> R["Final Architecture Freeze"]
Q --> R
R --> S["CLOSED DATA: INTERNAL_TEST"]
S --> T["CLOSED DATA: Future-B"]
T --> U["Final Results / Paper"]
I -.-> V["PARKED: DAY-128 optimized / 24h"]
A -.-> W["PARKED: HIST-WIDE / 118 sites"]
T -.-> X["PARKED: Dataset C"]
```

## Registro

| Etapa | Estado | Resultado |
|---|---|---|
| Split temporal | CLOSED | DEV_EARLY / DEV_MIDDLE / DEV_LATE; INTERNAL_TEST aislado |
| Eligibility | CLOSED | 320 -> 311 |
| D/T/P + redundancy | CLOSED | 311 -> 148 dimensiones efectivas |
| 06B + 06B-R1 | CLOSED | TOP-128 confirmado |
| BASE-128 | FROZEN | baseline single-capture principal |
| 06C | CLOSED | 2,653 perfiles DAY-128 / 41 fechas |
| 06D | CLOSED | contexto historico aporta ranking |
| 06E | CLOSED | recent-7 ayuda con mas historia, no de forma uniforme |
| 07A | CLOSED | LTD mejora ranking/Top-5, no Top-1/F1 robustamente |
| 07B-1 | NEXT | audit BASE-XGB vs LTD |
| 07B-2 | PLANNED | fusion/reranking si 07B-1 pasa el gate |
| INTERNAL_TEST | CLOSED DATA | abrir una sola vez tras freeze |
| Future-B | CLOSED DATA | benchmark externo tras INTERNAL_TEST |
| DAY-128 optimized | PARKED | rama 24h |
| HIST-WIDE / 118 sites | PARKED | extension posterior |
| Dataset C | PARKED | benchmark virgen |

## Resultados clave

### BASE-128 + XGBoost
- Fold A: Accuracy ~73.69%, Macro-F1 ~73.11%
- Fold B: Accuracy ~76.74%, Macro-F1 ~75.36%

### DAY-128
- 2,653 site-day profiles, 41 fechas, ~99.55% coverage
- Fold B: Accuracy ~66.63%, Macro-F1 ~64.28%, Top-5 ~88.64%

### 06D Historical Prototype
- Fold B single-capture: Accuracy ~22.61%, Top-5 ~59.04%, MRR ~0.392

### 06E Recent-7
Fold B vs static:
- Accuracy +3.00 pp
- Macro-F1 +2.74 pp
- Top-5 +1.55 pp
- MRR +0.027
- Mean rank 8.72 -> 7.22

### 07A Temporal Encoder
Fold A:
- BASE-MLP F1 ~60.61%, Top-5 ~88.90%
- LTD F1 ~59.74%, Top-5 ~89.20%

Fold B:
- BASE-MLP F1 ~66.88%, Top-5 ~93.23%
- LTD F1 ~66.41%, Top-5 ~95.19%

Conclusion: LTD no reemplaza BASE directamente; aporta ranking complementario.

## Camino activo

07B-1 -> 07B-2 (solo si pasa el gate) -> Final Freeze -> INTERNAL_TEST -> Future-B

## Regla de cierre

Una etapa solo se considera CLOSED cuando:
1. todos los runs declarados fueron ejecutados;
2. leakage audit paso;
3. evidencia fue guardada;
4. resultados fueron interpretados;
5. decision/status fue documentado;
6. existe commit normal, sin amend ni force.

## 07B-1 Decision Gate

Pregunta: **Cuando BASE-XGB falla, LTD contiene informacion complementaria?**

Medidas:
- both correct
- XGB-only correct
- LTD-only correct
- both wrong
- oracle union
- rescue rate
- true-class rank improvement
- Top-5 overlap
- per-class rescues

Solo pasar a 07B-2 si hay rescue pool no trivial o complementariedad clara de ranking.

## Holdout rule

DEV -> Final Architecture Freeze -> INTERNAL_TEST -> Future-B

Despues de abrir INTERNAL_TEST no se cambia feature set, arquitectura,
context window, preprocessing ni regla de fusion a partir de sus resultados.
