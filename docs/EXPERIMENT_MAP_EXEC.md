# LTD-Attack — Mapa ejecutivo de la investigación

> **Estado canónico de trabajo — 2026-10-04**
>
> Resumen visual de la trayectoria experimental desde la auditoría/reconstrucción hasta Phase 13.  
> Ver también: `EXPERIMENT_MAP.md` y `catalog/RESEARCH_BRANCHES.csv`.

## Leyenda

| Estado | Significado |
|---|---|
| ✅ CLOSED | Ejecutado, interpretado y cerrado |
| 🧊 FROZEN | Artefacto/configuración congelada |
| 🟢 POSITIVE | Señal útil, aunque no necesariamente promovida |
| ❌ REJECTED | Hipótesis descartada |
| 🧪 DIAGNOSTIC | Explica mecanismo; no es solución desplegable |
| 🚧 NEXT | Siguiente experimento |
| 🅿️ PARKED | Línea aplazada |
| 🔓 OPEN/FROZEN | Holdout ya observado; no puede usarse para selección nueva |

## Vista ejecutiva

```mermaid
flowchart TD
    A["Auditoría del código y resultados legacy"] --> B["Contaminación metodológica detectada<br/>45F/94F future-informed<br/>resultados no canónicos"]
    B --> C["Reconstrucción Historical-only<br/>65 sitios · 72,603 capturas · 52 días"]
    C --> D["DEV 57,916 / 41 días<br/>Internal Test 14,687 / 11 días"]

    D --> E["Macro feature engineering<br/>320 → 311 → 148 → BASE-128"]
    E --> F["Macro base<br/>XGBoost + LTD Transformer W=5<br/>Fold-B Acc 79.80%"]
    D --> G["Micro limpio<br/>CNN + Transformer<br/>Fold-B Acc 86.32%"]
    F --> H["Hybrid base<br/>Micro 45% + Macro 55%<br/>Fold-B Acc 92.05%"]
    G --> H

    H --> I["Internal Test<br/>Micro 95.00% · Macro 84.07% · Hybrid 97.44%"]
    I --> J["Future-B same checkpoint<br/>Micro 43.12% · Macro 22.30% · Hybrid 49.45%<br/>Hybrid Top-5 80.56%"]
    J --> K["Concept Drift confirmado<br/>mitigado, no resuelto"]

    K --> L["Phase 11: robustez Macro"]
    L --> L1["❌ 11A Stable features"]
    L --> L2["🟢 11B Multiscale memory"]
    L --> L3["❌ 11C Invariance"]
    L --> L4["❌ 11D Prototypes"]
    L --> L5["❌ 11E Trajectories"]
    L --> L6["🧪 11F Oracle refresh"]
    L --> L7["❌ 11G Pseudo-refresh"]
    L --> L8["🟢 11H Temporal XGB experts"]
    L --> L9["❌ 11I–11J Reweighting / DRO"]
    L --> L10["🟢 11K Arithmetic mixture"]
    L2 --> M["✅ 11L Robust Macro<br/>Temporal XGB + Multiscale LTD<br/>mean FAR F1 +0.579 pp"]
    L8 --> M
    L10 --> M

    M --> N["❌ 12A Robust Hybrid<br/>mean FAR gain +0.2095 pp<br/>gate +0.25 pp"]
    N --> O["✅ 13A Temporal Micro ensemble<br/>mean FAR F1 +6.45 pp"]
    O --> P["🚧 13B Mechanism control<br/>Temporal vs seed ensemble"]
    P --> Q["13C Robust Micro × Robust Macro"]
    Q --> R["13D Full-Historical refit"]
    R --> S["13E Future-B post-hoc"]
    S --> T["🅿️ Dataset C virgen"]
```

## Qué está demostrado

1. **Concept Drift es severo.** Hybrid: **97.44% → 49.45% Accuracy**; Micro: **95.00% → 43.12%**.
2. **Hybrid mitiga, no elimina.** En Future-B supera a Micro en Accuracy (+6.34 pp), Macro-F1 (+5.13 pp), Top-5 (+8.17 pp) y MRR (+6.91 pp).
3. **XGBoost es la rama Macro más estable en Future-B**; LTD aporta identidad, pero su contexto longitudinal envejece.
4. **No funcionó intentar “borrar” el tiempo**: stable-only, invariance, prototypes y trajectories fueron descartados.
5. **La memoria stale sí explica parte del deterioro**, pero actualizarla online no es la solución final: el oracle ayuda, pseudo-refresh empeora.
6. **Dos mecanismos frozen dieron señal Macro:** multiscale memory (11B) y temporal XGB experts (11H).
7. **11L pasa el gate Macro:** +0.579 pp FAR Macro-F1 medio.
8. **12A no pasa al Hybrid:** +0.2095 pp < +0.2500 pp.
9. **13A es la señal más fuerte actual:** FAR Macro-F1 **69.88→73.47** y **77.39→86.71**; media **+6.45 pp**.
10. **13B es obligatorio antes de integrar:** debe separar temporalidad específica de ensemble genérico.

## Estado actual

- Arquitectura confirmatoria: **LTD-HYBRID-FINAL**.
- Candidato Macro robusto: **TEMPORAL_SYMMETRIC3 + MULTISCALE5**.
- Candidato Micro robusto principal: **13A temporal ensemble**.
- Próximo experimento: **13B**.
- Future-B: **abierto y congelado**; no se usa para selección nueva.
- ORIGIN14/ORIGIN28: **desarrollo repetido**, no validación externa.
- Dataset C: **validación externa virgen preferida**.

## Regla científica central

> La robustez longitudinal debe adquirirse durante el entrenamiento y mantenerse con inferencia congelada.  
> La meta es degradar más lentamente y espaciar reentrenamientos, no hacer continual learning.
