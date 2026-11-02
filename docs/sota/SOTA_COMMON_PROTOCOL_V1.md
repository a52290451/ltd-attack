# LTD-Attack: protocolo común SOTA v1

**Estado:** protocolo de preparación (S0), sin entrenamiento ni evaluación nueva.

## Objetivo de comparación

Comparar baselines reproducidos de *website fingerprinting* con el LTD-Attack **sobre los mismos datos A/B, clases y ventanas temporales congeladas**. Las métricas publicadas en estudios con otras muestras no son directamente comparables.

## Comparadores priorizados

| Modelo | Publicación/código | Comparación | Prioridad |
|---|---|---|---|
| Deep Fingerprinting (DF) | [CCS 2018](https://doi.org/10.1145/3243734.3243768) · [código](https://github.com/deep-fingerprinting/df) | Baseline CNN de direcciones de paquetes | 1 |
| Var-CNN | [PoPETs 2019](https://doi.org/10.2478/popets-2019-0070) · [código](https://github.com/sanjit-bhat/Var-CNN) | Baseline eficiente en datos, dirección/tiempo según configuración | 1 |
| NetCLR | [WFlib](https://github.com/FIND-Lab/Website-Fingerprinting-Library) (CCS 2023) | Robustez a variabilidad de trazas | 2 |
| Tik-Tok | [WFlib](https://github.com/FIND-Lab/Website-Fingerprinting-Library) (PoPETs 2020) | Comparador del canal temporal si existe en A/B | 2 |
| Proteus | [NDSS 2026](https://www.ndss-symposium.org/ndss-paper/enhancing-website-fingerprinting-attacks-against-traffic-drift/) · [código](https://github.com/Xinhao-Deng/Adaptive-WF-Attack) | Adaptación al tráfico desplazado sin etiquetas; **experimento adaptativo separado** | 3 |

WFlib ofrece implementaciones en PyTorch con estructura común para DF, Var-CNN, Tik-Tok y NetCLR, pero su compatibilidad de entrada debe validarse con los datos A/B originales antes de usarla.

## Principios que no se negocian

1. **Historical DEV**: único corte admisible para entrenar, validar y seleccionar hiperparámetros. Respetar sus identificadores y exclusiones. No crear particiones aleatorias que rompan el diseño temporal.
2. **Future-B**: exclusivamente evaluación *post hoc*. Sin normalizar, escoger modelos, establecer umbrales, entrenar o recalibrar con etiquetas ni trazas futuras. Un método que requiere adaptación online debe reportarse **aparte** de la comparación no adaptativa.
3. **Mismas clases y muestras** en cada par comparado. Registrar explícitamente incompatibilidades, trazas ausentes y subcohortes; no imputar clases ni excluir casos en silencio.
4. **Representación de tráfico**: partir de la misma fuente de observación y extracción para cada baseline. Tamaños, direcciones y tiempos solo cuando procedan del tráfico realmente observado; sin HTML, identidad de dominio, URL ni metadatos privilegiados como entrada del clasificador.
5. **Comparación estadística**: aplicar el método ya congelado del proyecto (bootstrap pareado por bloques de fecha, 16 días, 2.500 remuestreos, IC percentil 95 %) solo a pares con unidad de comparación y fechas coincidentes.
6. **Métricas**: conservar las definiciones congeladas de precisión, MRR y demás indicadores del LTD-Attack; no reemplazarlas por cifras publicadas en otros datasets.
7. **Reproducibilidad**: registrar hash de manifiestos, versiones de código, semilla, adaptadores, presupuesto de ajuste, tiempos, fallos y predicciones por muestra. No hacer *commit* ni sobrescribir los resultados anteriores.
8. **Dataset C**: QA separado; los archivos C no se emplean para seleccionar ni entrenar modelos A/B. Persisten las restricciones NTP, metodología y almacenamiento antes de autorizar mediciones científicas recurrentes.

## Ejecución secuencial

**S0 — Inventario e integración.** Ejecutar `python3 scripts/sota/s0_readonly_inventory.py`. Identificar los manifiestos de Historical DEV/Future-B, selectores de muestras, código extractor, métricas y dependencias sin abrir Future-B. Salida: reporte local S0.

**S1 — Adaptadores auditables.** Definir tensores de dirección y tiempo de entrada y convertidores desde las trazas originales. Validar formas, cardinalidades de clase y orden temporal en Historical DEV. Future-B no se abre.

**S2 — Entrenamiento DEV.** Ejecutar DF y Var-CNN con semilla y búsqueda de hiperparámetros registradas; fijar los mejores experimentos por criterio decidido de antemano.

**S3 — Extensión.** Ejecutar NetCLR/Tik-Tok si sus canales e implementaciones son admisibles. Preparar Proteus como escenario de adaptación separado y definir qué información sin etiqueta podría observarse lícitamente.

**S4 — Evaluación final.** Evaluar los modelos congelados una vez sobre Future-B y calcular contrastes pareados siguiendo el método estadístico original. Generar tablas reproducibles y limitaciones.

## Punto de control QA Dataset C (no ciencia)

- Inventario histórico: 120 posiciones / 118 URL únicas.
- Cobertura QA reconciliada: **78/120 posiciones** y **47/65 clases de la cohorte histórica**.
- Las últimas cinco posiciones verificadas: Rituals, CoverManager, SaludVital, Hospital del Mar y Grupo CTO.
- La recuperación activa de C queda cerrada operativamente. Los casos restantes se conservan como inaccesibles o insuficientes **bajo las condiciones de QA probadas**; no se elimina su etiqueta ni se altera A/B.
