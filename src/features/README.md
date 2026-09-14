# 🔬 Módulo de Análisis Macro-Temporal (Features) y Diagnóstico de Concept Drift

Este directorio contiene la suite de experimentación dedicada a la evaluación del "ADN Inmutable" (características estadísticas del tráfico de red). Su objetivo es demostrar matemáticamente cómo el Concept Drift degrada los enfoques tabulares y de series temporales (Macro), justificando la transición hacia arquitecturas híbridas y Deep Metric Learning.

## 🚀 Orden de Ejecución y Pipeline

El pipeline se divide en dos fases: el modelado predictivo y la generación de evidencia diagnóstica. Todos los scripts aplican **Paridad Estricta de 65 Clases** para una comparación 1:1 con la rama Micro.

### Fase 1: Modelos Base (Entrenamiento y Evaluación)

| Script | Descripción |
| :--- | :--- |
| `EXP1_Train_Invariantes.py` | Entrena un *Hourly Encoder* (Transformer Temporal) sobre 94 características. Implementa PyTorch 2.0 (AMP, compile, Gradient Clipping) sin *Data Leakage*. |
| `EXP1_Evaluar_Invariantes.py` | Evalúa la resiliencia del EXP1 frente al dataset del Futuro (+2 meses). |
| `EXP2_XGBoost_MetaFeatures_24h.py` | Extrae Firmas Dinámicas 24h (Meta-Features 752D), purga colinealidad y entrena un modelo XGBoost hiper-regularizado. |

### Fase 2: Evidencia Visual y Diagnóstico Estructural

| Script | Descripción y Salida Gráfica |
| :--- | :--- |
| `GEN_01_Ranking_Outliers.py` | Extrae el *Gini Importance* (Random Forest) y visualiza el PCA base. Genera **G3** y **G4**. |
| `GEN_01B_Degradacion_Drift.py` | Ilustra la vulnerabilidad intrínseca de las características estadísticas. Genera **G2**. |
| `GEN_02_PCA.py` | Proyecta el hiperespacio de 752D para visualizar el impacto sobre flujos de 24h. Genera **G5**. |
| `GEN_03_Ortogonalidad.py` | Aplica un filtro de Pearson (\|r\| > 0.85) sobre las meta-features. Genera **G6**. |
| `GEN_04_PSI_KDE.py` | Mide el desplazamiento espacial (*Z-Shift*) entre distribuciones del Pasado y Futuro. Genera **G7A** y **G7B**. |
| `GEN_05_PCA_Traslacion.py` | Evidencia cómo el ruteo de Tor traslada un clúster específico en el espacio PCA. Genera **G8**. |
| `GEN_06_Similitud_Metric.py` | Evalúa la separabilidad Angular (*Cosine Similarity*), justificando el uso de DML. Genera **G9** y **G10**. |

## 🛠️ Ejecución Automatizada
Para ejecutar el ciclo completo de validación y generar todas las matrices y gráficas, utiliza el script de orquestación:
```bash
nohup ./run_features.sh > salida_features.log 2>&1 &