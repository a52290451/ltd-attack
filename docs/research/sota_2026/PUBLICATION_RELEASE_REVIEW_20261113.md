# LTD-ATTACK — Cierre científico integral SOTA
**Fecha:** 2026-11-13  
**Estatus experimental:** \`SCIENTIFIC_TABLES_INFERENCE_AND_METHOD_SCOPE_FROZEN\`  
**Estatus editorial:** borrador científico v0.2 compilado; **NO** implica cámara-ready ni verificación prospectiva.

## Entregas de los cinco frentes
1. **Congelación de evidencias — completado con trazabilidad conocida.**
   - Informe métricas y 168 IC: \`/data/datasets_brayan/ltd-sota-2026/audits/SOTA_ARTICLE_FINAL_20261113T172826Z/\`
   - Valores p para 168: \`/data/datasets_brayan/ltd-sota-2026/audits/SOTA_STATISTICAL_FINAL_20261113T175918Z/\`
   - Catorce pruebas exactas: \`/data/datasets_brayan/ltd-sota-2026/audits/SOTA_SCIENTIFIC_CLOSURE_20261113T181719Z/\`.
   - Los artefactos ya congelados y sus hashes están en los certificados originales de Zeus. El entorno de redacción del artículo no accedió directamente a todos los bytes originales. No debe presentarse su transcripción como una nueva auditoría binaria independiente.
2. **Integración LaTeX — completado en paquete editorial v0.2 separado.** Abstract, método basado en la auditoría del código original, tabla de 20 resultados (10 modelos × 2 cohortes), tabla de 14 Δ/IC/p exactos, bibliografía verificada, figura de arquitectura v5, discussion/validity/ethics/conclusion y anexos. El manuscrito v0.1 permanece intacto porque sus antiguos S6/WFlib/CUMUL resultados y declaraciones de NetCLR pendiente NO son coherentes con la nueva batería.
3. **Interpretación temporal — completada descriptivamente.** Pérdida de accuracy 09B→Future-B: Hybrid −47.99 pp, VarCNN dirección −46.82 pp, Micro −51.88, Macro −61.77, DF −54.42, Tik-Tok −76.31, NetCLR −64.38, RoFiRe −53.64, WFCAT −90.36, Swallow v2 −65.99. Los dos conjuntos están compuestos por capturas diferentes; no adjudicar significancia inferencial a estas caídas ni causalidad al drift. VarCNN Future-B Top-5 80.98 % supera Hybrid 80.56 %.
4. **Revisión metodológica — realizada y delimitada.** VarCNN solo rama dirección; RoFiRe DyWin x10 vs original x200; NetCLR y Swallow con presupuesto y protocolos modificados; WFCAT sin certificación completa de unidad temporal del recolector. Swallow BYOL v1 colapsado excluido, BNtrain v2 incorporado. Source/config del LTD congelado apunta a Micro 3000 dirección+tamaño; Macro XGBoost + LTDPairScorer condicionado por históricos de sitios; fusión geométrica. Quedan como límites el rehash binario independiente y disponibilidad real del banco histórico en despliegue. Las ablaciones 10C muestran que la mejora Hybrid sobre Micro NO debe atribuirse íntegramente a LTD: retirar LTD y renormalizar produce Δ +0.2912 pp IC95 [−0.1569,+0.7114], mientras retirar XGB produce +4.7943 pp IC95 [+4.2763,+5.2801].
5. **Versión publicación — borrador reproducible y compilado, no envío final de venue.** El paquete local de la conversación incluye 13 páginas de PDF, fuente LaTeX, editable SVG, fuente PDF de la figura, 20 filas en CSV, 14 pruebas exactas y manifiesto de integridad local. El CSV original de 168 IC+p permanece en Zeus y se debe adjuntar desde allí sin inferir sus valores a partir de logs parciales.

## Hallazgos y significancia
- **09B:** Hybrid 97.44 % vs RoFiRe adaptado 94.21 %. Δ +3.23 pp, IC95 [+2.83,+3.57], p exacto bilateral ≈0.000977, Holm14 **pos hoc** 0.006836; Holm168 Monte Carlo 0.098361.
- **Future-B:** Hybrid 49.45 % vs VarCNN dirección adaptado 44.55 %. Δ +4.90 pp, IC95 [+3.47,+6.24], p exacto bilateral ≈0.000214, Holm14 **pos hoc** 0.001709; Holm168 Monte Carlo 0.067173.
- **14/14** contrastes focalizados Hybrid accuracy p Holm14 exacto < 0.05. **0/14** de esos contrastes p Holm168 Monte Carlo < 0.05. Estas dos familias no se deben confundir. Holm14 fue definida durante la investigación y **no es prerregistrada**.
- Los IC son bootstrap pareado por 11/16 fechas, 2500 réplicas; 14 p exactos enumeran 2^11/2^16 inversiones de signo de bloques diarios; el resto de 168 p Monte Carlo 2500 y Holm168.
- Los hallazgos son **retrospectivos y condicionados a siete variantes de código fuente adaptado**, no prueban equivalencia paper-exact ni éxito prospectivo o causalidad de cambio temporal.

## Auditoría de bibliografía
Citas originales contrastadas con fuentes de editor o repositorios oficiales:
- DF: Sirinam et al., CCS 2018; Var-CNN: Bhat et al., PoPETs 2019, doi:10.2478/popets-2019-0070.
- Tik-Tok: Rahman et al., PoPETs 2020, doi:10.2478/popets-2020-0043.
- NetCLR/NetAugment: Bahramali et al., CCS 2023, doi:10.1145/3576915.3616639.
- RoFiRe: Jeon et al., WWW 2026, upstream https://github.com/AISec-EWHA/RoFiRe.
- WFCAT: Gong et al., IEEE TDSC 23(1), 2026, doi:10.1109/TDSC.2025.3605197.
- Swallow: Shen et al., CCS 2025, doi:10.1145/3719027.3744795.

## Requisitos NO certificados para entrega editorial definitiva
1. Hash independiente de los binarios finales LTD y correspondencia fuente→checkpoint.
2. Evidencia del banco de perfiles históricos realmente disponible y estrictamente anterior en cada predicción, con límites de acceso/actualización.
3. Unidades originales del recolector relevantes para WFCAT y diagnóstico causal de esa degradación.
4. Autorizaciones, ética, condiciones y retención de los datos de captura.
5. Estadística completa de 168 filas incorporada físicamente al suplemento desde Zeus.
6. Elección y adecuación a una plantilla editorial/venue específicos.
7. Para **afirmaciones prospectivas**, nuevo conjunto ciego independiente; para **reproducción paper-exact**, restauración de configuraciones originales.

## Decisión de publicación
Apto como **manuscrito científico completo de resultados retrospectivos de siete baselines adaptados**, con declaración explícita de limitaciones. No etiquetar como cámara-ready revisado institucionalmente, pre-registro estadístico o reproducción exacta de los papers.

## Material fuente del repositorio
- Guía de figura arquitectura verificada: \`docs/research/article_evidence/FIGURE1_ARCHITECTURE_VALIDATION_20261108.md\`.
- Resumen de ablaciones: \`docs/research/LTD_ATTACK_CLOSEOUT_RQ1_RQ4_FULL_20261026.md\`.
- Guía de fidelidad adaptada: \`docs/research/article_evidence/SOTA_ADAPTED_METHOD_VALIDITY_AUDIT_20261112.md\`.
- Generador de cierre y estadísticas existentes: \`scripts/sota/scientific_closeout_seven_sota_v1.py\` y \`scripts/sota/final_168_paired_date_randomization_pvalues_v1.py\`.
- Paquete editorial v0.2 disponible como artefacto ZIP de esta conversación. No se modifica el v0.1 existente.
