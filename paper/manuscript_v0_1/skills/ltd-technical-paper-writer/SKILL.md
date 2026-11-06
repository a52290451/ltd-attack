---
name: ltd-technical-paper-writer
description: Scientific English editor for Tor website fingerprinting, concept drift, chronological ML evaluation, and LTD-Attack LaTeX publication artifacts.
---

# LTD-Attack Scientific Writing Skill (project-local)

This is an **authored project-local skill** included in the manuscript package; it is not an automatically installed ChatGPT plugin. Apply it whenever editing the paper, appendix, figures, captions, tables or response-to-reviewers.

## Mandatory source priorities
1. Immutable experiment manifests and per-UID scores, with SHA/date/source validation.
2. Exact executed code/checkpoints and manuscript evidence records.
3. Original research publications and verified DOIs.
4. Preliminary workshop draft, clearly labeled superseded if it contradicts newer executed evidence.

## Writing workflow
1. Read `editorial/TECHNICAL_ENGLISH_PROTOCOL.md`.
2. Identify every factual/quantitative claim and its exact provenance; mark pending facts without invented values.
3. Write precise idiomatic English for a peer-reviewed security/privacy venue, preferring conservative conclusions.
4. Explain study design before results: closed-world task, temporal cohorts, label alignment, fit/selection split, threat model.
5. Distinguish a retrospective frozen evaluation from a prospectively blind test; never call one the other.
6. Keep baseline methodology names honest: `CUMUL-style` and `WFlib-adapted NetCLR` where reproduction is not exact.
7. For all figures ensure correspondence to trained implementation, readability when reduced to one column, editable master, vector export, descriptive alt text.
8. Recheck tables, references, terminology and mathematical units; compile and visually inspect the LaTeX PDF.

## Failure conditions
- Recasting S6 Holm8 p-values as covering additional S7–S9 baselines.
- Treating Dataset C or NetCLR as completed before their passing artifacts exist.
- Attributing the entire Future-B loss to concept drift without confounder assessment.
- Using the old 3,000-packet/45-feature description as the final architecture without verifying the frozen model code.
- Claiming deployment viability without verifying inference-time availability of macro features.

## Publication acceptance checklist
- Every statistic points to a frozen provenance artifact.
- All hypotheses and inferential families are explicitly stated and scoped.
- Every cited source is actually verified; bibliography includes primary SOTA papers.
- Distinct results and limitations survive an independent reviewer reading.
- Full project compiles and contains no missing references or unapproved placeholder figures.
