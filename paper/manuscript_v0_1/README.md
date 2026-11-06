# LTD-Attack — publication-neutral LaTeX manuscript v0.1

Working date: 6 November 2026. **Not submission-ready.** This directory is the initial manuscript snapshot based on the three-page preliminary ACM workshop summary and frozen S6, S7, S8 evidence.

## Build
From this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

For a clean build: `latexmk -C`. Without latexmk: `pdflatex -interaction=nonstopmode -halt-on-error main.tex` twice.

## Scientific restrictions
- NetCLR is running and has no frozen external evaluation yet; Dataset C remains pending.
- 09B and Future-B are retrospective and previously inspected, not prospective blind holdouts.
- No conclusive claim of concept drift causation from accuracy decline alone.
- Existing S6 Holm8 p values do not extend to CUMUL/Tik-Tok/NetCLR.
- The workshop architecture and fusion/feature counts must be audited against frozen implementation before publication.
- `figures/research_workflow.tex` depicts the experiment, not the internal model.
- An **additional** editable conceptual architectural Figure 1 is being developed separately; it must not be silently substituted for the current workflow or declared code-exact.
- Final bibliographic metadata and SOTA citations remain to be audited.

Full experiment evidence remains in `docs/research/article_evidence/` of this repository.
