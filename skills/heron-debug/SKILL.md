---
name: heron-debug
description: Diagnose HERON runs that fail or produce suspicious results. Use when a HERON input fails to parse, RAVEN errors during the outer or inner workflow, a dispatch solve is infeasible or crashes, or outputs look wrong (NaN or extreme NPV, flat or nonsensical dispatch, optimizer not converging). For building or configuring a model, use the heron skill instead.
---

# Debugging HERON

## Triage

<!-- TODO: an ordered procedure. Draft outline:
1. Identify which stage failed: HERON parsing, workflow generation,
   RAVEN outer, RAVEN inner, dispatch solve, or postprocessing.
2. Find the first real error (not the last traceback).
3. Reproduce in debug mode with minimal samples.
4. Look up the symptom in references/error-catalog.md.
-->

## Where to look

<!-- TODO: generated files (outer.xml, inner.xml), run directories,
     log files, dispatch output and plots from debug mode. -->

## Suspicious results (no error)

<!-- TODO: checks for results that run but look wrong. -->

## References

| File | Read when |
|------|-----------|
| `references/error-catalog.md` | You have a specific error message or symptom |
