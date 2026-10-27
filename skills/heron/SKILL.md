---
name: heron
description: Build, configure, and run HERON (Holistic Energy Resource Optimization Network) technoeconomic analyses. Use when the user is writing or editing a HERON XML input file; defining components, resources, or economics; setting up time series (synthetic history / ARMA) or other uncertainty; choosing between debug, sweep, and opt modes; or asking how to structure a HERON study. For failing or suspicious runs, use the heron-debug skill instead.
---

# HERON

<!-- TODO: 2-3 sentences: what HERON is (RAVEN plugin; outer capacity loop,
     inner stochastic dispatch loop; TEAL economics). -->

## Input file anatomy

<!-- TODO: the top-level nodes (<Case>, <Components>, <DataGenerators>) and
     how they relate. Keep to a short annotated skeleton; point to the user
     manual for node-level detail. -->

## Recommended workflow

<!-- TODO: the debug -> sweep -> opt progression in brief, with the reason
     for each step. Details live in references/modes.md. -->

1. Start in debug mode ...
2. ...

## References

Load these only when the task needs them:

| File | Read when |
|------|-----------|
| `references/components.md` | Defining producers, demanders, storage, capacities, transfer functions |
| `references/valued-params.md` | Choosing how a value is supplied (fixed, sweep/opt, Function, SyntheticHistory, ...) |
| `references/time-series.md` | Training or using synthetic history models, realizations, clustering |
| `references/uncertainty.md` | Deciding what is stochastic, sample counts, statistics and objectives |
| `references/modes.md` | Choosing debug vs. sweep vs. opt and configuring each |
| `references/best-practices.md` | Reviewing a study design or input for common mistakes |

## Authoritative sources

<!-- TODO: where the installed user manual and workshop examples live, and
     how the agent should find them. Depends on how the skills are packaged. -->
