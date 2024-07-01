# HERON Theory Manual

Setting up a HERON run requires the following activities (taken from the HERON User Guide):

1. Market Characterization
   - Electricity and/or Commodity Markets
   - Regulated vs. Deregulated Markets
   - Price Taker vs. Price Maker modeling
   - Grid-support functions (Ancilliary Services)
   - Additional Tax Incentives
2. Technology Identification
3. Technology Characterization
   - [(Technical) HERON Components](./ComponentCharacterization/HERON_Components.md)
   - [(Economic) HERON CashFlows](./ComponentCharacterization/HERON_Cashflows.md)
4. Time History Training
   - [Time Index](./TimeHistoryTraining/HERON_TimeIndexing.md)
5. Pre-analysis Calculations

Within these activities, some additional guides are presented for HERON definitions.

## HERON Workflows
Different workflows:
- `standard`: [(RAVEN-runs-RAVEN)](./Workflows/HERON_Standard_Workflow.md)
  - most common, standard for a reason
  - stochastic bi-level optimization scheme
- `MOPED`:
  - all-at-once solve (testing/development temporarily suspended)
- `DISPATCHES`:
  - all-at-once solve + integration with DISPATCHES algebraic models (development paused)
