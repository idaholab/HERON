# `<TestInfo>` node

Used only in the inputs for HERON's own regression tests (under `tests/`).
HERON's input parser ignores it. It records what the test is for, for the
test harness and for software quality (SQA) traceability. Leave it out of
user study inputs.

```xml
<TestInfo>
  <name>storage_limits</name>
  <author>jdoe</author>
  <created>2026-10-27</created>
  <description>
    Checks that storage charge and discharge rate limits are respected.
  </description>
  <classesTested>HERON</classesTested>
  <requirements>H-F-1</requirements>   <!-- optional -->
</TestInfo>
```

| Child | Meaning |
|-------|---------|
| `<name>` | Test name. |
| `<author>` | Author's username. |
| `<created>` | Date created, in `YYYY-MM-DD` format. |
| `<description>` | What the test checks and why. |
| `<classesTested>` | The HERON classes or features the test covers. |
| `<requirements>` | Optional. Space-separated SQA requirement IDs that the test verifies. |

New test inputs must include `<TestInfo>`. Each test also needs an entry in
the directory's `tests` spec file (`HeronIntegration` tester) and gold files
for its expected outputs.
