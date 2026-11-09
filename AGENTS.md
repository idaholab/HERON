# AGENTS.md

Guidance for AI coding agents working in this repository.

## What HERON is

HERON (Holistic Energy Resource Optimization Network) is a **plugin for RAVEN**.
Users write a HERON XML input file. HERON parses it (`src/Cases.py`,
`src/Components.py`, `src/ValuedParams/`, ...) and generates RAVEN workflows
from the templates in `templates/`. RAVEN then runs those workflows, using the
TEAL plugin for economics and HERON's Pyomo-based dispatchers (`src/dispatch/`)
for the inner dispatch optimization.

HERON is long-lived research software with many users and existing input
files. It is maintained as-is: **no large refactors, re-architectures, or
dependency swaps** unless a human explicitly asks for one.

Layout:
- `src/` — input parsing, components, valued params, dispatch, validators, testers
- `templates/` — RAVEN workflow templates and the template driver
- `tests/` — `unit_tests/`, `integration_tests/`, `workshop/`
- `doc/` — user manual (LaTeX), theory manual, workshop material, SQA docs
- `dependencies.xml` — conda/pip dependencies consumed by RAVEN's environment scripts

## Hard rules

1. **Do not edit tests unless a human explicitly instructs you to.** This
   includes test input files (`tests/**/*.xml`, test `.py` files, `tests` spec
   files, data/CSV inputs). Adding a *new* test is fine when asked or when
   proposing one; modifying an existing test is not.
2. **Never modify gold files** (anything under a `gold/` directory) without
   explicit human instruction for that specific change. If your code change
   alters a gold result, stop and report: which test, what differs, and why
   you believe it changed. Do not "regold" to make a test pass.
3. **Backwards compatibility is mandatory.** Every existing HERON input file
   must continue to parse and produce the same results. In particular:
   - Do not rename, remove, or change the meaning of XML nodes or attributes.
   - Do not change default values.
   - Do not change names of variables, outputs, or files in generated RAVEN
     workflows that users or downstream scripts may rely on.
   - New input features must be optional, with defaults that reproduce the
     prior behavior.
   - To retire an input, keep it working and emit a deprecation warning
     (see `src/TransferFuncs/Ratio.py` for the pattern). There are no input
     conversion scripts in HERON; inputs must never be broken.
4. **Do not change `dependencies.xml`, CI workflows (`.github/`), or build
   config (`build_cfg/`, `setup.cfg`)** without asking first.
5. Do not commit, push, or open PRs unless asked.

## Code style

HERON follows PEP 8 with RAVEN's conventions layered on top (see
`CONTRIBUTING.md` and the HERON/RAVEN coding-standard wikis). Match the
surrounding code.

- **Two-space indentation** (not four).
- `under_score` names for functions, methods, and variables;
  `CamelCase` for classes.
- Every file starts with the copyright header, then a module docstring:
  ```python
  # Copyright 2020, Battelle Energy Alliance, LLC
  # ALL RIGHTS RESERVED
  """
    Short description of the module.
  """
  ```
- Every function/method has a docstring with `@ In` / `@ Out` lines:
  ```python
  def read_input(self, xml, mode):
    """
      Sets settings from input file
      @ In, xml, xml.etree.ElementTree.Element, input from user
      @ In, mode, string, case mode to operate in (e.g. 'sweep' or 'opt')
      @ Out, None
    """
  ```
  Use `@ In, None` / `@ Out, None` when there are no arguments or returns.
- User-facing errors and messages go through the messaging methods
  (`self.raiseAnError(ErrorType, msg)`, `self.raiseAWarning(msg)`,
  `self.raiseAMessage(msg)`, `self.raiseADebug(msg)`) on classes that derive
  from `src/base.py`. Do not add new `print()` calls; existing ones are legacy.
- Use `assert` only for internal invariants a user cannot trigger. RAVEN may
  run with `python -O`, which strips assertions, so never use them to
  validate input.
- Write text files (CSV, XML) with `\n` newlines, not `os.linesep`.
- New input nodes are declared through RAVEN's `InputData` specs
  (`get_input_specs`), with descriptions, so they appear in the user manual.
- No Python 2 compatibility code (`__future__` imports, etc.).

## Development environment and testing

A local RAVEN checkout is **required** to develop and test HERON. Look for it
in a neighboring directory (`../raven`). If it is not there or is incomplete
(e.g. no `run_tests` script), **ask the user where RAVEN is**. Do not try to
clone, install, or build RAVEN yourself.

With RAVEN built and HERON registered as a plugin
(`python <raven>/scripts/install_plugins.py -s <path to HERON>`), tests run
through RAVEN's test harness:

```bash
<raven>/run_tests -j4 --plugins --re=HERON/tests            # all HERON tests
<raven>/run_tests --plugins --re=HERON/tests/integration_tests/workflows/storage
```

- Integration tests are defined by `tests` spec files (HIT format) using the
  `HeronIntegration` tester (`src/Testers/`) and compare outputs to `gold/`.
- Each test input XML has a `<TestInfo>` block (`name`, `author`, `created`
  in YYYY-MM-DD, `description`, `classesTested`). New tests must include it.
- Coverage: `bash coverage_scripts/check_py_coverage.sh -j4`.
- If you cannot run the tests, say so explicitly in your report; do not
  claim a change is verified.

### Unit tests without RAVEN

HERON is being made gradually independent of RAVEN, and some unit tests
already run without it. They run under pytest (configured in
`tests/pytest.ini`, `tests/conftest.py`), with RAVEN, TEAL, and FARM imports
blocked even where RAVEN is installed:

```bash
python -m pytest -c tests/pytest.ini --no-raven tests/unit_tests
```

This takes about a second, so it is a quick check while developing, and the
Linux CI job runs it after `run_tests`. Always pass the test path: run from the
HERON root without it, pytest recurses through the `build_cfg/HERON` symlink.

A test is included only if both of these hold:
1. Its file is listed under `raven_free_files` in `tests/pytest.ini`. List a
   file only when all of its imports work without RAVEN.
2. The test is marked `raven_free`: `pytestmark = pytest.mark.raven_free` for
   a whole file, or `@pytest.mark.raven_free` on single test methods.

When writing new unit tests, keep them RAVEN-free where possible: test HERON
code directly, and don't import `ravenframework` in the test. If a test passes
under `--no-raven`, mark it and list its file. The RAVEN harness ignores the
marks, so marked tests still run there. Tests that need RAVEN stay unmarked.
`tests/unit_tests/test_raven_optional.py` checks that HERON imports and fails
cleanly without RAVEN. It runs only under pytest and has no `tests` spec entry.

## Project agent SKILLs

HERON contains two agent SKILLs (in ./skills/) which are bundled with the pip package.
These must be maintained alongside the source code. In particular, skills/heron/ includes
descriptions of features and XML input file nodes and options. This must be kept up to
date with the current state of the source code so the skills provide accurate guidance.

## Making changes

- Development is issue-driven. Changes should trace to a GitHub issue; ask
  for the issue number if it is not clear.
- Keep changes minimal and focused: a PR should be a complete task, but as
  small as possible. Avoid drive-by reformatting or cleanup of unrelated code.
- Branches are created off `devel` (the main branch), named
  `username/branch_name`.
- New or changed input features require a matching update to the user
  manual in `doc/user_manual/`.
- Bug fixes and new features should come with a new regression test; propose
  the test (and any new gold files it needs) for human review.
- PRs are reviewed by a member of the Change Control Board against the
  RAVEN/HERON review checklists.
