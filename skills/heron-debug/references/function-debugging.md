# Debugging a `<Function>`

A cash flow `<Function>` is called twice: first while the Pyomo model is
built, with `meta['HERON']['activity']` values that are Pyomo expressions,
and then again with numbers when the final cash flows are computed. A
function that works when called by hand can still fail inside the dispatch.

- Run in debug mode, so there is one design point and one realization.
- Print from the function (`print(type(activity[...]), ...)`) to see which
  call is failing. Prints land in `generalOut` or the debug console.
- Keep arithmetic on activity linear and free of `if`, `min`, `max`, and
  NumPy. Branching on a signal value or a capacity is fine, because those
  are numbers.
- Check the return value: a tuple `(data, meta)` of two dicts, with the key
  the slot expects (`../heron/references/spec/functions.md`).
