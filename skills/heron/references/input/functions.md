# Python functions (`<Function>`)

How to write the Python functions that `<Function>` value slots and storage
`<strategy>` call. For declaring the file, see `data-generators.md`.

Use `<Function>` when a value can't be expressed with the other types: a
price computed from a signal, a driver built from several activities, or a
storage control rule. It is a little-used feature with few safeguards, so
keep functions simple and check the results in debug mode.

**Wiring.** Declare the Python file once in `<DataGenerators>`, then refer to
it from a value slot by name, using `method` to choose the function:

```xml
<!-- in <DataGenerators> -->
<Function name="funcs">functions.py</Function>

<!-- in a value slot -->
<reference_price>
  <Function method="price">funcs</Function>
</reference_price>
```

HERON imports the file when it reads the input. Every top-level callable in
the file can be used as a `method`. The same file can serve any number of
slots.

**Signature.** HERON calls `method(data, meta)`. The function must return a
tuple `(data, meta)` where both are dictionaries; anything else is an error.

- `data`: the request. Return a dict whose key is what the slot asks for:

  | Slot | Return key |
  |------|------------|
  | CashFlow `<driver>`, `<reference_price>`, `<reference_driver>`, `<scaling_factor_x>` | the slot name, e.g. `{'driver': value}` |
  | `<capacity>`, `<minimum>`, `<initial_stored>` | the capacity resource name, e.g. `{'electricity': value}` |
  | Storage `<strategy>` | `{'level': array}`, the storage level at every time step of the window |

- `meta`: HERON's state. Return it unchanged unless you have a reason not
  to. Useful entries in `meta['HERON']`:

  | Key | Contents |
  |-----|----------|
  | `RAVEN_vars` | Sampled values for the current year: capacities (e.g. `['npp_capacity']`), `dispatch_vars`, and time series signals (e.g. `['PRICE']`, an array over time). Scalars may arrive as one-element arrays, so wrap them in `float()` |
  | `time_index` | Index of the current time step within the year, so `RAVEN_vars['PRICE'][time_index]` is the current value |
  | `time_value` | Value of the time variable at that step |
  | `activity` | This component's activity at the current step, as `activity[tracking_var][resource]`, e.g. `activity['production']['electricity']`. Storage uses `level`, `charge`, `discharge`. CashFlow slots only. |
  | `component` | The component that owns this cash flow (CashFlow slots only) |
  | `Components` | All component objects |
  | `Case` | The case object, e.g. `meta['HERON']['Case'].get_labels()` |
  | `custom_input` | The `<AdditionalInfo>` XML elements for this slot, if any |

**Cash flow functions are also used inside the dispatch optimization.** For
hourly CashFlows on a Pyomo dispatch, HERON calls the function while building
the optimization model. At that point `activity` values are Pyomo
expressions, not numbers. Only use plain arithmetic on activity (`+`, `-`,
`*`, `/` by constants or signals). Don't use `if` tests, `min`/`max`, or
NumPy functions on it. The same function is called again with numbers when
the final cash flows are computed.

**Signs.** Activity follows the usual convention (positive for produced,
negative for consumed). A function returning a market's electricity activity
as a driver still needs a `<multiplier>-1</multiplier>`, or a sign flip in
the code, to count as revenue (see `economics.md`).

**Example: a price derived from a signal, and an activity driver.**

```python
def price(data, meta):
  """
    Scales the price signal to $/MWh.
    @ In, data, dict, request to be filled
    @ In, meta, dict, HERON state
    @ Out, data, dict, filled request
    @ Out, meta, dict, HERON state
  """
  t = meta['HERON']['time_index']
  signal = meta['HERON']['RAVEN_vars']['PRICE']
  return {'reference_price': 2.0 * signal[t]}, meta

def electricity_sold(data, meta):
  """
    Electricity taken by this market (negative activity).
    @ In, data, dict, request to be filled
    @ In, meta, dict, HERON state
    @ Out, data, dict, filled request
    @ Out, meta, dict, HERON state
  """
  activity = meta['HERON']['activity']
  return {'driver': activity['production']['electricity']}, meta
```

```xml
<CashFlow name="sales" type="repeating" taxable="True" inflation="none">
  <driver>
    <Function method="electricity_sold">funcs</Function>
    <multiplier>-1</multiplier>
  </driver>
  <reference_price>
    <Function method="price">funcs</Function>
  </reference_price>
</CashFlow>
```

**Example: passing settings with `<AdditionalInfo>`.**

```xml
<reference_price>
  <Function method="scaled_price">funcs</Function>
  <AdditionalInfo>
    <scale>1.5</scale>
  </AdditionalInfo>
</reference_price>
```

```python
def scaled_price(data, meta):
  """
    Price signal times a user-supplied scale.
    @ In, data, dict, request to be filled
    @ In, meta, dict, HERON state
    @ Out, data, dict, filled request
    @ Out, meta, dict, HERON state
  """
  scale = 1.0
  for node in meta['HERON']['custom_input']:
    if node.tag == 'scale':
      scale = float(node.text)
  t = meta['HERON']['time_index']
  return {'reference_price': scale * meta['HERON']['RAVEN_vars']['PRICE'][t]}, meta
```

**Storage `<strategy>`.** This replaces the optimized storage dispatch with
a rule you write. It is called once per dispatch window. `data['component']`
is the storage component and `data['time']` is the array of time values. It
must return `{'level': levels}`, with one storage level for each time step.
The current initial level is available from
`data['component'].get_interaction().get_initial_level(meta)`, and the
storage capacity from `data['component'].get_capacity(meta)[0][resource]`.
