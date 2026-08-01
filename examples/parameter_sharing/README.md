The reference price file for `example_02b` is included under `reference_prices`, so the experiment can be started directly:

```bash
python examples/parameter_sharing/run_experiments.py
```

Use `--reference-price-directory` only when comparing against reference files stored somewhere else.

## Current experiment matrix

The constants near the top of `run_experiments.py` control the algorithms, examples, training episodes, and seeds. The current variants are:
- `independent`
- `full_none`
- `full_one_hot`
- `full_semantic`
- `full_id_context`

The runner expects the parameter-sharing implementation from the `extended_with_parameter_sharing` branch to be installed in the active Python environment.
