# Parameter-sharing experiments

This runner compares independent actors with the currently implemented shared
actor configurations across algorithms and random seeds. It reuses the scenario
folders under `examples/inputs` and writes each run to an isolated database and
policy directory.

The reference price CSV files are not part of this repository. Pass the folder
containing files such as `prices_02b.csv` explicitly:

```bash
python examples/parameter_sharing/run_experiments.py \
  --reference-price-directory /path/to/reference_prices
```

The default experiment outputs are written under
`examples/outputs/parameter_sharing_experiments`, which is ignored by Git. A
different location can be selected with:

```bash
python examples/parameter_sharing/run_experiments.py \
  --reference-price-directory /path/to/reference_prices \
  --output-directory /path/to/outputs
```

Use `--inputs-directory` only when the scenario folders are stored outside the
repository's normal `examples/inputs` location.

## Current experiment matrix

The constants near the top of `run_experiments.py` control the algorithms,
examples, training episodes, and seeds. The current variants are:

- `independent`
- `full_none`
- `full_one_hot`
- `full_semantic`
- `full_id_context`

The runner expects the parameter-sharing implementation from the
`extended_with_parameter_sharing` branch to be installed in the active Python
environment.

Generated databases, policies, CSV files, and plots should not be committed.
