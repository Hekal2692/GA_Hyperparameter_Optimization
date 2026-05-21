# Benchmark JSON Validator

This script validates a generated scheduling benchmark JSON file.

It checks three main parts:

1. Application DAG
2. Platform graph
3. `can_run_on` processor mapping

The script is useful after generating new variants with `generate_dag_variants.py`.

---

## Purpose

The validator checks whether a generated JSON file is safe to use for scheduling experiments.

It validates:

- `application.jobs`
- `application.messages`
- whether the application graph is a valid DAG
- `platform.nodes`
- `platform.links`
- whether the platform is connected
- whether every job has a valid `can_run_on` list
- whether `can_run_on` contains only processor node IDs

---

## Important Rule for `can_run_on`

`can_run_on` must contain only processor node IDs.

A processor is a platform node where:

```json
{
  "is_router": false
}

# To run the validation

```bash 
  python3 dag_variant_generator/check_application_dag.py \
  --input generated_variants/70T_variant_000.json \
  --verbose
  ```