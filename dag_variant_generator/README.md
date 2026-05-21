# DAG Variant Generator

This tool generates new DAG/application/platform JSON variants from an existing scheduling benchmark JSON file. It is intended for real-time and reliability-aware scheduling experiments where each generated benchmark must preserve the original task count while changing the application DAG, the platform, or both.

The tool is self-contained in `dag_variant_generator/` and uses only the Python standard library.

## Input JSON Format

The expected input JSON contains:

- `application.jobs`
- `application.messages`
- `platform.nodes`
- `platform.links`

Example shape:

```json
{
  "application": {
    "jobs": [
      {
        "id": 0,
        "wcet_fullspeed": 35,
        "mcet": 0,
        "processing_times": 2,
        "deadline": 491,
        "can_run_on": [1, 2, 7]
      }
    ],
    "messages": [
      {
        "id": 0,
        "sender": 14,
        "receiver": 8,
        "size": 18,
        "timetriggered": true,
        "period": 10
      }
    ]
  },
  "platform": {
    "nodes": [
      {
        "id": 1,
        "is_router": false
      },
      {
        "id": 4,
        "is_router": true
      }
    ],
    "links": [
      {
        "start": 4,
        "end": 1
      }
    ]
  }
}
```

The number of jobs/tasks is always derived from `len(data["application"]["jobs"])`. Generated application variants preserve that task count. Unknown top-level fields such as `frequencies`, `schemes`, or future metadata are preserved.

## Output

Generated files are written to the configured output directory:

```text
generated_variants/
  variant_000.json
  variant_001.json
  generation_summary.json
```

Each `variant_*.json` keeps the overall input structure. The summary file records task/message counts, edge difference ratio, validation results, processor/router IDs, and warnings.

## Generation Modes

`application_only`

Changes the generated application jobs/messages/DAG structure while keeping the original platform unchanged. Job execution targets in `can_run_on` are repaired or regenerated so they contain only processors.

`platform_only`

Changes only the platform nodes and links. The application message edges are kept unchanged, but `can_run_on` is repaired because processor IDs may change.

`both`

Changes both the application DAG and the platform. The final `can_run_on` lists are generated from the new platform processor IDs.

## Important `can_run_on` Rule

`can_run_on` must contain only processor node IDs.

Processor nodes are platform nodes where:

```json
{ "is_router": false }
```

Router nodes are platform nodes where:

```json
{ "is_router": true }
```

Router IDs must never appear in `can_run_on`. If the input JSON contains router IDs or unknown node IDs in `can_run_on`, the generator treats them as invalid, removes them, and records a warning in the summary. If a job has no valid processors left, the tool resamples from the available processor IDs.

## How DAG Generation Works

The main DAG generation strategy assigns tasks to topological layers. Message edges are generated only from earlier layers to later layers, which guarantees acyclicity by construction.

A separate topological sort and cycle check is still performed before writing output.

Supported application layout strategies:

- `layered_random`
- `wide_front_narrow_end`
- `narrow_front_wide_middle`
- `chain_plus_branches`
- `fork_join`
- `random_dag`

The generator tries to respect configured in-degree, out-degree, average degree, and message count settings. If exact constraints cannot all be satisfied, it generates the closest valid DAG and records a warning.

## Validation Checks

Before writing a variant, the tool validates:

- same number of tasks as the original input
- unique task IDs
- valid message sender and receiver task IDs
- no message self-loops
- no duplicate message edges if enabled
- application graph is a DAG
- generated edge set differs from the original edge set enough
- platform node and link references are valid
- platform has no invalid self-loop or duplicate links
- platform is connected if required
- `can_run_on` contains only processor IDs
- router IDs never appear in `can_run_on`

Invalid variants are not written. The generator retries up to `validation.max_generation_attempts`.

## How To Run

Run commands from the repository root.

Dry run:

```bash
python3 dag_variant_generator/generate_dag_variants.py \
  --input path/to/example_70T.json \
  --config dag_variant_generator/dag_variant_config.json \
  --dry-run \
  --verbose
```

Generate variants:

```bash
python3 dag_variant_generator/generate_dag_variants.py \
  --input path/to/example_70T.json \
  --config dag_variant_generator/dag_variant_config.json \
  --output generated_variants \
  --num-variants 5 \
  --mode application_only \
  --seed 42 \
  --verbose
```

Generate both application and platform:

```bash
python3 dag_variant_generator/generate_dag_variants.py \
  --input path/to/example_70T.json \
  --config dag_variant_generator/dag_variant_config.json \
  --output generated_variants_both \
  --num-variants 5 \
  --mode both \
  --seed 42
```

## Config Highlights

Important fields in `dag_variant_config.json`:

- `generation_mode`: `application_only`, `platform_only`, or `both`
- `num_variants`: number of variants to generate
- `random_seed`: deterministic base seed
- `application.layout_strategy`: DAG layout strategy
- `application.in_degree`: minimum, maximum, and target average in-degree
- `application.out_degree`: minimum, maximum, and target average out-degree
- `application.message_size_range`: inclusive generated message size range
- `application.processing_times_range`: inclusive generated processing time range
- `application.deadline_range`: inclusive generated deadline range
- `application.can_run_on_policy`: use `resample_processors_only` to sample only processor IDs
- `platform.topology_strategy`: platform topology generator
- `validation.min_edge_difference_ratio`: minimum required edge-set difference from the original DAG

## Repository Usage Note

This tool is self-contained in `dag_variant_generator/` and does not require modifying any other files in the repository.
