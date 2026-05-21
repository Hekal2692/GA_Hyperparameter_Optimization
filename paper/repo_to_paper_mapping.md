# Repository-To-Paper Mapping

This file maps paper claims to repository evidence. It also identifies items that appear in input data but are not implemented as optimization constraints.

## Core Evidence

| Paper Element | Repository Evidence | Supported Claim | Limitations / Author Input Required |
|---|---|---|---|
| Project objective | `README.md` | GA-based task scheduling with optional outer GA hyperparameter tuning. | The broader application domain must be written by authors. |
| Inner scheduler GA | `GAImplementation.py`, `SchedulerGa.py` | The scheduler searches for schedules minimizing makespan. | The implementation is stochastic; final reported claims require selected run folders. |
| Public scheduler API | `SchedulerGa.py` | `run_scheduler_ga` wraps the legacy inner scheduler implementation. | None. |
| Static benchmark | `SchedulerGa.py`, `ga_config.static_baseline.json`, `logs/example_*standalone_scheduler_ga_*` | Fixed-hyperparameter scheduler GA can be repeated to estimate stochastic performance and stability. | Repeat counts vary across existing logs. Authors must choose final artifacts. |
| Nested outer GA | `OptimizerGa.py`, `ga_config.json`, `logs/multiobjectiverun_*` | Outer GA evolves inner-GA hyperparameters. | Final novelty framing needs author wording. |
| Comparison pipeline | `CompareGaRuns.py`, `SchedulerGa.py`, `logs/ga_comparison_batch_*` | Static and tuned nested runs can be compared by instance, including makespan/runtime distributions, Welch t-test, and Cohen's d. | Some older summary fields are missing mode labels; authors should prefer newer logs with explicit validation/training source. |
| Architecture diagrams | `Charts/*.mmd` | Existing diagrams document module workflow and experiment workflow. | Convert/redraw for publication. |

## Implemented Models

| Model | Evidence | Paper Treatment |
|---|---|---|
| Application model | `example_*T.json`, `construct_task_dag_from_json`, `extract_message_list` | Define jobs/tasks and directed messages as a DAG-like precedence model. |
| Platform model | `example_*T.json`, `construct_graph_from_json`, `load_problem` | Define processors and routers as an undirected graph. |
| Processing time | `processing_times` field read in `construct_task_dag_from_json` | Use as task execution time. |
| Communication size | `messages[].size` read in `extract_message_list` | Use as base communication cost. |
| Routing/path choice | `diverse_k_shortest_paths`, `load_problem`, `ComputeMappingsAndPaths`, `find_suitable_paths` | Use k shortest simple paths with router-count path cost; path index is part of the genome. |
| Precedence reconstruction | `reconstruct_schedule_with_precedenceX_updated` | Tasks start after predecessors and processor availability allow. |
| Makespan | `compute_makespan` and evaluation functions | Primary inner objective. |
| Runtime | `perf_counter` in `SchedulerGa.py` and `OptimizerGa.py` | Secondary outer objective and experimental metric. |

## Genome And Optimization Mapping

| Optimization Layer | Genes / Variables | Operators | Fitness / Objective |
|---|---|---|---|
| Inner scheduler GA | task order permutation; processor allocation; message priority ordering; message path index list | tournament selection; ordered crossover for permutations; uniform crossover for allocation/path genes; swap/shuffle/random-reset mutation | minimize schedule makespan |
| Outer hyperparameter GA | `pop_size`, `cxpb`, `mutpb`, `ngen`, four inner mutation probabilities | tournament selection; gene-wise random crossover; random mutation within configured bounds; elitism; random immigrants | minimize normalized weighted score combining mean makespan and mean runtime |

## Input Instances

| Instance | Jobs | Messages | Platform Nodes | Processors | Routers | Links |
|---|---:|---:|---:|---:|---:|---:|
| `example_30T.json` | 30 | 35 | 17 | 8 | 9 | 20 |
| `example_40T.json` | 40 | 47 | 17 | 8 | 9 | 20 |
| `example_50T.json` | 50 | 57 | 17 | 8 | 9 | 20 |
| `example_60T.json` | 60 | 68 | 17 | 8 | 9 | 20 |
| `example_70T.json` | 70 | 82 | 17 | 8 | 9 | 20 |
| `example_80T.json` | 80 | 92 | 17 | 8 | 9 | 20 |
| `example_90T.json` | 90 | 103 | 17 | 8 | 9 | 20 |
| `example_100T.json` | 100 | 114 | 17 | 8 | 9 | 20 |

## Representative Existing Results

These are existing artifacts, not final paper claims. Authors should choose one consistent experimental campaign before submission.

| Artifact | Evidence Available | Example Values Observed |
|---|---|---|
| Nested 50T run | `logs/multiobjectiverun_50T_20260428_141321` | best mean makespan 331; avg inner runtime about 12.52 s |
| Nested 70T run | `logs/multiobjectiverun_70T_20260428_153327` | best mean makespan 418; avg inner runtime about 5.38 s; validation candidate means 444, 430.14, 439.14 |
| Standalone 50T runs | `logs/example_50T_standalone_scheduler_ga_*` | repeated logs show mean makespan 356.67 with 3 repeats in several run folders |
| Standalone 70T runs | `logs/example_70T_standalone_scheduler_ga_*` | repeated logs show mean makespan 466.33 with 3 repeats in several run folders |
| Comparison batch | `logs/ga_comparison_batch_20260424_085523` | example 50T and 70T validation/training comparisons with Welch p-values and improvement percentages |

## Unsupported Or Partially Supported Topics

| Topic | Repository Status | Paper Instruction |
|---|---|---|
| Reliability modeling | No implementation found in source. | Do not claim. Mark as Author Input Required or future work only. |
| Fault handling / fault injection | No implementation found in source. | Do not claim. |
| Redundancy mechanisms | No implementation found in source. | Do not claim. |
| Graph partitioning | No implementation found in source. | Do not claim. |
| Deadline satisfaction | `deadline` exists in JSON inputs, but no scheduler constraint or objective enforcement was found. | Mention only as available input metadata not used by the current implementation, unless authors add evidence. |
| Processor eligibility | `can_run_on` exists in JSON inputs, but processor-allocation genes choose from all non-router processors and do not enforce job-specific eligibility. | Mark as Author Input Required if discussed. |
| Energy or power | No implementation found. | Do not claim. |
| Real-time schedulability guarantees | Not implemented as formal proof or constraint. | Do not claim. |
| Parallel outer GA evaluation | Static benchmark can run multiple input instances with `ProcessPoolExecutor`; inner/outer individual evaluations are not broadly parallelized in the inspected code. | Discuss only the implemented standalone multi-instance parallel execution if relevant. |

## Recommended Claim Boundary

Safe core claim:

The repository provides a nested GA framework for communication-aware task scheduling on processor/router platform graphs, where an inner GA constructs schedules and an outer GA tunes scheduler hyperparameters using repeated stochastic evaluations and a normalized weighted makespan/runtime objective.

Claims requiring author input:

- domain-specific motivation
- comparison against published methods
- final statistical significance statement
- generalization beyond the provided synthetic/example instances
- deadline-aware or eligibility-aware scheduling
