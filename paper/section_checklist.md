# Section Checklist

Use this checklist while turning `IEEE_Access_Paper_Template.tex` into the final manuscript.

## Global Checks

- [ ] Every technical claim is supported by repository evidence or marked `Author Input Required`.
- [ ] No unsupported methods are claimed: reliability, redundancy, fault injection, graph partitioning, energy optimization, or formal deadline guarantees.
- [ ] All final results come from explicitly named run folders.
- [ ] All plots are regenerated/exported consistently for publication.
- [ ] All acronyms are defined on first use.
- [ ] IEEE Access formatting, reference style, figure captions, and table captions are checked.

## Abstract

- [ ] State the implemented problem: GA-based task scheduling on application/platform graphs.
- [ ] State the two-level optimization idea without overclaiming.
- [ ] Mention actual metrics: makespan, runtime, repeat stability, validation behavior.
- [ ] Add numeric results only after authors select final experiment artifacts.

## Introduction

- [ ] Motivate task scheduling with communication costs.
- [ ] Identify the stochastic hyperparameter sensitivity problem.
- [ ] State contributions supported by the repository.
- [ ] Avoid claiming deadline, reliability, redundancy, or fault-tolerant scheduling.

## Related Work

- [ ] Review GA/evolutionary task scheduling for DAGs.
- [ ] Review communication-aware multiprocessor scheduling.
- [ ] Review hyperparameter tuning/meta-optimization for evolutionary algorithms.
- [ ] Review benchmarking and statistical comparison for stochastic optimizers.
- [ ] Fill the literature comparison table with cited papers.

## System Model

- [ ] Define application jobs, messages, processing times, and message sizes.
- [ ] Define platform processors, routers, links, and paths.
- [ ] State instance sizes from `example_*T.json`.
- [ ] State that deadlines and `can_run_on` fields exist but are not enforced by the current implementation.

## Mathematical Model

- [ ] Define task set, message set, processor set, platform graph, and candidate paths.
- [ ] Define processor assignment and task order variables.
- [ ] Define communication delay as message size plus selected path cost.
- [ ] Define start/end time reconstruction.
- [ ] Define makespan objective.
- [ ] Define outer weighted normalized objective.
- [ ] Do not include equations for unimplemented constraints.

## Proposed Method

- [ ] Explain inner chromosome segments.
- [ ] Explain initialization, selection, crossover, mutation, and replacement.
- [ ] Explain schedule reconstruction and fallback path behavior.
- [ ] Explain outer hyperparameter vector, bounds, elitism, random immigrants, repeated inner runs, and unseen-seed validation.
- [ ] Include pseudocode for the inner GA, outer GA, and comparison pipeline.

## Experimental Setup

- [ ] List hardware/software environment. Author Input Required.
- [ ] List Python libraries: DEAP, NetworkX, Matplotlib, SciPy, NumPy.
- [ ] List config files used.
- [ ] List instance sizes and selected runs.
- [ ] Explain random seed strategy.
- [ ] Explain metrics and statistical tests.

## Results And Discussion

- [ ] Report static standalone baseline statistics.
- [ ] Report nested GA tuning behavior.
- [ ] Report validation on unseen seeds.
- [ ] Report tuned-vs-standalone comparisons.
- [ ] Discuss runtime/makespan tradeoff.
- [ ] Discuss stochastic variability and repeat stability.
- [ ] Avoid conclusions unsupported by the selected logs.

## Limitations

- [ ] State that job-specific processor eligibility is not enforced.
- [ ] State that deadline metadata is not part of the objective/constraints.
- [ ] State that reliability/fault tolerance/redundancy are not implemented.
- [ ] State that benchmark instances and external baselines require author-provided context.

## Conclusion

- [ ] Summarize the nested GA scheduler contribution.
- [ ] Mention supported empirical outcomes only.
- [ ] Point to future work grounded in actual gaps.
