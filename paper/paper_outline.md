# IEEE Access Paper Outline

This outline is based strictly on repository evidence.

## Candidate Title

Nested Genetic Hyperparameter Optimization for Communication-Aware Task Scheduling on Processor-Router Platform Graphs

Author Input Required: final title, application domain, institutional context, and target contribution phrasing.

## Core Research Contribution

The repository supports the following contribution:

A nested genetic-algorithm framework for task scheduling in which an inner GA searches task order, processor allocation, message priority, and communication path choices, while an outer GA tunes the inner GA's population size, crossover probability, mutation probability, generation count, and mutation sub-probabilities using repeated stochastic evaluations and a weighted normalized objective over mean makespan and runtime.

## Main Novelty Supported By The Repository

- Joint scheduling genome with task order, processor assignment, message priority, and path-index components.
- Outer GA that tunes inner scheduler hyperparameters, including mutation sub-probabilities for each genome segment.
- Weighted multi-objective outer score based on normalized mean makespan and mean runtime.
- Unseen-seed validation of top tuned candidates.
- Static baseline and comparison pipeline with distribution plots, Welch t-test, and effect-size support.

Author Input Required: novelty relative to published literature.

## 1. Introduction

### A. Why This Section Is Needed

IEEE Access papers require a clear motivation, gap, contribution list, and article organization. The repository implements a scheduling optimizer and experiment pipeline, so the introduction should frame why stochastic GA scheduling and hyperparameter tuning matter.

### B. What Must Be Covered

- DAG/task scheduling with communication delays.
- Processor/router platform mapping.
- Sensitivity of GA performance to hyperparameters.
- Need for repeat-based evaluation because runs are stochastic.
- Contributions grounded in the repository.

### C. What Already Exists In The Repository

- `README.md` describes the scheduler GA, optional outer tuning GA, static baseline, comparison pipeline, and logged artifacts.
- `Charts/*.mmd` provide architecture and experiment workflows.
- `ga_config*.json` define static and nested settings.

### D. What Requires Author Input

- Real application domain and motivation.
- Why the provided instances are representative.
- Final contribution wording.
- Any quantitative headline results.

### E. Suggested Equations

None required in the introduction. Mention makespan and runtime objectives conceptually.

### F. Suggested Figures

- High-level workflow diagram adapted from `Charts/flowchart TD.mmd`.

### G. Suggested Tables

- Contribution-to-evidence table, optional.

### H. IEEE Access Writing Guidance

Keep the opening broad but technical. End the introduction with 3-5 contributions. Avoid unsupported claims about reliability, deadlines, or fault tolerance.

## 2. Related Work

### A. Why This Section Is Needed

The repository topic must be positioned against existing work in evolutionary DAG scheduling, communication-aware scheduling, and GA hyperparameter/meta-optimization.

### B. Required Related Work Subsections

1. Evolutionary Algorithms for DAG Task Scheduling
2. Communication-Aware Multiprocessor Scheduling
3. Hyperparameter Optimization and Meta-GA Methods
4. Benchmarking and Statistical Evaluation of Stochastic Scheduling Algorithms

### C. What Literature Should Be Reviewed

- GA and evolutionary methods for multiprocessor task scheduling.
- DAG scheduling with precedence and communication costs.
- Routing/path-aware scheduling where communication topology affects latency.
- Meta-optimization, nested GA, or automated tuning for evolutionary algorithms.
- Statistical comparison of stochastic optimizers using repeated runs.

### D. What Comparisons Matter

- Whether prior methods optimize only schedules or also tune scheduler hyperparameters.
- Whether communication path choice is part of the schedule representation.
- Whether runtime is included as an optimization objective.
- Whether validation uses unseen random seeds.
- Whether comparisons report statistical tests.

### E. What Limitations To Highlight

Frame limitations only after citations are added. Candidate limitation categories:

- Fixed GA hyperparameters in many scheduling studies.
- Lack of repeat-based validation.
- Limited reporting of runtime/quality tradeoffs.
- Limited integration of communication-path decisions into the chromosome.

### F. How To Position The Proposed Work

Position as a repository-implemented nested GA scheduling framework, not as a guaranteed optimal scheduler. Emphasize empirical optimization of makespan/runtime behavior.

### G. Literature Comparison Table Template

| Work | Problem Model | Scheduling Method | Communication Cost | Path Selection | Hyperparameter Tuning | Runtime Objective | Repeat/Seed Validation | Statistical Test | Repository-Relevant Limitation |
|---|---|---|---|---|---|---|---|---|---|
| Author Input Required | DAG / task graph / other | GA / heuristic / exact / hybrid | Yes/No | Yes/No | Manual / automated / none | Yes/No | Yes/No | Yes/No | Author Input Required |

### H. Suggested Equations, Figures, Tables

- No equations required.
- Table: literature comparison table above.
- Figure: none unless adding a taxonomy.

### I. IEEE Access Writing Guidance

Use recent and foundational references. Do not cite papers to justify unimplemented features. The final paragraph should explicitly identify the gap that this repository actually addresses.

## 3. System Model

### A. Why This Section Is Needed

The implementation depends on application JSON models, platform JSON models, processing times, messages, processors, routers, links, and candidate paths.

### B. What Must Be Covered

- Application jobs/tasks and messages.
- Platform graph with processors and routers.
- Processing times and message sizes.
- Candidate path generation using k shortest simple paths.
- Relationship between task mapping and communication path selection.

### C. What Already Exists In The Repository

- `example_*T.json` contain `application.jobs`, `application.messages`, `platform.nodes`, and `platform.links`.
- `GAImplementation.py` loads the application and platform model.
- `load_problem` builds NetworkX graphs and merged path dictionaries.
- Instances cover 30, 40, 50, 60, 70, 80, 90, and 100 jobs.

### D. What Requires Author Input

- Whether the instances are synthetic, generated, benchmark-derived, or domain-derived.
- Interpretation of `wcet_fullspeed`, `mcet`, `deadline`, and `can_run_on` fields.
- Whether future versions should enforce deadline or processor eligibility constraints.

### E. Suggested Equations

Application graph:

\[
\mathcal{A}=(\mathcal{T},\mathcal{M})
\]

Platform graph:

\[
\mathcal{P}=(\mathcal{V},\mathcal{E}), \quad \mathcal{V}=\mathcal{P}_{cpu}\cup\mathcal{R}
\]

Candidate paths:

\[
\mathcal{K}_{uv}=\{k_1,k_2,\ldots,k_K\}, \quad u,v\in\mathcal{P}_{cpu}
\]

### F. Suggested Figures

- Application/platform model diagram.
- Candidate path generation diagram.

### G. Suggested Tables

- Instance-size table from `repo_to_paper_mapping.md`.
- Notation table.

### H. IEEE Access Writing Guidance

Clearly separate data fields that are loaded and used from fields that are merely present in JSON. State that deadlines and `can_run_on` are not enforced by the inspected implementation.

## 4. Mathematical Model And Problem Formulation

### A. Why This Section Is Needed

The paper needs formal definitions of the implemented objective and reconstruction behavior.

### B. What Must Be Covered

- Task execution time.
- Processor assignment.
- Message communication cost.
- Path cost.
- Precedence-based start time.
- Processor availability.
- Makespan minimization.
- Outer weighted objective.

### C. What Already Exists In The Repository

- `compute_makespan` minimizes maximum end time.
- `reconstruct_schedule_with_precedenceX_updated` computes start/end times using predecessor completion, message size plus path cost, and processor availability.
- `resolve_outer_objective_weights` and `refresh_evaluation_archive_scores` normalize and combine mean makespan and runtime.

### D. What Requires Author Input

- Formal proof of feasibility or optimality, if desired. Not implemented.
- Any deadline or eligibility constraints. Not implemented.

### E. Suggested Equations

Processor assignment:

\[
x_i \in \mathcal{P}_{cpu}, \quad i\in\mathcal{T}
\]

Communication delay for message \(m=(i,j)\):

\[
d_m = s_m + c(k_m)
\]

where \(s_m\) is message size and \(c(k_m)\) is selected path cost.

Task start time:

\[
S_i=\max\left(A_{x_i}, \max_{m=(j,i)\in\mathcal{M}}\{F_j+d_m\}\right)
\]

Task finish time following the current implementation's reconstruction:

\[
F_i=S_i+p_i+\sum_{m=(j,i)\in\mathcal{M}} d_m
\]

Makespan:

\[
C_{\max}=\max_{i\in\mathcal{T}} F_i
\]

Inner objective:

\[
\min C_{\max}
\]

Outer normalized weighted objective:

\[
\min \; w_C \hat{C}_{mean}(\theta)+w_R \hat{R}_{mean}(\theta)
\]

where \(\theta\) is the outer hyperparameter vector and \(w_C+w_R=1\).

### F. Suggested Figures

- Schedule reconstruction timing diagram.

### G. Suggested Tables

- Notation table.
- Implemented constraints vs non-implemented metadata table.

### H. IEEE Access Writing Guidance

Make the equations match the code even if the authors later refine the model. Do not present unimplemented constraints as active constraints.

## 5. Proposed Nested GA Scheduling Framework

### A. Why This Section Is Needed

This is the central technical section.

### B. What Must Be Covered

- Inner scheduler chromosome.
- Inner GA operators.
- Schedule reconstruction.
- Outer GA chromosome.
- Outer objective and repeated inner evaluation.
- Validation on unseen seeds.
- Comparison pipeline.

### C. What Already Exists In The Repository

- Inner scheduler: `GAImplementation.py`.
- Static public runner: `SchedulerGa.py`.
- Outer tuner: `OptimizerGa.py`.
- Comparison: `CompareGaRuns.py`.

### D. What Requires Author Input

- Algorithm naming.
- Complexity discussion beyond empirical runtime.
- Motivation for selected hyperparameter bounds.

### E. Suggested Equations

Inner chromosome:

\[
g=[\pi, x, \rho, \kappa]
\]

Outer chromosome:

\[
\theta=[N_{pop}, p_c, p_m, N_{gen}, q_{\pi}, q_x, q_{\rho}, q_{\kappa}]
\]

### F. Suggested Figures

- Nested GA architecture from `Charts/CodeArchitetcure.mmd`.
- Experiment workflow from `Charts/Expworkflow.mmd`.

### G. Suggested Tables

- Inner genome segment table.
- Outer hyperparameter search-space table.
- Operator table.

### H. IEEE Access Writing Guidance

Use pseudocode rather than source-code-level narration. Keep implementation-specific names in parentheses when useful.

## 6. Experimental Setup

### A. Why This Section Is Needed

IEEE Access requires repeatable experiment settings.

### B. What Must Be Covered

- Dataset/instance sizes.
- Static baseline config.
- Nested GA config.
- Random seed strategy.
- Metrics: makespan, runtime, standard deviation, confidence interval, validation mean/std, Welch t-test, Cohen's d.
- Logged artifacts and plots.

### C. What Already Exists In The Repository

- Config files: `ga_config.json`, `ga_config.static_baseline.json`, small-run variants, equal-weights config.
- Logs: static, nested, validation, comparison batches.
- Result JSON files store run-level and aggregate metrics.

### D. What Requires Author Input

- Hardware/OS/Python version.
- Exact final experiment run set.
- Whether all instances from 30T to 100T should be rerun consistently.

### E. Suggested Equations

Mean:

\[
\bar{C}=\frac{1}{n}\sum_{r=1}^{n} C_r
\]

Sample standard deviation:

\[
s_C=\sqrt{\frac{1}{n-1}\sum_{r=1}^{n}(C_r-\bar{C})^2}
\]

### F. Suggested Figures

- Static convergence plot.
- Nested objective history plot.
- Runtime tradeoff plot.
- Validation boxplot.
- Static-vs-nested comparison plot.

### G. Suggested Tables

- Experiment configuration table.
- Instance table.
- Metrics table.

### H. IEEE Access Writing Guidance

Name each run folder used for final values. Separate training/search behavior from unseen-seed validation behavior.

## 7. Results And Discussion

### A. Why This Section Is Needed

This section interprets the logged artifacts.

### B. What Must Be Covered

- Static baseline performance.
- Nested GA best hyperparameters.
- Makespan/runtime tradeoff.
- Repeat stability.
- Validation on unseen seeds.
- Statistical comparison where available.

### C. What Already Exists In The Repository

- `logs/multiobjectiverun_*` include history, evaluations, best result, validation results, and plots.
- `logs/example_*standalone_scheduler_ga_*` include repeated static benchmark results.
- `logs/ga_comparison_batch_*` include tuned-vs-static comparisons.

### D. What Requires Author Input

- Final selection of which logs are authoritative.
- Interpretation of inconsistent or older log formats.
- Whether to rerun all experiments with a single reproducible protocol.

### E. Suggested Equations

Relative improvement:

\[
\Delta_C(\%)=100\frac{\bar{C}_{static}-\bar{C}_{tuned}}{\bar{C}_{static}}
\]

Runtime change:

\[
\Delta_R(\%)=100\frac{\bar{R}_{tuned}-\bar{R}_{static}}{\bar{R}_{static}}
\]

### F. Suggested Figures

- Comparison plot from `logs/ga_comparison_batch_20260424_085523`.
- Objective component plot from a selected `multiobjectiverun_*`.
- Hyperparameter trajectory plot.

### G. Suggested Tables

- Best tuned hyperparameters per instance.
- Static vs tuned comparison table.
- Validation results table.

### H. IEEE Access Writing Guidance

Avoid overgeneralization. Use "in the selected experimental runs" unless all experiments have been systematically rerun.

## 8. Limitations And Future Work

### A. Why This Section Is Needed

The repository contains unused metadata and lacks several commonly expected scheduling features.

### B. What Must Be Covered

- Deadlines exist in JSON but are not enforced.
- `can_run_on` exists in JSON but is not enforced by the current processor-allocation mutation/initialization.
- No reliability/fault/redundancy/graph-partitioning implementation found.
- Need for broader baselines and full-scale reproducible experiment campaign.

### C. What Already Exists In The Repository

- JSON fields reveal potential future constraints.
- Current code shows the active objective and constraints.

### D. What Requires Author Input

- Which limitations are acceptable for the intended paper.
- Future implementation roadmap.

### E. Suggested Equations

None unless future work is formalized.

### F. Suggested Figures

None required.

### G. Suggested Tables

- Current implementation vs future extensions table.

### H. IEEE Access Writing Guidance

Be direct. A transparent limitations section improves credibility.

## 9. Conclusion

### A. Why This Section Is Needed

Summarizes the contribution and empirical evidence.

### B. What Must Be Covered

- Nested GA scheduling framework.
- Inner genome and outer tuning.
- Evidence from logged experiments.
- Future work.

### C. What Already Exists In The Repository

All core implementation and artifacts.

### D. What Requires Author Input

Final quantitative results and paper-level conclusion.

### E. Suggested Equations, Figures, Tables

None.

### F. IEEE Access Writing Guidance

Keep concise. Do not introduce new claims.
