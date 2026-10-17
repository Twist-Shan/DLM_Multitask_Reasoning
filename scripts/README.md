# Script index

Run scripts from the repository root after `python -m pip install -e .`. This directory retains flat paths because runners and analysis scripts import one another. Grouping below describes execution dependencies without changing those imports.

## Answer-only precommit workflow

See [task and data contracts](../docs/precommit.md). All versioned runners have `--help`; `--prepare-only` on v2.2 requires no model.

| Stage | Entry point | Input / output |
|---|---|---|
| v1/v2 calibration | `run_precommit.py`, `run_precommit_v2.py` | Internally generated tasks; distinct task/ending contracts |
| Wording/slot factorial and initial patching | `run_precommit_v21.py` | Fixed all-mask answer canvas |
| Robustness grid | `run_precommit_v22.py` | Compensated Sources, layers, independent drift, resampling |
| Current-code package | `package_precommit_code.py --out FILE.zip` | Allowlisted code/tests/dependencies with hashes |
| Bounded A100 suite | `colab_precommit_v22_suite.py` | Colab-specific paths; saves complete or partial archive |
| Full audit | `analyze_precommit_v22.py RUNS --out FILE` | Complete two-model supervisor layout and code ZIP |
| Candidate diagnostic | `audit_precommit_v22_candidates.py RUNS --out FILE` | iLLaDA saved probabilities; candidate versus Target changes |
| Carry/borrow diagnostic | `audit_precommit_v22_carry.py RUNS --out FILE` | iLLaDA saved probabilities; tens-only effect, split summaries |
| Dependency / position / space exploration | `run_precommit_v24.py --out RUN` | Independent A/C parents, fixed B position, B-first interventions, space sweep |
| Bounded v2.4 A100 suite | `colab_precommit_v24_suite.py` | iLLaDA; private archive even after failure |
| v2.4 audit | `analyze_precommit_v24.py RUN --out FILE` | One completed model folder; probability replay and group-paired contrasts |
| v2.4 figures | `plot_precommit_v24.py SUMMARY --out DIR` | Audited summary JSON; dependency, persistence, and space plots |

Mediation, projected-state interventions, and B-first repeated-forward patching are not implemented in v2.2. Sustained here means multiple layer outputs inside one forward; it is distinct from sustained decoding interventions below.

## Earlier generated-intermediate mechanism workflow

| Stage | Script | Input | Output |
|---|---|---|---|
| Components, heads, states | `run_mechanism_pilot.py` | Internally generated cases; pinned iLLaDA | `runs/mechanism-pilot/` |
| Component audit | `analyze_mechanism_pilot.py RUN --out FILE` | Complete component run | Recomputed summary JSON |
| Component/state figures | `plot_mechanism_pilot.py RUN` | Audit JSON, manifest, representations | Figures and distance summary in RUN |
| Head-bank selection/validation | `run_headbank_pilot.py` | Previous component run, new cases | `runs/headbank-pilot/` |
| Head-bank audit | `analyze_headbank_pilot.py RUN PREVIOUS --out FILE` | Both complete runs | Recomputed summary JSON |
| Head-bank figure | `plot_headbank_pilot.py RUN` | Audited head-bank run | Figure in RUN |

## Earlier controlled interfaces

| Purpose | Runner | Audit |
|---|---|---|
| Single residual patch | `run_shared_causal.py --out RUN --n 6 --seed 123` | `analyze_shared_causal.py --run RUN --out FILE` |
| Sustained/branch control | `run_shared_followup.py --out RUN --n 10 --seed 124` | `analyze_shared_followup.py --run RUN --out FILE` |
| Prompt/layout calibration | `run_baseline_repair.py` | `analyze_baseline_repair.py runs/baseline-repair --out FILE` |

The no-argument runners start inference immediately. They do not implement `--help`, arbitrary output paths, or resume. Do not run them as CLI discovery commands.

## Behavioral tasks

| Stage | Entry point |
|---|---|
| Pinned upstream graph export | `generate_igsm_pairs.py --help` |
| Ordinary-integer explicit equations | `generate_igsm_explicit.py --help` |
| Natural-text legacy runner | `run_igsm_text.py --help` |
| Explicit-equation runner | `run_igsm_explicit.py --help` |
| Late output-token interventions | `run_answer_interp.py --help` |
| Behavior summary and audit | `summarize_igsm.py`, `audit_igsm_explicit.py` |
| Screen-level analysis and plots | `analyze_screen30.py`, `plot_screen30.py` |

The screen-level utilities expect the collected-run directory layout described in their `--artifacts` argument handling. They are not interchangeable with component/head-bank analysis. All consume your own generated inputs or outputs; no historical measurements are bundled.

## Free-generation workflow

See [protocol and commands](../docs/free_generation.md) for generation, continuation, Source/Target patching and commitment audits. The clean Colab launcher is `notebooks/Run_Free_Reasoning.ipynb`. Historical results remain private.
