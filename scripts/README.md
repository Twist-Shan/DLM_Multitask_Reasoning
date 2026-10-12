# Script index

Run scripts from the repository root after `python -m pip install -e .`. This directory retains flat paths because runners and analysis scripts import one another. Grouping below describes execution dependencies without changing those imports.

## Current mechanism workflow

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
