# DLM Multitask Reasoning

Controlled experiments on how diffusion language models answer related questions, use intermediate states, and transmit information through attention heads.

The code covers three task relationships—**independent**, **shared intermediate**, and **upstream–downstream**—with iLLaDA and Dream adapters, behavioral evaluation, activation patching, hidden-state measurements, and multi-layer head-bank interventions. The mechanistic pipeline currently uses iLLaDA; Dream is supported by the behavioral runners.

**Start here:** [Quick start](#quick-start--no-model-or-gpu) · [Execution workflows](docs/running.md) · [Methods and task contract](docs/methods.md) · [Script index](scripts/README.md) · [Validation](docs/validation.md)

This is a **code-only research repository**. Model-run records, empirical results, research reports, figures, activation arrays, and pretrained weights are maintained separately. The examples below define tasks and expected arithmetic; they are not model results.

## Research problem

When two answers depend on the same quantity, does a model use an intermediate state that influences both outputs? Which positions, layers, and attention-head groups can transmit that information, and how selective is the effect?

| Relationship | Illustrative task | Dependency |
|---|---|---|
| Independent | `A=3+4; B=5*2` | The answers have disjoint computational dependencies. |
| Shared intermediate | `C=3+4; A=C+2; B=2*C+1` | Both answers depend on C, but neither answer depends on the other. |
| Upstream–downstream | `A=3+4; B=2*A+1` | A is an input to B. |

Behavioral experiments compare separate A/B calls with joint AB/BA calls over the same facts. The controlled mechanistic task adds branch-specific intermediates:

```text
C = x + y       U = u + r       V = v + s
A = C + U                      B = 2*C + V
```

A change `C → C+1` should change `(A,B)` by `(1,2)`; `U → U+1` should change only A; `V → V+1` should change only B. These arithmetic expectations make it possible to distinguish a task-consistent counterfactual from an arbitrary answer change.

The generator defines sharing in the **task graph**. That alone does not establish internal computation reuse. The fixed-slot pipeline also schedules intermediate generation before answers; it does not measure autonomous planning.

## Repository structure

```text
DLM_Multitask_Reasoning/
├── README.md
├── pyproject.toml                  Installable dlm-coupling package and CLI
├── requirements/                   Model and figure dependencies
├── docs/
│   ├── running.md                  Ordered workflows and input/output contracts
│   ├── methods.md                  Task, prompts, patching, metrics, limitations
│   ├── environment.md              CPU/GPU setup and pinned model revisions
│   └── validation.md               Software checks and their scope
├── scripts/                        Experiment runners, audits, and plot builders
├── notebooks/                      Code-only Colab entry point
├── configs/                        Model adapters and legacy fixed-slot presets
├── src/dlm_coupling/
│   ├── tasks/                      Arithmetic graphs, prompts, and validators
│   ├── backends/                   Mock, iLLaDA, and Dream forward adapters
│   ├── text_generation.py          Full-sequence and blockwise generation
│   ├── shared_causal.py            Single-forward residual patching
│   ├── shared_followup.py          Sustained patching and branch controls
│   ├── mechanism_pilot.py          Component/head scans and state capture
│   └── headbank_pilot.py           Multi-layer query-head output patching
└── tests/                          Offline mechanical and synthetic-tensor tests
```

Script paths and historical data schemas are retained so that existing calls remain usable. [The script index](scripts/README.md) groups entry points by workflow and distinguishes current mechanism experiments from legacy interface checks.

## Quick start — no model or GPU

Python **3.10+** is required; CI uses Python **3.12**. From the repository root:

```sh
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS, or `.\.venv\Scripts\Activate.ps1` in PowerShell. Then:

```sh
python -m pip install -e .
python -m unittest discover -s tests -v
python -m dlm_coupling smoke --out runs/smoke
```

The core package uses the standard library. Three tests require optional PyTorch; without it, they are skipped. To run the complete CPU suite, install the appropriate CPU build as described in [environment setup](docs/environment.md).

The smoke command generates synthetic inputs, runs the **mock backend**, and writes local traces and a summary. It validates plumbing rather than pretrained-model reasoning. Use a new or empty output path; rerunning into a nonempty directory is rejected. Outputs remain Git-ignored.

## GPU experiments

[Open the code-only Colab notebook](https://colab.research.google.com/github/Twist-Shan/DLM_Multitask_Reasoning/blob/main/notebooks/Run_Mechanism_Headbank.ipynb)

Use a CUDA PyTorch environment; the supplied mechanism configuration targets an A100 40GB with BF16 and batch size 1. Model downloads and GPU resources are required. The notebook clones this repository; set its `REVISION` to an exact commit for a reproducible checkout. It contains no embedded source archive, historical outputs, or automatic upload step.

Install dependencies after selecting your PyTorch/CUDA build:

```sh
python -m pip install -e .
python -m pip install -r requirements/models.txt -r requirements/analysis.txt
```

Run the component stage before the head-bank stage:

```sh
python scripts/run_mechanism_pilot.py
python scripts/analyze_mechanism_pilot.py runs/mechanism-pilot --out runs/mechanism-pilot/analysis.json
python scripts/run_headbank_pilot.py
python scripts/analyze_headbank_pilot.py runs/headbank-pilot runs/mechanism-pilot --out runs/headbank-pilot/analysis.json
```

These two runners generate their own task inputs. They require new output directories and do not expose a general-purpose resume option. The head-bank stage reads the preceding component run. Plotting, earlier causal controls, and behavioral-data preparation are documented in [execution workflows](docs/running.md).

## Experiment map

| Study | Entry point | Input / role |
|---|---|---|
| Behavioral task relationships | `run_igsm_explicit.py` | Generated explicit-equation pairs; iLLaDA or Dream; separate and joint questions |
| Late answer-token effects | `run_answer_interp.py` | A completed behavioral run with recorded token trajectories |
| Single versus sustained intermediate patching | `run_shared_causal.py`, `run_shared_followup.py` | Generated arithmetic cases; explicit or masked intermediate slots |
| Prompt and layout calibration | `run_baseline_repair.py` | Separately generated calibration and held-out cases |
| Components and representations | `run_mechanism_pilot.py` | Residual / attention / MLP positions, single query heads, state capture |
| Multi-layer head banks | `run_headbank_pilot.py` | Component discovery states plus independently generated validation cases |

The mechanism defaults use seed 127 for discovery, 128 for component validation, and 129 for head-bank validation. Earlier stages use 123–126. Seeds are explicit and unrelated to dates. Conditions and repeated interventions on the same base case are not independent samples.

## Patching and measurement

**Source** supplies activations; **Target** receives them. Internal legacy variable names are retained for compatibility. Single patching acts on one forward; sustained patching acts on every answer-generation forward, recomputing Source states with its partial A/B tokens aligned to Target.

A head bank is a set of `(layer, query-head)` pairs. The intervention replaces selected head outputs at the input to `o_proj`, at specified answer positions; it does not replace attention probabilities. A bank with eight layers and sixteen heads per layer contains 128 layer–head units.

| Measurement | Meaning and qualification |
|---|---|
| Pair accuracy | Both final integer answers must match the arithmetic target; invalid output is not success. |
| Newly induced counterfactual | Both patched answers match the specified counterfactual, excluding cases where the unpatched output already matched it. |
| Unaffected-branch preservation | The other branch equals its unpatched output; preservation does not imply correctness if the baseline was wrong. |
| Relative hidden-state L2 change | Describes activation differences; it is not evidence of decodability or causal use by itself. |
| Source-state interchange | Tests sufficiency under a specified intervention; it does not establish necessity, minimality, or compute-once reuse. |

See [methods](docs/methods.md) for prompt construction, mask layout, counterfactual formulas, controls, and layer/head indexing.

## Validation and generated files

CI runs offline unit tests using CPU PyTorch. Tests cover arithmetic invariants, parsers, tokenizer/logit alignment, token submission, recovery rules in the legacy CLI, and intervention hooks on synthetic tensors. They do not load pretrained weights or establish model performance. The [validation record](docs/validation.md) lists checks and unverified scope.

`data/`, `runs/`, `reports/`, `.env`, and `configs/local/` are Git-ignored. Analysis and plotting tools consume files generated by your own runs. There are no bundled experimental scores or activation caches, and no command uploads results.

## Sources and licensing

Model adapters use [iLLaDA](https://huggingface.co/GSAI-ML/iLLaDA-8B-Instruct) and [Dream](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B); checkpoint identifiers are listed in [environment setup](docs/environment.md). The iGSM export tools require a separately obtained upstream generator at the revision checked by the code. Adapted explicit-equation tasks are not original natural-language benchmark scores.

No project-wide software license has been selected. Public visibility alone does not grant a software license; model, upstream-generator, and dependency use remains subject to their respective terms.
