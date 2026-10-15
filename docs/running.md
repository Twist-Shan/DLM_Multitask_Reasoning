# Execution workflows

All commands below run from the repository root after editable installation. Model experiments require the [GPU environment](environment.md); audits use existing local files. Scripts do not fetch historical experimental results.

The current answer-only workflow is documented in [precommit experiments](precommit.md): pure-addition task graphs, native ending, a wording/slot factorial, Source/Target controls at t=0, and saved-probability diagnostics. Its clean A100 entry point is `notebooks/Run_Precommit_v22.ipynb`. The workflows below retain the earlier explicit-intermediate and free-generation contracts.

## 1. Offline harness check

```sh
python -m unittest discover -s tests -v
python -m dlm_coupling smoke --out runs/smoke
```

The smoke command uses the legacy mock fixed-slot harness. It writes a dataset, run manifest, token records, and a local summary under `runs/smoke`. Use a fresh or empty directory. Its outputs do not represent a language model.

## 2. Component and representation pipeline

```sh
python scripts/run_mechanism_pilot.py
python scripts/analyze_mechanism_pilot.py runs/mechanism-pilot --out runs/mechanism-pilot/analysis.json
python scripts/plot_mechanism_pilot.py runs/mechanism-pilot
```

The runner generates paired arithmetic inputs internally: seed 127 for discovery, seed 128 for independent validation. Model-generated C/U/V values precede A/B generation. It scans residual, attention, and MLP outputs at specified layers/positions, evaluates single query heads, and captures hidden states. See [the task contract](methods.md) for the exact intervention meanings.

Local outputs include `manifest.json`, `results.jsonl`, and `representations.npz`. The audit recomputes arithmetic, submitted tokens, Source/Target alignment, qualification, candidate selection, and counters. The plot builder requires the resulting `analysis.json` plus state arrays; it produces component/head figures and relative-L2 distance summaries. These are generated files, not repository assets.

## 3. Head-bank pipeline

```sh
python scripts/run_headbank_pilot.py
python scripts/analyze_headbank_pilot.py runs/headbank-pilot runs/mechanism-pilot --out runs/headbank-pilot/analysis.json
python scripts/plot_headbank_pilot.py runs/headbank-pilot
```

Run step 2 first: the bank runner reads `runs/mechanism-pilot/results.jsonl` and reconstructs its discovery states. It searches predefined layer/head groups, freezes a candidate using discovery criteria, and generates new validation inputs with seed 129. Source/Target partial-answer alignment is recomputed during sustained patching. The audit requires both runs.

Both runners use fixed output directories and refuse to create them if they already exist. They do not implement resume. Preserve existing files and use another clean checkout for a new run; do not delete failed traces merely to obtain a successful summary. Analysis commands can replace their explicitly specified summary file. Plot labels currently follow the supplied pilot design; changing sample sizes requires checking figure denominators as well as analysis logic.

## 4. Earlier intermediate controls and prompt calibration

These stages generate their own inputs and can be run independently of steps 2–3:

```sh
python scripts/run_shared_causal.py --out runs/shared-causal --n 6 --seed 123
python scripts/analyze_shared_causal.py --run runs/shared-causal --out runs/shared-causal/analysis.json
python scripts/run_shared_followup.py --out runs/shared-followup --n 10 --seed 124
python scripts/analyze_shared_followup.py --run runs/shared-followup --out runs/shared-followup/analysis.json
python scripts/run_baseline_repair.py
python scripts/analyze_baseline_repair.py runs/baseline-repair --out runs/baseline-repair/analysis.json
```

Single/sustained controls distinguish explicit gold-intermediate conditions from permanently masked intermediate slots. Calibration compares prompt examples, canvas layouts, generated intermediates, and natural short reasoning. These interfaces have different information and generation budgets; do not combine them as one uniform accuracy benchmark.

## 5. Behavioral question relationships

First obtain the upstream iGSM generator separately. The export tools require its unmodified revision `a1ed1d04600add811beb08b58d9912ed30999642`. The explicit-equation exporter expects the checkout at `data/upstream/igsm`; the older pair exporter also accepts `--upstream`. Install that project's own dependencies before export.

```sh
python scripts/generate_igsm_explicit.py --split screen --seed 123 --per-relation 10 --out data/generated/explicit-screen.jsonl
python scripts/run_igsm_explicit.py --model illada --data data/generated/explicit-screen.jsonl --out runs/behavior-illada --single-budget 1024 --tokens-per-step 1 --prompt-version v2 --sampler blockwise --block-size 32 --seed 123
python scripts/run_igsm_explicit.py --model dream --data data/generated/explicit-screen.jsonl --out runs/behavior-dream --single-budget 1024 --tokens-per-step 1 --prompt-version v2 --sampler blockwise --block-size 32 --seed 123
```

This samples explicit equations before model evaluation. For disjoint follow-up sets, supply prior data files through the generator's repeatable `--exclude` argument and use a new prespecified seed. Explicit-equation tasks use ordinary integers; the older natural-text pipeline has a distinct modular-arithmetic contract.

Late answer-token intervention consumes a completed behavioral run:

```sh
python scripts/run_answer_interp.py --run runs/behavior-illada --out runs/behavior-illada-interventions --per-relation 3
```

It selects eligible same-block answer spans, forks the recorded state, checks exact unmodified continuation, changes a visible answer numeral, and completes generation. It does not locate an internal shared-computation circuit. Screen-wide collectors expect specific combined-run layouts; inspect `analyze_screen30.py` and its `--help` before combining independently named run directories.

## Retaining a local experiment

Save the actual repository commit, model revision, environment, run manifest, task inputs, output traces, and audit files. The current repository does not include a full environment lock or historical result packages. Discovery trials and repeated conditions on a base case are not independent samples. All generated directories are ignored by Git; no script uploads them automatically.

## Free-generation workflow

See [protocol and commands](free_generation.md) for generation, continuation, Source/Target patching and commitment audits. The clean Colab launcher is `notebooks/Run_Free_Reasoning.ipynb`. Historical results remain private.
