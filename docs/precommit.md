# Answer-only pre-commitment experiments

This workflow tests whether a still-masked answer region can affect another answer before any answer token is committed. It uses existing iLLaDA and Dream checkpoints and explicit arithmetic. This page specifies software and task contracts; empirical results and historical run archives are not included.

## Task and canvas

Define `C=x+y`, `D=w+z`, `U=u+r`, `V=v+s`, and `A=C+U`. The addition family uses `B=D+V` (independent), `B=C+V` (shared), or `B=A+V` (downstream). All facts and both definitions remain in single-question prompts. Gold values are used for scoring, not written into the model input.

The numeric answer scaffold has two digit positions per requested answer, plus optional unused masks before the model's native chat suffix. Tokenization is audited in the complete serialized context over values 10–99. The suffix is extracted from the native chat template. Unused masks remain masked. Dream reads answer position j from raw-logit position j−1; iLLaDA reads j.

An illustrative downstream prompt is:

```text
Use ordinary integer arithmetic.
Facts: x = 4; y = 5; w = 7; z = 6; u = 2; r = 6; v = 5; s = 7.
Definition of A: A = (4 + 5) + (2 + 6).
Definition of B: B = A + (5 + 7).
Requested outputs: A, B.
Fill only the requested final integer values in the supplied answer template. No explanation. Do not output intermediate values.
```

The arithmetic answers are A=17, B=29. The supplied canvas is `A = [M][M]; B = [M][M].[M][M]` plus the native suffix; `[M]` denotes the actual mask token ID. This is an explanatory task example, not a recorded model output.

## Implemented versions

| Version | Module / runner suffix | Contract |
|---|---|---|
| v1 | `precommit` | Multiplicative B definitions; historical calibration and masked-region interfaces |
| v2 | `precommit_v2` | Addition family, native suffix, answer-token audit, single-question calibration |
| v2.1 | `precommit_v21` | Requested wording × physical slots; initial-forward A-to-B patching |
| v2.2 | `precommit_v22` | Compensated Sources, independent drift, adjacent-layer and sustained-layer interventions, resampling |

The v2.2 runner does not implement prompt/A-region mediation, subspace interventions, or B-first persistence. Those are future experiments, not alternate flags on this runner. Historical source files remain separate so that a new experiment cannot silently change an earlier protocol.

## Offline preparation

Python 3.10+; install the package from the repository root. The following command generates and audits the task families and intervention specification without loading a tokenizer or model:

```sh
python scripts/run_precommit_v22.py --model illada --seed 123 --prepare-only --out runs/prepare-v22
python -m unittest discover -s tests -p "test_precommit*.py" -v
```

Use a new output directory. Preparation audits Target/Source family overlap, not only exact Target duplication. Reserved families flagged by the audit must be replaced and frozen in a new dataset artifact before a subsequent independent confirmation experiment. The runner does not evaluate the confirmation split. Seeds are explicit and unrelated to dates.

## A100 execution

[Open clean Colab launcher](https://colab.research.google.com/github/Twist-Shan/DLM_Multitask_Reasoning/blob/main/notebooks/Run_Precommit_v22.ipynb). Select A100 and set `REVISION` to an exact repository commit when freezing a run. The notebook installs pinned precommit dependencies in a separate environment, records the actual checkout and source hashes, runs tests, and executes the bounded supervisor. It contains no private outputs or embedded historical archive.

For a configured CUDA environment, the individual runners are:

```sh
python -m pip install -e . -r requirements/precommit.txt
python scripts/run_precommit_v22.py --model illada --seed 123 --out runs/illada-v22
python scripts/run_precommit_v22.py --model dream --seed 123 --out runs/dream-v22
```

iLLaDA runs calibration then discovery; Dream runs the discovery behavior factorial. No activation patching is performed on Dream by v2.2. Each model has a 3600-second process budget. The Colab supervisor has an outer timeout, runs the models sequentially, archives partial results after a failure, and does not upload anything. Check completion and download your archive before releasing the runtime.

`package_precommit_code.py --out PATH` creates a new code-only ZIP with per-file hashes. The Colab supervisor expects that ZIP at `/content/precommit-v22-code.zip` and copies it into `runs/code-v22.zip`. A newly packaged checkout is a new run snapshot, not the original private experiment artifact.

## Source/Target interventions and metric

Source supplies layer states; Target receives them. At t=0 all answer digits are masked. For each Target and each `sign` in `{-1,+1}`:

| Source | Operand change | Downstream truth change |
|---|---|---|
| `change_A` | x += sign | A and B change by sign |
| `same_A` | x += sign, y -= sign | Both unchanged |
| `A_change_B_fixed` | x += sign, v -= sign | A changes, Source B unchanged |
| `A_fixed_B_change` | v += sign | A unchanged, Source B changes |
| `self` | None | Both unchanged |

The candidate is always **Target B+sign**, including controls. In the independent relation this is an off-graph drift diagnostic, not the correct Source answer. All source formulas are checked mechanically.

The primary metric is the change in `mean_digit(log P(candidate digit) − log P(Target digit))` after patching, using same-forward probabilities. It is not a sequence joint probability or a final-answer accuracy. Signs are averaged within a base group before group-level bootstrap; repeated layers, signs, and resampling choices are not independent samples.

Layer numbers are one-based block outputs. A single-layer patch replaces one layer's A states. **Sustained across layers** replaces each selected layer using its corresponding clean Source state within the same forward. This differs from sustained intervention across decoding forwards in the earlier intermediate-generation workflow. v2.2 does not run patched rollouts. The complete layer/site grid and primary contrasts are in `precommit_v22.specification()`.

Same-A, self, unused positions, independent B, last-layer cross-position nulls, and compensation address different alternatives. A positive effect does not prove a unique representation, a necessary route, or computation performed once. Resample loss and slot-addition gain come from different interventions and do not define a mediation percentage.

## Audit your own saved outputs

The full audit expects the **complete supervisor layout**, including both model folders, `suite_status.json`, and `code-v22.zip`. After extracting the downloaded archive, pass the folder containing these entries:

```sh
python scripts/analyze_precommit_v22.py PATH_TO_RUNS --out analysis.json
python scripts/audit_precommit_v22_candidates.py PATH_TO_RUNS --out candidates.json
python scripts/audit_precommit_v22_carry.py PATH_TO_RUNS --out carry.json
```

The last two diagnostics require only the two iLLaDA split folders, each with `contexts.jsonl` and `patch.jsonl`; they also work after the individual iLLaDA runner. All three refuse to overwrite their output file. The full analyzer is bound to the seed-123 specification; changing the dataset or seed requires a separately versioned analysis.

The carry audit selects cases using arithmetic alone: B±1 changes B's tens digit while A±1 retains A's tens digit. It computes **the B tens position alone**, without averaging with the ones position. It saves input hashes, case identities, per-position probabilities, and separate/pool summaries. Both strict-positive counts and a 1e−6 zero tolerance are recorded so that floating-point roundoff is not counted as a substantive effect. This is a post-hoc diagnostic; arithmetic-consistent movement does not by itself establish a particular addition algorithm or circuit.

All generated datasets, model outputs, audit JSON, archives, and activation arrays remain private and Git-ignored.
