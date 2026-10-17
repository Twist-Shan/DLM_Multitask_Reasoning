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
| v2.4 | `precommit_v24` | A/C dependency selection, parent left/right of B, B-first persistent interventions, mask/EOS space controls |

The v2.2 runner does not implement prompt/A-region mediation, subspace interventions, or B-first persistence. v2.4 adds B-first persistence; mediation and subspace interventions remain unimplemented. Historical source files remain separate so that a new experiment cannot silently change an earlier protocol.

## v2.4: dependency, position, and space

This separate iLLaDA runner extends the inspected calibration task family with an independently defined C. A and C each sum four one-digit operands. The prompt specifies either `B=A+V` or `B=C+V`, never both simultaneously. Half of the generated groups have A=C and half have different values. In this version C is the alternative parent, not the intermediate named C in v2. The generator and seed are explicit; no reserved discovery or confirmation examples are evaluated.

For example, A=(6+6)+(3+7)=22, C=(4+8)+(6+6)=24, V=4+5=9. B is 31 with parent A and 33 with parent C. The exact prompt is:

```text
Solve the arithmetic exactly.
Definition of A: A = (6 + 6) + (3 + 7).
Definition of C: C = (4 + 8) + (6 + 6).
Definition of B: B = A + (4 + 5).
Fill the labelled answer fields in the supplied response template. Each answer is a two-digit integer. Do not write intermediate steps or extra text.
```

`ABC` and `CBA` canvases hold B's absolute position and total token count fixed. Every answer digit starts masked. Change-A and change-C Sources act on one operand; same-value Sources compensate within that variable; compensated Sources offset V when the changed variable is B's parent. `B_only`, unused, self, and final-layer cross-position conditions provide additional controls. The primary contrast is related `(change−same)` minus unrelated `(change−same)`, averaged within base group. Candidate B+sign is a diagnostic off-graph candidate for unrelated-variable interventions.

The B-first decoder commits both B digits before either parent field. It compares a clean baseline, first-forward-only intervention, and intervention on every forward until B is complete. Each active forward recomputes Source states using Source facts and the current Target visible canvas; this mixed context can disagree with Source's arithmetic and is saved explicitly. After B, patching stops and A/C complete. These are manipulated commitment schedules, not claims about the model's natural order.

Space quantity uses 0/2/8/32 extra positions, filled with masks or native EOS IDs before the native chat suffix. Space usage compares a labelled A slot, labelled C slot, and length-matched mask/EOS tails. Prompt and B positions stay fixed; in usage comparisons total length and suffix offsets also match. Only B commits, always in two forwards; other masks remain masked. All usage patches replace the same two physical positions. Extra tokens change per-forward compute, so equal forward counts do not imply equal FLOPs.

```sh
python scripts/run_precommit_v24.py --seed 123 --prepare-only --out runs/prepare-v24
python scripts/run_precommit_v24.py --seed 123 --audit-only --out runs/audit-v24
python scripts/run_precommit_v24.py --seed 123 --out runs/illada-v24
python scripts/analyze_precommit_v24.py runs/illada-v24 --out runs/v24-analysis.json
python -m pip install matplotlib numpy
python scripts/plot_precommit_v24.py runs/v24-analysis.json --out runs/v24-figures
```

The first command needs no model dependencies; the second loads the real tokenizer only. Full execution needs CUDA and pinned dependencies. Every run requires a fresh output directory. `--limit` supports smoke checks; the default runs the complete generated exploration set. The analyzer replays saved digit probabilities, checks Sources and canvas alignment, audits B-first/space commitments, and bootstraps base groups rather than repeated conditions. Its output is exploratory, not a held-out confirmation.

[Open the clean v2.4 A100 launcher](https://colab.research.google.com/github/Twist-Shan/DLM_Multitask_Reasoning/blob/main/notebooks/Run_Precommit_v24.ipynb). Its supervisor expects a newly packaged code ZIP at `/content/precommit-v24-code.zip`, runs iLLaDA with a 5400-second budget, and preserves partial records on failure. Download and verify your private archive before releasing the runtime. Prompt-region mediation and subspace patching remain separate, unimplemented plans.

## v2.5 original-task controls

The core experiment reuses the exact v2 calibration facts and IDs, arithmetic definitions, numeric prompt, semicolon scaffold, and native ending. Before inference it checks token-for-token equality with the original AB canvas with two unused masks. The three relationships retain their original meanings; C still denotes x+y.

The two-answer core changes one factor at a time:

- AB versus BA canvas, with Requested outputs always A,B. Both confidence and B-first commitment policies are measured. B-first orders answer fields, while the two B digits still compete by their current confidence; it does not freeze tens-versus-ones order. AB/BA changes B's absolute position, so the raw order difference is not a position-free effect.
- On downstream questions, 0/2/8/32 extra masks or 2/8/32 EOS tokens are inserted immediately before the native ending. Question and A/B positions stay fixed. All variants commit four answer tokens; extra positions stay fixed. Equal forward count is not equal FLOPs.
- A-region Source/Target patching uses downstream and independent controls, same-value operands, B-compensated Sources, unused sites, self, and the final-layer null. B-first rollouts compare a first-forward intervention with repeated interventions until B completes.

The redundancy experiment is recorded separately. R repeats A's numeric expression in Target, but is not defined as R=A; Source can change either numeric definition independently. B is defined as either A+V or R+V. It first evaluates an added R definition with only AB output, then ABR and RBA output. The latter pair holds B's absolute position and total length fixed. Adding R is a task extension, not a replay of the original prompt. Baseline A=R does not imply shared representations, and a result here does not establish behavior for unequal candidate values.

The primary redundancy contrast is related (change−same) minus unrelated (change−same), grouped by the underlying original question. The original-task contrast is downstream (change−same) minus independent (change−same). The metric is the mean per-digit log-ratio shift for B+sign against Target B. A diagnostic off-graph candidate is not called the correct Source answer. All layers and controls are fixed in `precommit_v25.specification()`. No cases are filtered by accuracy.

```sh
python scripts/run_precommit_v25.py --prepare-only --seed 123 --out runs/prepare-v25
python scripts/run_precommit_v25.py --audit-only --seed 123 --out runs/audit-v25
python scripts/run_precommit_v25.py --seed 123 --out runs/illada-v25
python scripts/analyze_precommit_v25.py runs/illada-v25 --out runs/v25-analysis.json
```

[Open the clean v2.5 A100 launcher](https://colab.research.google.com/github/Twist-Shan/DLM_Multitask_Reasoning/blob/main/notebooks/Run_Precommit_v25.ipynb). It preserves the executed source archive, dependency versions, token maps used by the metric, raw outputs, and partial outputs on failure. The analyzer validates arithmetic, masks, compensation, direct-readout exclusion, probability replay, null controls, fixed commitment counts, and paired group-level contrasts. It reports EOS against both equal-length masks and zero extra positions, so recovery from masks is not confused with a benefit of adding space. Both the original-task and redundancy results are exploratory reuses of calibration data; the reserved confirmation set is not evaluated.

## v2.6 ARB routing and directed attention

The task generator reuses the same original calibration questions and seed 123. In the redundancy extension, Target A=R, with independent numeric definitions; B is defined as A+V or R+V in separate prompts. ARB is paired with RAB, keeping B's position and sequence length fixed while swapping the two parent slots. The v2.5 prompt remains unchanged: definitions A,R,B and requested outputs A,B,R. Consequently, this is not a fully counterbalanced test of wording or definition order.

At t=0 all answer digits remain masked. Source changes A or R by ±1. Controls include operand changes preserving the sum, a compensated Source holding B fixed, B-only changes, equal-width unused positions, self, and a final-layer cross-position null. Single-layer 30/31/32 and sustained-layer 24–31 interventions use each layer's corresponding Source state. The primary contrast holds the patched site and layout fixed: `(change−same | B depends on site) − (change−same | B depends on other site)`. Near/far comparisons then average matched names and dependency roles within each question. This runner measures t=0 patch effects and unpatched confidence/B-first baselines; it does not run patched rollouts.

The attention experiment retains the original downstream AB/BA questions. Only 0/2/32 masks or 32 EOS tail positions vary, immediately before the native ending. Extra positions stay fixed while four answer digits are committed. Information direction is **key/value sender → query receiver**. All heads are blocked by additive −∞ before softmax; the remaining keys renormalize.

| Mode | Blocked edges / layers |
|---|---|
| `native` | Original full attention, no added bias |
| `sham` | All-zero bias in all layers; controls the attention interface/numerics |
| `tail_to_original` | All original queries stop reading tail keys; all 32 layers |
| `tail_to_answers` | Only A/B digit queries stop reading tail keys; all 32 layers |
| `original_to_tail` | Tail queries stop reading original keys; a direction control, not a zero-effect null |
| `tail_to_original_early` | Original queries stop reading tail keys in layers 1–16 |
| `tail_to_original_late` | Same edge set in layers 17–32 |

All seven modes run with 32 masks and 32 EOS; 2 masks use native/sham/full isolation, while zero extra tokens use native/sham. Non-native conditions keep the official SDPA implementation and pass a bias in every layer; layers outside a selected window receive zero bias. Compare active blocks to sham and report sham against native separately. Full isolation must make equal-length mask/EOS fillers indistinguishable at the original answers, but does not remove tail length or restore the original ending position. Recovery after blocking cannot establish a benefit of added space or a specific attention-dilution mechanism.

```sh
python scripts/run_precommit_v26.py --prepare-only --seed 123 --out runs/prepare-v26
python scripts/run_precommit_v26.py --seed 123 --out runs/illada-v26
python scripts/analyze_precommit_v26.py runs/illada-v26 --out runs/v26-analysis.json
```

Optionally pass `--prior runs/illada-v25` to audit native behavior against your own earlier run. The analyzer verifies the complete condition grid, exact arithmetic, masks, patch/probability replay, nulls, directional hooks, frozen tails, four/six commitment budgets, and full-isolation filler invariance. Confidence intervals bootstrap original question groups 4000 times with seed 123. Repeated signs, sites, layouts, and policies do not enlarge the independent sample count. The [clean A100 launcher](https://colab.research.google.com/github/Twist-Shan/DLM_Multitask_Reasoning/blob/main/notebooks/Run_Precommit_v26.ipynb) saves provenance and partial outputs before the runtime is released. All results remain private and Git-ignored.

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
