# Free generation and natural-span patching

## Purpose

Compare the controlled C/U/V interface with free responses requesting necessary calculations and final answers labeled A and B. No worked examples, intermediate answers, output canvas, or instruction to reuse a sum are supplied. Numeric prompts cover independent, shared and downstream relations; symbolic shared prompts support initial localization. The default uses two groups from seed 123 without correctness filtering. These are diagnostic cases, not a new held-out benchmark.

## Run

Choose an A100 runtime and install requirements/models.txt. The model revision is pinned in the generation script. Record the repository commit and installed packages with each run.

```bash
python scripts/run_free_reasoning.py
python scripts/continue_free_reasoning.py
python scripts/run_free_reasoning_patching.py
python scripts/trace_free_patching.py --run runs/free-reasoning-patching
```

Generation compares a 512-position full canvas with appended blocks of 32. Both commit one highest-confidence token per forward. Continuation preserves original tokens and writes, extending only truncated block responses to 1024 tokens. It replaces the original trajectory in subsequent analysis and is not an independent sample. Generation has a 20-minute guard. Scripts refuse to overwrite existing outputs; preserve artifacts before restarting.

## Source and Target

For shared tasks, A=(x+y)+(u+r), B=2(x+y)+(v+s). A changed-sum Source uses x+1, u-1, v-2, changing the sum while keeping final A/B fixed. A control Source uses u+2, v+3, keeping the sum fixed while changing final answers. Sources independently generate their responses.

The locator requires an explicit first x+y result, a `For B:` boundary, and a numerical x+y reference or the multiplier operand in `2*C`. Inspect extracted spans manually. Unsupported Source phrasing is logged as skipped; unsupported Target phrasing aborts. Numerals must align exactly to token boundaries; the contrast requires equal token widths and a changed final digit. This is a narrow diagnostic locator, not a general reasoning parser.

Replay includes only tokens actually committed before the first B-reference write, excluding future text. Completed text is used only to locate spans. Source and Target use their own semantic events; absolute positions, wording, lengths and visible contexts can differ.

Scan residual module outputs at layers 1, 4, 8, 12, 16, 20, 24, 28, 31. Replace Target's earlier common-sum positions with Source states. Controls include self-patching, same-sum Source, and the same vector at unrelated text positions.

L = ln P(Source final digit) - ln P(Target final digit), using full-vocabulary probabilities; delta = L_patch - L_baseline. Select the layer with largest local delta within each case. This is exploratory selection on evaluation cases, not held-out evidence. Continue with one-forward or current-block patching, alongside unmodified replay and direct numeric-token replacement. Token replacement can make preceding calculations inconsistent.

## Decode order and interpretation

A high query probability does not imply commitment: another position may have higher confidence. Patching can alter predictions and the subsequent schedule. The trace audit re-evaluates saved pre-write states and asserts agreement with selected positions and tokens. Report local prediction, actual commitment, final answer and calculation consistency separately.

A future fixed-position-schedule comparison could separate some scheduling effects from state effects. It changes the decoding procedure and should accompany native scheduling. This comparison is not implemented. Text reuse or a local residual effect does not establish computation only once, an abstract sharing module, or DLM-specific mechanisms.

## Outputs

Outputs stay under ignored `runs/free-reasoning/` and `runs/free-reasoning-patching/`. Manifests record models and scope; JSONL stores prompts, responses, states and commitments. The notebook contains no embedded results and never uploads them. Download artifacts before releasing Colab.

`DLM_PATCH_RUN_NAME` chooses another patch directory name under runs; `DLM_PATCH_REUSE` reuses Sources from a prior run under runs. Verify matching inputs and model revision before reuse. Pass the chosen directory to the trace command.
