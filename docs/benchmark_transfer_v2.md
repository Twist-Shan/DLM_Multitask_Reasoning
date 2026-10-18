# Benchmark transfer v2: completion and joint-answer calibration

Exploratory continuation of the [v1 benchmark adaptation](benchmark_transfer.md).
Use the existing seed123 candidate dataset; preserve all v1 outputs. No new
confirmation claims, activation patching, or DLM-versus-AR claims.

## Fixed before inference

- Model: pinned iLLaDA checkpoint and native chat template, BF16, A100.
- Cases: first three per benchmark × actual hop count {2,3,4}, 18 total.
  The ordering interleaves benchmark and depth; no correctness filtering.
- Main conditions: B, AB, BA, RB, all full context, 72 generations.
  Questions, paragraphs and R controls remain those of v1. There is no oracle
  paragraph selection in this run. These remain adapted, not official scores.
- Prompt allows brief reasoning and asks for `FINAL:` followed by labeled
  short answers. It does not request hidden intermediate variables or inject
  ground-truth answers. Native `<think>` output is allowed.
- Main decoding: append 32-token blocks on demand; commit up to four highest
  confidence tokens per forward, completing each block before EOS stopping.
  Maximum output budget1024. This is a semi-autoregressive DLM baseline.
- Full-canvas control: same budget1024 and four commits/forward on each
  benchmark's first 2-hop case, B and AB (four additional generations).
  Same questions and prompts in both decoders. Full-canvas generation fills
  its entire canvas before output trimming; blockwise generation stops at a
  finished block containing EOS. Report actual forwards and processed token
  positions; equal token budgets do not imply equal computation.
- Time limit3600 seconds, external process watchdog3660 seconds. Preserve
  partial runs and report completeness; do not silently replace failed cases.

## Scoring

Freeze extraction before v2: if `<think>` appears, require exactly one properly
closed leading block and only score what follows. Accept one line containing
`FINAL:` (discard preceding reasoning), or directly labeled answers after a
closed reasoning block, or directly labeled answers without reasoning tags.
Reject duplicate final markers, missing labels, duplicate labels, wrong order,
and extra final-answer lines. Never search unfinished reasoning for an answer.
MuSiQue EM/F1 uses the annotated answer aliases; StepGame uses exact canonical
direction labels. Missing/invalid final answers count as failures in overall
accuracy. Report format/extraction success, EOS completion and conditional
accuracy separately. Retain raw full-response compliance as a diagnostic.

Report B results by benchmark and hops, paired AB−B / BA−B / RB−B at the
question level, and the small decoder comparison separately. Avoid significance
claims for nine questions per benchmark. A=B spatial cases, incomplete aliases,
template-parser selection, extra R paragraphs, and pretraining contamination
remain limitations. Answer quality improvements do not establish precommit
information sharing. Blockwise vs full changes future masked space and decoding
constraints together; it is not an isolated attention intervention.

## Run

```shell
python scripts/run_benchmark_transfer_v2.py --dataset runs/transfer-prepared/dataset.json --out runs/transfer-v2 --prepare-only
python scripts/run_benchmark_transfer_v2.py --dataset runs/transfer-prepared/dataset.json --out runs/transfer-v2-illada --max-seconds 3600
python scripts/analyze_benchmark_transfer_v2.py --run runs/transfer-v2-illada --out reports/transfer-v2
```

Use a new output directory for each invocation. Source/Target causal adaptation
remains a later stage after completion, extraction and basic accuracy are usable.
