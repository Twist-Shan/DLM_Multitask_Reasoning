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

## Additional diagnostic (post-hoc)

The analyzer offers a separately labeled diagnostic for explicit final answers
independently of think-tag validity. It requires EOS and exactly one standalone
`FINAL:` marker, then applies the
unchanged label and answer scorer to its suffix. It does not require valid
think tags, and does not score unfinished outputs or search reasoning for an
answer. This is post-hoc format diagnosis, not a replacement primary endpoint.

## Author-decoder follow-up (exploratory, added after v2 began)

The author's [evaluation configuration](https://github.com/ML-GSAI/LLaDA/blob/9182493720ed723ef8031210d85959364e51cbe0/eval_illada.sh)
uses variable-length blocks and end-of-thinking guidance for long reasoning.
A four-condition diagnostic uses the first 2-hop question of each benchmark,
B and AB, unchanged v2 prompts and a 1024-token cap. The pinned author's
`var_generate` uses 32 steps per 32-token block, end-think logit boost 100,
power 3, greedy selection, and EOS stopping. The cap differs from the author's
2048/4096-token reasoning evaluations. Commit rate, guidance and numerical
implementation change together, so this is not a single-factor causal test.

Obtain `generate.py` from the pinned author revision above; the runner verifies
SHA256 before importing it. Upstream code and downloaded data are not vendored.

```shell
python scripts/run_benchmark_official_diagnostic.py --dataset runs/transfer-prepared/dataset.json --upstream /path/to/generate.py --out runs/transfer-author-diagnostic
python scripts/analyze_benchmark_transfer_v2.py --run runs/transfer-v2-illada --official-run runs/transfer-author-diagnostic --out reports/transfer-v2
```

Use an external 1800-second watchdog for this four-condition diagnostic and
retain partial outputs. It is not a confirmation set or an official benchmark
score. Do not treat successful termination under guided decoding as evidence
of autonomous planning or precommit information sharing.


## Concrete-label template diagnostic (v3)

`--prompt-version v3` on `scripts/run_benchmark_official_diagnostic.py` changes only the output-format instructions: use the actual requested labels and append a concrete ordered template. It preserves the v2 dataset selection (one two-hop case per benchmark, B and AB), author decoder, seed 123, 1024-token cap, and v2 strict scoring. The default remains v2 so old invocations retain their meaning. This is a paired prompt diagnostic, not a benchmark accuracy estimate.

Run with the same separately obtained, hash-verified author `generate.py`:

```sh
python scripts/run_benchmark_official_diagnostic.py --dataset prepared/dataset.json --upstream upstream/illada_generate.py --out runs/template_v3 --prompt-version v3
python scripts/analyze_benchmark_template.py --previous runs/author_v2 --current runs/template_v3 --out reports/template_comparison
```

The comparison audits dataset, exact prompts, re-scored outputs, model revision and decoder configuration before displaying paired outputs. Keep generated reports and raw records private.
