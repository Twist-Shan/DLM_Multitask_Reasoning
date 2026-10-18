# Benchmark transfer: exploratory protocol v1

This extension tests whether dependency-sensitive joint answering transfers beyond arithmetic. It does not establish precommit causal exchange or a DLM advantage. Use seed **123**, never date-derived seeds. Keep downloaded datasets, prepared cases, execution notebooks and results private.

## Sources and prior experimental practice

| Benchmark | Construction and original evaluation | What we borrow |
|---|---|---|
| [MuSiQue](https://aclanthology.org/2022.tacl-1.31/) | Compose connected single-hop questions; retain decompositions, supporting passages and answer aliases; 2–4 hops. Compare end-to-end, support selection and decomposed reasoning. | Original composed question and passages; annotated dependency graph and upstream question. |
| [StepGame](https://github.com/ShiZhengyan/StepGame) | Synthetic natural-language spatial relations, varied reasoning depth and distractors. | Original stories, original query, a symbolic coordinate checker. It is not a naturally occurring text corpus. |
| [StepGame audit](https://arxiv.org/abs/2401.03991) | Identify faulty templates; compare direct prompting, CoT/ToT, and semantic parsing plus symbolic reasoning. | Read the published template mapping as literal data; exclude unresolved templates and disagreements, never silently relabel. |
| [CLUTRR](https://github.com/facebookresearch/clutrr) | Family-relation graphs with paraphrases and systematic generalization. | Candidate future extension for relation composition; not yet implemented. |
| [ProofWriter](https://aclanthology.org/2021.findings-acl.317/) | Natural-language facts and rules with proof supervision. | Candidate future branch/merge tasks; unknown must remain distinct from false. Not yet implemented. |
| [GSM-Infinite](https://arxiv.org/abs/2502.05252) | Computational graphs rendered as word problems, varying reasoning complexity and distractors. | Future bridge from arithmetic DAGs to word problems; not yet implemented. |

## Data and audit

- MuSiQue answerable **development** split, not a confirmation set. Official download: author repository Google Drive archive. If quota blocks it, record the exact alternative mirror/revision and SHA256; do not assert byte equality with the inaccessible official archive.
- StepGame author HF validation revision `6d859381dfd518cae3f073b268aaa323bf4dcf04`. Each sentence must have one unambiguous semantic mapping. Solve integer coordinates and compare the result to the published label. Retain published labels and source rows. Template-parser coverage is a selection limitation, and a disagreement is a candidate annotation/parser issue, not automatically a proven dataset error.
- Store `k_hop` separately from the measured shortest query path. The author card describes k both as a maximum and as the required hops; the observed fields are not interchangeable. Stratify this adaptation by the measured path length.
- Select ten cases per benchmark × {2,3,4} hops with seed123 before any model outcomes. Initial smoke pilot takes the first three from each cell, for 18 cases and 108 generation conditions. Do not substitute failures with easier questions.
- Public benchmark questions may have appeared in pretraining; contamination is not ruled out. Correct answers alone do not establish passage-grounded multihop reasoning. Future coherent counterfactual passages and unrelated-information controls are required.

## Questions and controls

- **B**: original composed question, verbatim.
- **A**: MuSiQue root subquestion on an annotated path to B. Relation shorthand is mechanically rendered as a question; reference answers are never substituted. StepGame asks about one original edge on the query path. An arbitrary multi-edge direction label would lose displacement magnitude, so it is not used as a sufficient intermediate variable.
- **R**: MuSiQue root question from another case with its supporting paragraph; reject matching titles or simple answer overlap. This is an approximate unrelated control, not a proven semantically independent world. StepGame adds a disconnected pair of unused entities; its direction varies.
- Compare B, AB, BA, RB. All four receive identical context, including the R fact/paragraph. AB vs BA changes requested response order; confidence decoding may commit tokens in a different order. These comparisons jointly change requested outputs and scaffolding; they are not isolated physical-slot interventions.
- MuSiQue full context retains all original paragraphs and appends the R paragraph. A separate oracle-support condition keeps gold supporting paragraphs plus R. Neither is an official benchmark score: both add R, and the latter supplies oracle retrieval. Joint prompting additionally provides structural supervision through A.

## Generation and measurement

Use the existing pinned model adapter and native chat template. Generate a full 128-token masked canvas, greedily committing four highest-confidence tokens per forward, with no gold-answer length, candidate choices or forced gold tokens. No silent context truncation: exclude all four paired conditions together if any exceeds context length, and report exclusions. This pilot uses iLLaDA; Dream is supported by the runner but not assumed to behave identically.

Require one `LABEL: answer` line per requested question in order. Report format compliance separately. Primary B score counts invalid responses as zero. MuSiQue uses SQuAD-style normalized exact match and token F1 maximized over annotated aliases. StepGame uses exact canonical direction labels. MuSiQue A aliases are incomplete, so A EM is diagnostic only. Free-generation outputs are not evidence about t=0 or masked-region information exchange.

Report paired B differences AB−B, BA−B and RB−B at the question level, together with raw counts and examples. Three cases per hop is a smoke test; no significance or mechanism claim. Preserve failures, actual output strings, commit traces, model revision, input hashes and context lengths. A later causal experiment needs valid Source/Target passage edits, unchanged irrelevant facts, and explicit tests that the counterfactual answer is well defined.

## Commands

Place private source files under `data/external/realistic-v1/`: `musique-dev.jsonl`, `stepgame-valid.jsonl`, and `stepgame-semantic-mapping.py` from the cited audit repository. Record provenance sidecars and hashes. The mapping is parsed with AST literal evaluation, never executed.

```shell
python scripts/fetch_benchmark_inputs.py
python scripts/run_benchmark_transfer.py --prepare --out runs/transfer-prepared --per-hop 10 --seed 123
python scripts/run_benchmark_transfer.py --dataset runs/transfer-prepared/dataset.json --out runs/transfer-tokenizer --audit-only
python scripts/run_benchmark_transfer.py --dataset runs/transfer-prepared/dataset.json --out runs/transfer-illada --cases-per-cell 3 --max-seconds 1800
python -m unittest discover -s tests -p test_benchmark_transfer.py
```

If the short canvas is consumed by unfinished reasoning, preserve the original run and diagnose completion separately. For example, keep the first two-hop case from each benchmark and compare B/AB with a larger canvas while retaining four commits per forward:

```shell
python scripts/run_benchmark_transfer.py --dataset runs/transfer-prepared/dataset.json --out runs/transfer-long-diagnostic --cases-per-cell 1 --hops 2 --contexts full --requests B AB --max-new-tokens 1024 --max-seconds 1200
python scripts/analyze_benchmark_transfer.py --run runs/transfer-long-diagnostic --out reports/transfer-long-diagnostic.html
```

The analyzer preserves strict whole-response scores and separately labels a post-hoc diagnostic that scores text after exactly one completed `<think>...</think>` block. It never extracts an answer from unfinished reasoning. Completion diagnostics must not be reported as confirmation of a joint-answer effect. Larger canvases also change the masked workspace, so their effect is not solely extra elapsed reasoning time.

The runner refuses to overwrite an output directory. Model weights load only after tokenizer/context auditing. A separate external process timeout is recommended to bound model download or a stalled forward.

## Follow-up causal design (not implemented by this calibration)

For StepGame, change a single upstream edge in Source, recompute coordinates, and check whether B's direction actually changes; changing an edge need not cross a direction boundary. Keep Target's other edges fixed. Include same-B edge changes, a disconnected R edge, and self-patching. Separately report A=B cases to address copying. An independent-condition effect must be evaluated at a matched counterfactual direction, not an arbitrary nearby label.

For MuSiQue, ordinary unrelated examples are not automatically valid Source/Target pairs. A passage edit must retain a unique answer and a coherent bridge entity across relevant passages. Entity aliases, duplicate mentions, alternative supporting passages and token alignment require explicit auditing. Start with manually verified pairs; do not treat an upstream question's presence as evidence that its masked answer region mediates B. Full-context and oracle-support results should determine whether retrieval or reasoning warrants the first intervention.

Before causal transfer, freeze eligible conditions and a separate confirmation set. Investigate output positions and commitment trajectories, then test Source/Target patching and attention interventions. Comparing AB to B alone measures the effect of the whole prompt change. Isolating a masked-region mechanism still requires the request × canvas controls used in the toy experiments.
