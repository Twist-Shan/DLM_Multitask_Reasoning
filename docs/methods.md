# Task and intervention contract

## Task families

Behavioral experiments represent arithmetic as a dependency graph. Independent answer nodes have disjoint dependencies; shared nodes have a common ancestor without an ancestor relation between the two answers; upstream–downstream pairs have one requested answer upstream of the other. Joint AB/BA runs change question order, not the underlying facts.

The controlled branch task is:

$$
C=x+y,\quad U=u+r,\quad V=v+s,\quad A=C+U,\quad B=2C+V.
$$

The implementation in `tasks/igsm_explicit.py` uses ordinary integers in explicit equations. The earlier `tasks/igsm.py` workflow uses its own modular-arithmetic task contract. Do not combine scores from these representations.

## Prompt and fixed canvas

An illustrative user message in the generated-intermediate branch workflow is:

```text
Compute C = 5 + 6; U = 6 + 8; V = 4 + 7; A = C + U; B = 2 * C + V. Fill the five integer values in the supplied answer template. No explanation.
```

The code applies the native chat template, appends a fixed answer canvas, tokenizes the complete sequence, checks exact numeric boundaries, and replaces digit positions with the model's mask ID:

```text
C = [M][M]; U = [M][M]; V = [M][M]; A = [M][M]; B = [M][M].
```

`[M]` is documentation notation for one mask token, not a literal string sent to the model. Each value occupies two token positions. The program first permits C/U/V positions to commit, then permits A/B positions. Within the permitted set it commits one highest-confidence token per forward, breaking ties by position. Predictions use the full vocabulary with mask excluded, not a digits-only vocabulary.

The current generated-intermediate workflow does not inject gold C/U/V. Earlier visible-intermediate controls do supply correct intermediate values; masked controls leave these slots masked while generating answers. `baseline_repair.py` implements example removal, answer-only canvases, generated intermediates, and natural short reasoning. Their prompts and execution schedules are different experimental treatments.

## Source and Target

**Source** is the matched input supplying activations; **Target** is the original input receiving them. Source variants change C, U, or V, or preserve the intermediate while changing operands. Current component/head-bank variants change one operand at a time. Earlier shared-state pilots use Source inputs with additional changed branch parameters; their expected output is a hybrid retaining Target's non-intervened parameters.

- Residual patching replaces the selected block-output vectors.
- Attention/MLP patching replaces the corresponding module output.
- Head patching replaces selected query-head slices of the input to `o_proj`.
- A head bank contains specified `(layer, head)` pairs, potentially spanning layers.

Paper-facing layer numbers start at 1; model arrays and query-head indices start at 0. iLLaDA has 32 query heads and 8 KV heads per layer. Query-head output slices remain distinct under grouped-query attention; a query-head intervention is not a KV-head intervention.

A single patch acts on the first answer forward only. A sustained patch acts on each answer forward. Source states are recomputed after copying Target's already-committed A/B tokens into Source's corresponding positions. Mask patterns must align. This creates a controlled hybrid input, which may conflict with Source's own arithmetic.

Every forward recomputes hidden states from the current tokens; hidden vectors are not carried continuously between denoising steps. An instantaneous argmax over all answer positions need not become the final answer because only one token is committed at that step.

## Metrics

For N base cases, pair accuracy is the fraction for which both final answers are correct. Invalid formats remain failures.

The expected branch counterfactuals are:

| Changed intermediate | Expected answer |
|---|---|
| C → C+δ | (A+δ, B+2δ) |
| U → U+δ | (A+δ, B) |
| V → V+δ | (A, B+δ) |

A newly induced counterfactual requires (1) both patched final answers equal the expected pair and (2) the unpatched pair did not already equal it. Count success over all assigned base cases; report the base-and-Source-correct subset separately. Non-target preservation compares the unaffected answer to its unpatched value, not automatically to gold.

For residual vectors at a layer and numeric position, relative L2 change is `||h_source - h_target||₂ / max(||h_target||₂, 1e-8)`. The figure builder averages the two digit positions of a variable, then averages cases. This is a descriptive distance, not a probe or a causal feature estimate.

Forward counts include Source capture calls. Multiple layers, conditions, output orders, and Source variants from one base case do not increase the number of independent cases.

## Controls and limits

The supplied workflows include self-state replacement, same-intermediate inputs, syntax-position controls where applicable, branch-specific changes, and matched-size random/complement banks. Discovery and validation inputs are generated separately. Full-layer/full-head answer-position replacement is an implementation positive control, not a localized mechanism discovery.

The current code tests interchange sufficiency. It does not by itself establish necessity, minimal circuits, autonomous planning, natural-language generalization, or a quantity being computed once and reused. Representation distances, prediction changes, and final counterfactual effects answer different questions.
