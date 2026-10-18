# Small answer graphs

This exploratory workflow extends the answer-only interface to five named variables. It calibrates model behavior before causal patching. It does not contain historical model outputs.

| Graph | Canonical structure |
|---|---|
| Independent | A, B, C, D, E have no parents |
| Chain | A → B → C → D; E isolated |
| Fork | A → B, A → C, A → D; E isolated |
| Merge | A, B → C → D; E isolated |
| Diamond | A → B, C; B, C → D; E isolated |

Variable names and definition order are permuted independently. The letters above describe structural roles, not fixed printed positions. The same label permutation is used across task families for a given graph/index.

Four families share this graph interface:

- **Addition:** four one-digit operands at a root; parents plus two local operands elsewhere.
- **Subtraction:** two-digit subtraction at roots; a local sum minus all parents elsewhere. Effects can reverse sign along an edge.
- **Boolean:** local XOR combined with AND, OR or XOR of parents. Saturation and reconvergent cancellation are recorded explicitly.
- **Symbolic lookup:** all unary or ordered binary tables are given in the prompt, with answers W/X/Y/Z. No external knowledge is required.

Each node has `change`, `same_value`, and `self` Source constructors. Changing a local definition leaves other node functions unchanged. An exact evaluator verifies equivalence to intervening on that node's value in Target. Structural descendants and actually changed answers are separate fields. Lookup tables contain an inactive duplicate output so that `same_value` can match the exact lexical edit made by `change`; this constrains the sampling distribution.

## Run

Use Python 3.10+, install the package, and prepare without model downloads:

```sh
python -m pip install -e .
python -m unittest discover -s tests -p test_answer_graphs.py -v
python scripts/run_answer_graphs.py --prepare-only --seed 123 --per-cell 10 --out runs/graphs-prepared
```

With CUDA PyTorch and `requirements/precommit.txt` installed:

```sh
python scripts/run_answer_graphs.py --model illada --seed 123 --per-cell 10 --out runs/graphs-illada
python scripts/analyze_answer_graphs.py runs/graphs-illada --out runs/graphs-illada/analysis.json
```

The default is four families × five graphs × ten instances = 200 cases. Each case has five single-output calls plus two joint layouts crossed with two commitment rules, yielding 1,800 output records. Output directories must be new. `--audit-only` checks the real tokenizer without loading weights; `--families` and `--graphs` allow an explicitly smaller calibration. `--model dream` is also supported; passing software checks does not establish model capability.

The [Colab notebook](../notebooks/Run_Answer_Graphs.ipynb) uses a bounded A100 run and exports local results. Set an exact Git revision to reproduce a code snapshot.

## Measurement contract

All five definitions appear even in single-output prompts. Joint requests use a topological physical layout and its reverse. Each layout is crossed with confidence-based commitment and strictly left-to-right token commitment. Full-vocabulary greedy predictions are used, excluding only MASK; invalid outputs count as errors. Native chat endings are retained, with no extra unused positions. Token spans may include adjacent whitespace, but may never consume a label or delimiter; all legal values and all Source/Target positions are audited before weights load.

The first forward has all answers masked. Identical first-forward predictions are reused between commitment policies, with actual and logical forward counts recorded separately. The analyzer replays every saved commit, checks exact answers/prompts, the complete condition grid, and forward accounting. It reports t=0/final field accuracy, whole-response correctness, invalid fields, and paired layout/joint differences. Bootstrap intervals resample graph instances, not answer fields (4,000 draws, seed 123).

This runner performs **calibration only**, without activation patching or correctness filtering. Source inputs are audited, not evaluated by the model. Joint versus single changes prompt wording, length, number of slots and total forwards. Graphs and task families are not matched for difficulty, expression length or answer distribution. Behavioral differences therefore do not establish internal information flow. Follow-up patching requires fixed conditions and relevant/irrelevant/self controls on a separate declared evaluation set.
