# Software validation

## Checks

```sh
python -m unittest discover -s tests -v
python -m dlm_coupling smoke --out runs/smoke
```

The suite includes the earlier harness and versioned precommit tests. Synthetic-tensor hook tests require PyTorch and are skipped when it is absent. The GitHub Actions workflow installs CPU PyTorch and reruns the complete suite on pushes; test success does not establish pretrained-model performance.

Coverage includes arithmetic graph evaluation and relations, parsers, data splitting, tokenizer/logit alignment, full-vocabulary predictions, greedy token submission, exact continuation, fixed-slot validation, legacy CLI locking/recovery, and synthetic-tensor checks of intervention hooks and cleanup.

Precommit checks cover native suffix placement, two-digit answer maps, Source compensation, split-family collision auditing, frozen layer schedules, per-layer Source capture, resample derangements, and paired statistical units. Carry-audit checks include decimal boundaries/borrowing, position-map alignment, absence of an erroneous division by two, group-level aggregation, and numerical zero tolerance. They use synthetic inputs and do not bundle empirical records.

The smoke workflow uses a mock backend and generated synthetic inputs. It checks that task construction, execution, trace storage, and summarization work together. It is not a model-accuracy experiment.

## Boundaries

No pretrained-model inference, GPU compatibility, download access, performance measurement, or full statistical analysis is performed by CPU CI. Plot builders require locally produced measurements; they are not validated against published model results in this code-only repository. External iGSM export requires its separate upstream checkout and dependencies.

The public Colab notebook is a code launcher with empty outputs, not a frozen record of a completed run. Pin its repository revision and save your environment and outputs locally when conducting experiments.
