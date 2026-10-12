# Software validation

## Checks

```sh
python -m unittest discover -s tests -v
python -m dlm_coupling smoke --out runs/smoke
```

The suite contains 77 tests. Three require PyTorch; without it they are skipped. CPU-PyTorch CI passed all 77 tests on the initial public code revision. The GitHub Actions workflow reruns the suite for subsequent pushes.

Coverage includes arithmetic graph evaluation and relations, parsers, data splitting, tokenizer/logit alignment, full-vocabulary predictions, greedy token submission, exact continuation, fixed-slot validation, legacy CLI locking/recovery, and synthetic-tensor checks of intervention hooks and cleanup.

The smoke workflow uses a mock backend and generated synthetic inputs. It checks that task construction, execution, trace storage, and summarization work together. It is not a model-accuracy experiment.

## Boundaries

No pretrained-model inference, GPU compatibility, download access, performance measurement, or full statistical analysis is performed by CPU CI. Plot builders require locally produced measurements; they are not validated against published model results in this code-only repository. External iGSM export requires its separate upstream checkout and dependencies.

The public Colab notebook is a code launcher with empty outputs, not a frozen record of a completed run. Pin its repository revision and save your environment and outputs locally when conducting experiments.
