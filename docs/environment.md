# Environment setup

## CPU validation

Python 3.10+ is supported by package metadata; GitHub CI uses Python 3.12.

```sh
python -m pip install -e .
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m unittest discover -s tests -v
python -m dlm_coupling smoke --out runs/smoke
```

The core package has no required external dependencies. Without PyTorch, three tensor-level tests are skipped. The CPU suite does not download model weights. Install `requirements/analysis.txt` only if you need figure generation.

## CUDA inference

Choose the PyTorch build appropriate for your CUDA environment first, then:

```sh
python -m pip install -e .
python -m pip install -r requirements/models.txt -r requirements/analysis.txt
python -m pip check
```

The current fixed-slot mechanism runners request CUDA, BF16, batch size 1, and iLLaDA. Their supplied resource target is an A100 40GB. This does not establish memory requirements for other settings or models. CPU CI does not validate a CUDA runtime.

| Adapter | Model | Pinned revision used by the experiment runners |
|---|---|---|
| iLLaDA | GSAI-ML/iLLaDA-8B-Instruct | `5769f049224b2cff61106ebce0429ede5deeb434` |
| Dream | Dream-org/Dream-v0-Instruct-7B | `05334cb9faaf763692dcf9d8737c642be2b2a6ae` |

The adapter uses the model's native tokenizer and chat template. iLLaDA's separate `chat_template.jinja` is loaded from the same revision. Official remote model code is enabled with `trust_remote_code`; weights and downloaded source are not distributed in this repository.

`requirements/models.txt` pins the Transformers/tokenizer/Hub compatibility combination. Other dependencies are ranges or unpinned; this is not a complete environment lock. Save `pip freeze`, Python and CUDA versions, repository commit, model revisions, and run manifests with your local run.

## Colab

Use `notebooks/Run_Mechanism_Headbank.ipynb` and select a GPU runtime before executing it. Set `REVISION` to an immutable repository commit for reproducibility. It creates `/content/DLM_Multitask_Reasoning` and refuses to reuse that directory. Colab storage is temporary: preserve outputs yourself before releasing the runtime. Nothing is uploaded automatically.
