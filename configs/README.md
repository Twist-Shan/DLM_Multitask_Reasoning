# Configuration scope

| File | Intended use |
|---|---|
| `models/mock.json` | Offline legacy CLI smoke/backend checks |
| `models/illada.json`, `models/dream.json` | Model adapter configuration templates for the legacy CLI |
| `smoke.json` | Small mock interface check |
| `colab_preflight.json` | Legacy shared single-digit-domain model preflight |
| `screen.json`, `pilot.json` | Earlier fixed-slot design presets; not the current mechanism pipeline |

The current component/head-bank runners specify their model revision and discovery/validation setup in their scripts. Editing these JSON presets does not change those runners. Fixed single-token answer-slot assumptions in the legacy workflow must pass the common tokenizer audit before real model execution. Do not silently drop incompatible cases for only one model.

Resolved local model configs belong under `configs/local/`, which is ignored by Git. Record effective settings from the run manifest rather than assuming a template alone captures the experiment.
