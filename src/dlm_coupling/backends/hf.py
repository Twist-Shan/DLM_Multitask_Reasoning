"""Pinned Hugging Face adapters for fixed-slot iLLaDA and Dream forwards.

The tokenizer loads at construction. Weights and torch load only on the first
predict call; tokenization audits never download model weights.
"""

import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import re
from typing import Any

from .base import PreparedCanvas, build_canvas, raw_logit_positions, validate_prediction_request


MODEL_FAMILIES = {
    "GSAI-ML/iLLaDA-8B-Instruct": "illada",
    "Dream-org/Dream-v0-Instruct-7B": "dream",
}
MASK_IDS = {"illada": 5, "dream": 151666}


class HFBackend:
    def __init__(self, config: dict[str, Any], *, load_model: bool = False):
        self.config = dict(config)
        self.model_id = str(self.config.get("model_id", ""))
        self.family = self.config.get("family", MODEL_FAMILIES.get(self.model_id))
        if self.family not in MASK_IDS:
            raise ValueError("Set family to 'illada' or 'dream' with the matching model checkpoint.")
        known_family = MODEL_FAMILIES.get(self.model_id)
        if known_family is not None and self.family != known_family:
            raise ValueError("The configured family does not match the official checkpoint.")
        self.revision = str(self.config.get("revision", ""))
        if not re.fullmatch(r"[0-9a-fA-F]{40}", self.revision):
            raise ValueError("HF experiments require an immutable 40-character model commit SHA in revision.")
        if self.config.get("trust_remote_code") is not True:
            raise ValueError("These official custom checkpoints require explicit trust_remote_code=true.")
        if self.config.get("dtype", "bfloat16") not in {"bfloat16", "float16", "float32"}:
            raise ValueError("dtype must be bfloat16, float16, or float32.")
        self.mask_id = MASK_IDS[self.family]
        self._model = None
        self._torch = None
        self._cuda_memory_device = None
        try:
            from transformers import AutoConfig, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install the model inference dependencies before using the HF backend.") from exc
        self._load_kwargs = {
            "revision": self.revision,
            "code_revision": self.revision,
            "trust_remote_code": True,
            "local_files_only": bool(self.config.get("local_files_only", False)),
        }
        if self.config.get("cache_dir"):
            self._load_kwargs["cache_dir"] = self.config["cache_dir"]
        self.model_config = AutoConfig.from_pretrained(self.model_id, **self._load_kwargs)
        loaded_revision = getattr(self.model_config, "_commit_hash", None)
        if loaded_revision and loaded_revision.lower() != self.revision.lower():
            raise RuntimeError("Loaded model config does not match the requested immutable revision.")
        # Do not silently fetch custom code from a second, unpinned repository.
        # AutoConfig qualifies even same-repository references with repo_id--.
        # Those still use our pinned code_revision; reject only a different repo.
        if any("--" in str(value) and str(value).split("--", 1)[0] != self.model_id
               for value in getattr(self.model_config, "auto_map", {}).values()):
            raise ValueError("External auto_map code repositories need a separate pinned adapter.")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, **self._load_kwargs)
        configured_mask = getattr(self.model_config, "mask_token_id", None)
        tokenizer_mask = getattr(self.tokenizer, "mask_token_id", None)
        for source, value in (("model config", configured_mask), ("tokenizer", tokenizer_mask)):
            if value is not None and value != self.mask_id:
                raise ValueError(f"Unexpected {source} mask ID {value}; expected {self.mask_id}.")
        template_source = "tokenizer"
        if not getattr(self.tokenizer, "chat_template", None):
            # Older Transformers does not load standalone chat_template.jinja.
            from huggingface_hub import hf_hub_download
            template_path = hf_hub_download(
                self.model_id, "chat_template.jinja", revision=self.revision,
                cache_dir=self.config.get("cache_dir"),
                local_files_only=bool(self.config.get("local_files_only", False)),
            )
            self.tokenizer.chat_template = Path(template_path).read_text(encoding="utf-8")
            template_source = "same pinned repository: chat_template.jinja"
        if not getattr(self.tokenizer, "chat_template", None):
            raise ValueError("The instruct checkpoint must provide its own chat template.")
        configuration = self.model_config.to_dict()
        self.metadata = {
            "backend": "hf",
            "family": self.family,
            "model_id": self.model_id,
            "revision": self.revision,
            "resolved_revision": loaded_revision or self.revision,
            "code_revision": self.revision,
            "tokenizer_revision": self.revision,
            "is_mock": False,
            "weights_loaded": False,
            "mask_id": self.mask_id,
            "dtype_requested": self.config.get("dtype", "bfloat16"),
            "device_requested": self.config.get("device", "cuda"),
            "cuda_max_memory_allocated_bytes": None,
            "cuda_max_memory_reserved_bytes": None,
            "cuda_memory_device": None,
            "cuda_memory_scope": "Not measured; CUDA weights have not been loaded",
            "attention": "full bidirectional, batch size 1, no padding",
            "logit_alignment": "position j uses raw j-1 (j=0 uses 0)" if self.family == "dream" else "same position",
            "use_cache": False,
            "cfg": 0.0,
            "temperature": 0.0,
            "model_config": configuration,
            "model_config_sha256": hashlib.sha256(json.dumps(configuration, sort_keys=True, default=str).encode()).hexdigest(),
            "chat_template_sha256": hashlib.sha256(str(self.tokenizer.chat_template).encode()).hexdigest(),
            "chat_template_source": template_source,
            "tokenizer_class": type(self.tokenizer).__name__,
            "versions": self._versions(),
        }
        if load_model:
            self._load_model()

    @staticmethod
    def _versions() -> dict[str, str]:
        versions = {}
        for package in ("torch", "transformers", "tokenizers", "huggingface-hub", "safetensors"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = "not installed"
        return versions

    def prepare(self, prompt: str, slot_keys: list[str]) -> PreparedCanvas:
        canvas = build_canvas(self.tokenizer, prompt, slot_keys, self.mask_id)
        limits = [
            getattr(self.model_config, name, None)
            for name in ("max_position_embeddings", "max_sequence_length", "max_seq_len")
        ]
        limits = [value for value in limits if isinstance(value, int) and value > 0]
        limit = min(limits) if limits else None
        if limit is not None and len(canvas.token_ids) > limit:
            raise ValueError(f"Canvas length {len(canvas.token_ids)} exceeds model context limit {limit}.")
        return canvas

    def _load_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel

        device = torch.device(str(self.config.get("device", "cuda")))
        is_cuda = device.type == "cuda"
        if is_cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Tokenizer audit works without a GPU; inference needs a configured device.")
        # Resolve a bare 'cuda' once so measurements remain tied to this GPU
        # even if another caller subsequently changes the current device.
        if is_cuda and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        dtype = getattr(torch, self.config.get("dtype", "bfloat16"))
        if is_cuda and dtype == torch.bfloat16:
            # This API checks the current device, not the configured device.
            with torch.cuda.device(device):
                if not torch.cuda.is_bf16_supported():
                    raise RuntimeError("This CUDA device does not support bfloat16; choose a supported GPU or explicit dtype.")
        if is_cuda:
            torch.cuda.reset_peak_memory_stats(device)
            self._cuda_memory_device = device
            self.metadata.update({
                "cuda_memory_device": str(device),
                "cuda_memory_scope": (
                    "PyTorch allocator peaks on the selected device in this process, "
                    "from the reset before model loading through completed predictions; "
                    "includes model loading, warm-up and any pre-existing allocations, "
                    "but excludes non-PyTorch allocations"
                ),
            })
        else:
            self.metadata["cuda_memory_scope"] = "Not applicable; model device is not CUDA"
        model = AutoModel.from_pretrained(
            self.model_id,
            config=self.model_config,
            torch_dtype=dtype,
            **self._load_kwargs,
        ).to(device).eval()
        model.config.use_cache = False
        self._model = model
        self._torch = torch
        self.metadata.update({
            "weights_loaded": True,
            "device": str(model.device),
            "dtype": str(model.dtype),
            "model_class": type(model).__name__,
            "model_module": type(model).__module__,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        })
        if is_cuda:
            self.metadata["cuda_version"] = torch.version.cuda
            self.metadata["gpu_name"] = torch.cuda.get_device_name(device)
            self._record_cuda_peak_memory()
        # Hash the actual imported model source, in addition to the Hub commit.
        source_path = inspect.getsourcefile(type(model))
        if source_path and Path(source_path).is_file():
            self.metadata["model_source_sha256"] = hashlib.sha256(Path(source_path).read_bytes()).hexdigest()

    def _record_cuda_peak_memory(self) -> None:
        # A CPU fixture may construct the adapter with __new__, without any
        # metadata. It must neither require metadata nor touch the CUDA API.
        device = getattr(self, "_cuda_memory_device", None)
        if device is None:
            return
        self.metadata.update({
            "cuda_max_memory_allocated_bytes": int(self._torch.cuda.max_memory_allocated(device)),
            "cuda_max_memory_reserved_bytes": int(self._torch.cuda.max_memory_reserved(device)),
        })

    def decode_token(self, token_id: int) -> str:
        return self.tokenizer.decode(
            [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

    def predict(
        self,
        token_ids: list[int],
        positions: list[int],
        value_token_ids: list[list[int]],
    ) -> list[dict[str, Any]]:
        validate_prediction_request(token_ids, positions, value_token_ids)
        if not positions:
            return []
        self._load_model()
        torch, model = self._torch, self._model
        inputs = torch.tensor([token_ids], dtype=torch.long, device=model.device)
        kwargs = {"input_ids": inputs, "use_cache": False, "return_dict": True}
        if self.family == "dream":
            # Matches Dream's native no-padding forward. No causal mask.
            kwargs["attention_mask"] = "full"
        with torch.inference_mode():
            raw = model(**kwargs).logits
            if raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[1] != len(token_ids):
                raise RuntimeError("Expected full [1, sequence_length, vocabulary] raw model logits.")
            rows = raw[0, raw_logit_positions(positions, self.family), :].float()
            if not bool(torch.isfinite(rows).all()):
                raise RuntimeError("Model returned non-finite logits.")
            probabilities = torch.softmax(rows, dim=-1)
            raw_tops = rows.argmax(dim=-1)
            candidates = rows.clone()
            candidates[:, self.mask_id] = -torch.inf
            selections = candidates.argmax(dim=-1).tolist()
            raw_tops = raw_tops.tolist()
            result = []
            for index, probes in enumerate(value_token_ids):
                if any(token < 0 or token >= rows.shape[-1] for token in probes):
                    raise ValueError("A probe token lies outside the model vocabulary.")
                selected = int(selections[index])
                raw_top = int(raw_tops[index])
                # Transfer the small probe vector once, not one GPU scalar per
                # value (which would impose repeated synchronization overhead).
                requested = list(dict.fromkeys([selected, *probes]))
                values = probabilities[index, requested].tolist()
                recorded = dict(zip(requested, values))
                result.append({
                    "token_id": selected,
                    "token_text": self.decode_token(selected),
                    # Preserve original full-vocabulary probability; never
                    # renormalize over numeric values or after MASK exclusion.
                    "confidence": float(recorded[selected]),
                    "raw_top_id": raw_top,
                    "raw_top_text": self.decode_token(raw_top),
                    "probabilities": {str(token): float(recorded[token]) for token in probes},
                })
        self._record_cuda_peak_memory()
        return result
