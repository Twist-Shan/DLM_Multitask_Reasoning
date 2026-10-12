import importlib.util
from contextlib import contextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

from dlm_coupling.backends import MockBackend, make_backend
from dlm_coupling.backends.base import build_canvas, output_template, raw_logit_positions
from dlm_coupling.backends.hf import HFBackend
from dlm_coupling.backends.mock import MockTokenizer


class BoundaryMergingTokenizer(MockTokenizer):
    def encode(self, text, add_special_tokens=False):
        # Value 7 merges with '='; it is not eligible for a fixed slot.
        text = text.replace("=7\n", "\u2603\n")
        return super().encode(text, add_special_tokens)


class BackendTests(unittest.TestCase):
    def _load_with_cuda_stub(self, requested_device):
        """Exercise load instrumentation without torch, downloads, or a GPU."""
        events = []

        class Device:
            def __init__(self, kind, index=None):
                pieces = str(kind).split(":")
                self.type = pieces[0]
                self.index = index if index is not None else (int(pieces[1]) if len(pieces) == 2 else None)

            def __str__(self):
                return self.type if self.index is None else f"{self.type}:{self.index}"

        class Cuda:
            current = 0
            allocated = 120
            reserved = 200

            def is_available(self):
                events.append(("available",))
                return True

            def current_device(self):
                events.append(("current", self.current))
                return self.current

            @contextmanager
            def device(self, target):
                previous = self.current
                self.current = target.index
                try:
                    yield
                finally:
                    self.current = previous

            def is_bf16_supported(self):
                events.append(("bf16", self.current))
                # The default GPU is deliberately incompatible; cuda:1 works.
                return self.current == 1

            def reset_peak_memory_stats(self, target):
                events.append(("reset", str(target)))

            def max_memory_allocated(self, target):
                events.append(("allocated", str(target)))
                return self.allocated

            def max_memory_reserved(self, target):
                events.append(("reserved", str(target)))
                return self.reserved

            def get_device_name(self, target):
                events.append(("name", str(target)))
                return "fixture GPU"

        class Model:
            config = SimpleNamespace(use_cache=True)
            dtype = "bfloat16"

            def to(self, device):
                events.append(("to", str(device)))
                self.device = device
                return self

            def eval(self):
                return self

            def parameters(self):
                return [SimpleNamespace(numel=lambda: 12)]

        def load(*args, **kwargs):
            events.append(("load",))
            return Model()

        cuda = Cuda()
        fake_torch = SimpleNamespace(device=Device, cuda=cuda, bfloat16="bfloat16", version=SimpleNamespace(cuda="fixture"))
        fake_transformers = SimpleNamespace(AutoModel=SimpleNamespace(from_pretrained=load))
        backend = HFBackend.__new__(HFBackend)
        backend._model = None
        backend._torch = None
        backend._cuda_memory_device = None
        backend.metadata = {}
        backend.config = {"device": requested_device, "dtype": "bfloat16"}
        backend.model_id = "fixture"
        backend.model_config = SimpleNamespace()
        backend._load_kwargs = {}
        with patch.dict("sys.modules", {"torch": fake_torch, "transformers": fake_transformers}):
            backend._load_model()
        return backend, cuda, events

    def test_cuda_peaks_include_loading_and_bf16_checks_selected_device(self):
        backend, cuda, events = self._load_with_cuda_stub("cuda:1")
        self.assertIn(("bf16", 1), events)
        self.assertEqual(cuda.current, 0)
        self.assertLess(events.index(("reset", "cuda:1")), events.index(("load",)))
        self.assertEqual(backend.metadata["cuda_memory_device"], "cuda:1")
        self.assertEqual(backend.metadata["cuda_max_memory_allocated_bytes"], 120)
        self.assertEqual(backend.metadata["cuda_max_memory_reserved_bytes"], 200)
        self.assertIn("before model loading", backend.metadata["cuda_memory_scope"])
        self.assertIn("pre-existing allocations", backend.metadata["cuda_memory_scope"])
        cuda.allocated, cuda.reserved = 300, 400
        backend._record_cuda_peak_memory()
        self.assertEqual(backend.metadata["cuda_max_memory_allocated_bytes"], 300)
        self.assertEqual(backend.metadata["cuda_max_memory_reserved_bytes"], 400)
        self.assertEqual(events.count(("reset", "cuda:1")), 1)
        self.assertEqual(events[-2:], [("allocated", "cuda:1"), ("reserved", "cuda:1")])

    def test_cpu_load_never_calls_cuda_memory_or_capability_api(self):
        backend, _, events = self._load_with_cuda_stub("cpu")
        self.assertEqual(events, [("load",), ("to", "cpu")])
        self.assertIsNone(backend._cuda_memory_device)
        self.assertIn("not CUDA", backend.metadata["cuda_memory_scope"])
        backend._record_cuda_peak_memory()
        self.assertEqual(events, [("load",), ("to", "cpu")])

    def test_scaffold_matches_declared_grammar_and_task_order(self):
        keys = ["B.intermediate", "B.answer", "A.intermediate", "A.answer"]
        self.assertEqual(output_template(keys, ["1", "2", "3", "4"]),
                         "B: intermediate=1; answer=2\nA: intermediate=3; answer=4\n")

    def test_canvas_is_gold_independent_and_records_each_slot(self):
        backend = MockBackend()
        canvas = backend.prepare("Compute both results.", ["A.answer", "B.answer"])
        self.assertEqual(len(canvas.slot_positions), 2)
        self.assertEqual(sum(token == backend.mask_id for token in canvas.token_ids), 2)
        self.assertFalse(canvas.metadata["uses_gold_to_prepare"])
        for key, position in canvas.slot_positions.items():
            self.assertEqual(set(canvas.value_token_ids[key]), {str(i) for i in range(100)})
            self.assertGreater(position, canvas.prompt_length)

    def test_context_merging_value_is_rejected(self):
        canvas = build_canvas(BoundaryMergingTokenizer(), "Compute.", ["A.answer"], 0)
        self.assertNotIn("7", canvas.value_token_ids["A.answer"])
        self.assertIn("8", canvas.value_token_ids["A.answer"])

    def test_prompt_mask_and_duplicate_slots_are_rejected(self):
        backend = MockBackend()
        with self.assertRaises(ValueError):
            backend.prepare("Injected <mask>", ["A.answer"])
        with self.assertRaises(ValueError):
            backend.prepare("Compute.", ["A.answer", "A.answer"])
        with self.assertRaises(ValueError):
            output_template(["A.answer", "B.answer", "A.intermediate"], ["1"] * 3)

    def test_dream_alignment_once_and_illada_no_shift(self):
        self.assertEqual(raw_logit_positions([0, 1, 5], "dream"), [0, 0, 4])
        self.assertEqual(raw_logit_positions([0, 1, 5], "illada"), [0, 1, 5])

    def test_mock_is_deterministic_and_never_receives_answers(self):
        backend = MockBackend({"seed": 17})
        canvas = backend.prepare("A and B.", ["A.answer", "B.answer"])
        positions = list(canvas.slot_positions.values())
        probes = [list(values.values()) for values in canvas.value_token_ids.values()]
        first = backend.predict(canvas.token_ids, positions, probes)
        self.assertEqual(first, backend.predict(canvas.token_ids, positions, probes))
        self.assertTrue(backend.metadata["is_mock"])
        self.assertFalse(backend.metadata["scientific_results"])
        self.assertTrue(all(row["token_id"] != backend.mask_id for row in first))
        changed = list(canvas.token_ids)
        changed[positions[1]] = canvas.value_token_ids["B.answer"]["3"]
        self.assertNotEqual(first, backend.predict(changed, positions, probes))
        self.assertEqual(backend.predict([], [], []), [])

    def test_bad_probe_shape_and_positions_are_rejected(self):
        backend = MockBackend()
        with self.assertRaises(ValueError):
            backend.predict([0], [0], [])
        with self.assertRaises(ValueError):
            backend.predict([0], [1], [[1]])
        with self.assertRaises(ValueError):
            backend.predict([0], [0], [[999]])

    def test_factory_and_revision_guards_need_no_gpu_dependencies(self):
        self.assertIsInstance(make_backend({"backend": "mock"}), MockBackend)
        with self.assertRaises(ValueError):
            make_backend({"family": "dream"})
        with self.assertRaises(ValueError):
            make_backend({"backend": "mock", "family": "dream"})
        with self.assertRaises(ValueError):
            HFBackend({"model_id": "Dream-org/Dream-v0-Instruct-7B", "revision": "main"})
        with self.assertRaises(ValueError):
            HFBackend({"model_id": "Dream-org/Dream-v0-Instruct-7B", "revision": "a" * 40})

    def test_hf_constructor_loads_only_pinned_config_and_tokenizer(self):
        calls = []
        model_config = SimpleNamespace(
            _commit_hash="a" * 40, mask_token_id=151666,
            auto_map={"AutoModel": "Dream-org/Dream-v0-Instruct-7B--modeling_dream.DreamModel"},
            to_dict=lambda: {"mask_token_id": 151666},
        )
        tokenizer = SimpleNamespace(mask_token_id=151666, chat_template="fixture")

        def config_loader(model_id, **kwargs):
            calls.append(("config", model_id, kwargs))
            return model_config

        def tokenizer_loader(model_id, **kwargs):
            calls.append(("tokenizer", model_id, kwargs))
            return tokenizer

        fake_transformers = SimpleNamespace(
            AutoConfig=SimpleNamespace(from_pretrained=config_loader),
            AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_loader),
        )
        with patch.dict("sys.modules", {"transformers": fake_transformers}):
            backend = HFBackend({"model_id": "Dream-org/Dream-v0-Instruct-7B",
                                 "revision": "a" * 40, "trust_remote_code": True})
        self.assertIsNone(backend._model)
        self.assertIsNone(backend._torch)
        self.assertFalse(backend.metadata["weights_loaded"])
        self.assertEqual([call[0] for call in calls], ["config", "tokenizer"])
        self.assertTrue(all(call[2]["revision"] == "a" * 40 for call in calls))
        self.assertTrue(all(call[2]["code_revision"] == "a" * 40 for call in calls))
        model_config.auto_map = {"AutoModel": "another/repository--modeling.OtherModel"}
        with patch.dict("sys.modules", {"transformers": fake_transformers}):
            with self.assertRaisesRegex(ValueError, "External auto_map"):
                HFBackend({"model_id": "Dream-org/Dream-v0-Instruct-7B",
                           "revision": "a" * 40, "trust_remote_code": True})
        model_config.auto_map = {}
        tokenizer.chat_template = None
        download = Mock(return_value="pinned-template.jinja")
        with patch.dict("sys.modules", {"transformers": fake_transformers,
                                        "huggingface_hub": SimpleNamespace(hf_hub_download=download)}):
            with patch("dlm_coupling.backends.hf.Path", return_value=SimpleNamespace(read_text=lambda **kwargs: "official template")):
                fallback = HFBackend({"model_id": "Dream-org/Dream-v0-Instruct-7B",
                                      "revision": "a" * 40, "trust_remote_code": True})
        self.assertEqual(download.call_args.kwargs["revision"], "a" * 40)
        self.assertEqual(fallback.tokenizer.chat_template, "official template")
        self.assertIn("same pinned repository", fallback.metadata["chat_template_source"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "optional torch is not installed")
    def test_hf_predict_aligns_and_preserves_unconstrained_probabilities(self):
        import torch

        raw = torch.zeros((1, 4, 6))
        raw[0, 1, 0] = 10.0  # Raw top is MASK, excluded only from commitment.
        raw[0, 1, 2] = 9.0
        raw[0, 2, 3] = 8.0
        calls = []

        class ModelFixture:
            device = torch.device("cpu")

            def __call__(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(logits=raw)

        backend = HFBackend.__new__(HFBackend)
        backend.family = "dream"
        backend.mask_id = 0
        backend.tokenizer = MockTokenizer()
        backend._torch = torch
        backend._model = ModelFixture()
        with patch.object(backend, "_record_cuda_peak_memory", wraps=backend._record_cuda_peak_memory) as record_memory:
            predictions = backend.predict([1, 2, 0, 0], [2, 3], [list(range(6))] * 2)
            record_memory.assert_called_once()
        self.assertEqual([row["token_id"] for row in predictions], [2, 3])
        self.assertEqual(predictions[0]["raw_top_id"], 0)
        self.assertEqual(calls[0]["attention_mask"], "full")
        self.assertFalse(calls[0]["use_cache"])
        self.assertAlmostEqual(predictions[0]["confidence"], torch.softmax(raw[0, 1], -1)[2].item())
        self.assertAlmostEqual(sum(predictions[0]["probabilities"].values()), 1.0, places=6)
        self.assertLess(predictions[0]["confidence"], 0.3)
        backend.family = "illada"
        self.assertEqual(backend.predict([1, 2, 0, 0], [2], [[3]])[0]["token_id"], 3)
        self.assertNotIn("attention_mask", calls[-1])
        raw[0, 2, 2] = float("nan")
        with self.assertRaises(RuntimeError):
            backend.predict([1, 2, 0, 0], [2], [[3]])


if __name__ == "__main__":
    unittest.main()
