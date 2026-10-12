"""Deterministic mechanical fixture, never a scientific model or an oracle."""

import hashlib
import math
import re
from typing import Any

from .base import PreparedCanvas, build_canvas, validate_prediction_request


class MockTokenizer:
    """A tiny reversible tokenizer with integer tokens for CPU-only checks."""

    mask_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if add_special_tokens:
            raise ValueError("The fixture has no implicit special tokens.")
        tokens = []
        for piece in re.findall(r"<mask>|[0-9]+|[\s\S]", text):
            if piece == "<mask>":
                tokens.append(0)
            elif piece.isascii() and piece.isdigit() and str(int(piece)) == piece and int(piece) < 100:
                tokens.append(int(piece) + 1)
            else:
                tokens.extend(1000 + ord(char) for char in piece)
        return tokens

    def decode(self, token_ids: list[int], **kwargs: Any) -> str:
        pieces = []
        for token_id in token_ids:
            if token_id == 0:
                pieces.append("<mask>")
            elif 1 <= token_id <= 100:
                pieces.append(str(token_id - 1))
            elif token_id == 101:
                pieces.append("?")
            elif token_id >= 1000:
                pieces.append(chr(token_id - 1000))
            else:
                raise ValueError(f"Unknown fixture token ID: {token_id}")
        return "".join(pieces)

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        return "<user>\n" + messages[0]["content"] + "\n<assistant>\n"


class MockBackend:
    mask_id = 0

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = dict(config or {})
        self.tokenizer = MockTokenizer()
        self.metadata = {
            "backend": "mock",
            "model_id": "mechanical-fixture",
            "revision": "mock-v1",
            "is_mock": True,
            "scientific_results": False,
            "description": "Input-hash fixture for file, decoder and intervention tests; does not solve tasks.",
            "seed": int(self.config.get("seed", 123)),
            "mask_id": self.mask_id,
        }

    def prepare(self, prompt: str, slot_keys: list[str]) -> PreparedCanvas:
        return build_canvas(self.tokenizer, prompt, slot_keys, self.mask_id)

    def decode_token(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id])

    def predict(
        self,
        token_ids: list[int],
        positions: list[int],
        value_token_ids: list[list[int]],
    ) -> list[dict[str, Any]]:
        validate_prediction_request(token_ids, positions, value_token_ids)
        payload = f"{self.metadata['seed']}:{','.join(map(str, token_ids))}".encode("ascii")
        digest = hashlib.sha256(payload).digest()
        result = []
        for position, probes in zip(positions, value_token_ids):
            if any(probe < 0 or probe > 101 for probe in probes):
                raise ValueError("Fixture probes must be in the 102-token prediction vocabulary.")
            winner = (int.from_bytes(digest[:4], "big") + position * 7) % 101 + 1
            weights = [1.0] * 102
            weights[winner] = 30.0 + digest[4] / 255.0
            # Some states make MASK the raw top token, exercising exclusion.
            weights[0] = 40.0 if digest[5] % 7 == 0 else 1.0
            total = math.fsum(weights)
            raw_top = max(range(len(weights)), key=weights.__getitem__)
            result.append({
                "token_id": winner,
                "token_text": self.decode_token(winner),
                "confidence": weights[winner] / total,
                "raw_top_id": raw_top,
                "raw_top_text": self.decode_token(raw_top),
                "probabilities": {str(probe): weights[probe] / total for probe in probes},
                "is_mock": True,
            })
        return result

