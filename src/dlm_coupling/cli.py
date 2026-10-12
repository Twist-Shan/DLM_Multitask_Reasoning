"""Portable commands for dataset creation, auditing, execution and analysis."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from urllib.parse import quote
from urllib.request import Request, urlopen
from uuid import uuid4

from . import __version__
from .backends import make_backend
from .decoding import decode
from .experiment import audit, cases, validate_config, validate_dataset
from .metrics import save_summary
from .storage import canonical, digest, read_json, read_jsonl, write_json, write_jsonl
from .tasks.arithmetic import generate_families


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _environment():
    versions = {}
    for name in ("torch", "transformers", "numpy", "huggingface-hub", "safetensors"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": versions}


def _source_digest():
    root = Path(__file__).parent
    return digest({str(p.relative_to(root)): p.read_text(encoding="utf-8") for p in sorted(root.rglob("*.py"))})


def _git_revision():
    try:
        process = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        return process.stdout.strip() if process.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _load_data(path, limit=None):
    data = read_jsonl(path)
    validate_dataset(data)
    if limit is not None:
        if limit < 1 or limit > len(data):
            raise ValueError("limit-families must be between 1 and the dataset family count")
        data = data[:limit]
    return data


def _write_new_json(path, value):
    if Path(path).exists():
        raise ValueError(f"Refusing to overwrite {path}; use a new path")
    write_json(path, value)


def generate_command(args):
    config = read_json(args.config)
    validate_config(config)
    data = generate_families(**config["dataset"])
    validate_dataset(data)
    write_jsonl(args.out, data)
    print(f"Generated {len(data)} paired families -> {args.out}")


def validate_command(args):
    data = _load_data(args.data)
    semantics = {f["variants"][0]["metadata"]["semantic_id"] for f in data}
    for path in args.against:
        other = _load_data(path)
        overlap = semantics & {f["variants"][0]["metadata"]["semantic_id"] for f in other}
        if overlap:
            raise ValueError(f"Semantic overlap with {path}: {len(overlap)} families")
    print(f"Validated {len(data)} families; SHA256 {digest(data)}")


def pin_command(args):
    config = read_json(args.config)
    if config.get("backend") != "hf":
        raise ValueError("pin-model requires a Hugging Face model config")
    if Path(args.out).exists():
        raise ValueError(f"Refusing to overwrite {args.out}")
    model_id = config["model_id"]
    revision = config.get("revision", "main")
    url = f"https://huggingface.co/api/models/{quote(model_id, safe='/')}/revision/{quote(revision, safe='')}"
    with urlopen(Request(url, headers={"User-Agent": "dlm-coupling/0.1"}), timeout=30) as response:
        metadata = json.load(response)
    sha = metadata.get("sha", "")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", sha):
        raise ValueError("The official model API did not return an immutable revision")
    config.update(revision=sha, code_revision=sha, tokenizer_revision=sha,
                  pinned_from=revision, pinned_at=_utc())
    write_json(args.out, config)
    print(f"Pinned {model_id} to {sha}; no weights downloaded")


def audit_command(args):
    if Path(args.out).exists():
        raise ValueError(f"Refusing to overwrite {args.out}")
    data, config = _load_data(args.data, args.limit_families), read_json(args.config)
    models = [read_json(path) for path in args.model_config]
    fingerprints = [digest(model) for model in models]
    if len(set(fingerprints)) != len(models):
        raise ValueError("Duplicate model configuration in audit")
    audits = []
    for model in models:
        backend = make_backend(model)
        report, _ = audit(backend, data, config)
        report["model_config_digest"] = digest(model)
        audits.append(report)
    common = set.intersection(*(set(row["common_slot_integer_domain"]) for row in audits))
    report = {"status": "passed", "created_at": _utc(), "dataset_digest": digest(data),
              "config_digest": digest(config), "audits": audits,
              "common_integer_domain": sorted(common)}
    write_json(args.out, report)
    print(f"Audit passed for {len(models)} backend(s), {len(data)} families; no weights loaded -> {args.out}")


def _verify_audit(path, data, config, model):
    report = read_json(path)
    if report.get("status") != "passed" or report.get("dataset_digest") != digest(data) or report.get("config_digest") != digest(config):
        raise ValueError("Audit does not match this exact dataset and experiment config")
    records = report.get("audits", [])
    if digest(model) not in {r.get("model_config_digest") for r in records}:
        raise ValueError("This model revision/config was not included in the shared audit")
    families = {r["backend"].get("family") for r in records}
    if not {"dream", "illada"} <= families:
        raise ValueError("Scientific runs require a shared tokenizer audit containing both Dream and iLLaDA")


@contextmanager
def _exclusive_run_lock(output):
    """Own the run directory before inspecting or changing its saved state.

    A killed process can leave a stale lock. An operator must check that the
    owner is no longer running before removing it; active locks are never stolen.
    """
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".run.lock"
    owner = f"{uuid4()} pid={os.getpid()} created={_utc()}\n"
    try:
        lock = lock_path.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise ValueError(f"Run directory is locked: {lock_path}. Check the existing owner before removing a stale lock.") from exc
    identity = os.fstat(lock.fileno())
    owner_written = False
    try:
        with lock:
            lock.write(owner)
            lock.flush()
            owner_written = True
        yield lock_path
    finally:
        try:
            current = lock_path.stat()
            same_file = (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino)
            if same_file and (not owner_written or lock_path.read_text(encoding="utf-8") == owner):
                lock_path.unlink()
        except FileNotFoundError:
            pass


def run_experiment(data, config, model, output, resume=False, audit_report=None):
    validate_config(config)
    validate_dataset(data)
    output = Path(output)
    is_mock = model.get("backend") == "mock"
    if not is_mock:
        if not audit_report:
            raise ValueError("Run a shared tokenizer audit on both models first; provide --audit-report")
        _verify_audit(audit_report, data, config, model)
    env = _environment()
    identity = {"dataset_digest": digest(data), "experiment_config": config, "model_config": model,
                "source_digest": _source_digest(), "environment": env}
    fingerprint = digest(identity)
    manifest_path, results_path = output / "manifest.json", output / "results.jsonl"
    design = list(cases(data, config))
    expected_keys = {case["run_key"] for case in design}
    expected_nfe = sum(len(c["slots"]) - len(c["forced_values"]) for c in design)
    if len(expected_keys) != len(design):
        raise ValueError("Duplicate run key in experimental design")
    with _exclusive_run_lock(output) as lock_path:
        done = set()
        if any(item != lock_path for item in output.iterdir()):
            if not resume or not manifest_path.exists():
                raise ValueError(f"Nonempty run directory {output}; use --resume only for the same run")
            manifest = read_json(manifest_path)
            if manifest.get("fingerprint") != fingerprint:
                raise ValueError("Resume mismatch: data, config, source code or environment changed. Use a new run directory.")
            old_rows = read_jsonl(results_path) if results_path.exists() else []
            done = {r["run_key"] for r in old_rows}
            if len(done) != len(old_rows) or not done <= expected_keys:
                raise ValueError("Duplicate or unexpected records in resume file")
            if done == expected_keys:
                manifest.update(status="complete", completed_records=len(done), completed_at=manifest.get("completed_at", _utc()))
                write_json(manifest_path, manifest)
                print(f"Already complete: {len(done)} records -> {output}")
                return manifest
        else:
            output.mkdir(parents=True, exist_ok=True)
            manifest = {"schema_version": 1, "repo_version": __version__, **identity,
                        "fingerprint": fingerprint, "git_revision": _git_revision(),
                        "is_mock": is_mock, "created_at": _utc(), "status": "preflight",
                        "expected_records": len(design), "expected_nfe": expected_nfe,
                        "family_count": len(data), "timing_scope": "decode including compact CPU logging; excludes model/tokenizer load and one warm-up forward"}
            write_json(manifest_path, manifest)
        try:
            backend = make_backend(model)
            report, prepared = audit(backend, data, config)
            write_json(output / "local_audit.json", report)
            # Warm-up loads HF weights lazily and is excluded from experimental NFE.
            warm_case = next(c for c in design if c["branch"] == "baseline")
            canvas = prepared[(warm_case["prompt"], tuple(warm_case["slots"]))]
            warm_start = time.perf_counter()
            backend.predict(canvas.token_ids, list(canvas.slot_positions.values()),
                            [list(canvas.value_token_ids[key].values()) for key in canvas.slot_positions])
            manifest.update(status="running", backend=backend.metadata,
                            warmup_nfe=1, load_and_warmup_seconds=time.perf_counter() - warm_start)
            write_json(manifest_path, manifest)
            with results_path.open("a", encoding="utf-8") as stream:
                for case in design:
                    if case["run_key"] in done:
                        continue
                    canvas = prepared[(case["prompt"], tuple(case["slots"]))]
                    start = time.perf_counter()
                    trace = decode(backend, canvas, case["gold"], config.get("policy", "confidence"),
                                   case["forced_values"], config.get("starting_task", "A"))
                    record = {**case, "schema_version": 1, "is_mock": is_mock,
                              "slot_positions": canvas.slot_positions, "canvas_metadata": canvas.metadata,
                              "trace": trace, "elapsed_seconds": time.perf_counter() - start}
                    stream.write(canonical(record) + "\n")
                    stream.flush()
                    done.add(case["run_key"])
                    if len(done) % 50 == 0 or len(done) == len(design):
                        print(f"{len(done)}/{len(design)} records complete", flush=True)
            manifest.update(status="complete", completed_at=_utc(), completed_records=len(done), backend=backend.metadata)
            write_json(manifest_path, manifest)
            print(f"{'MOCK ONLY' if is_mock else 'Model run'}: {len(done)} records, {expected_nfe} experimental forwards -> {output}")
            return manifest
        except BaseException as exc:
            manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=f"{type(exc).__name__}: {exc}", completed_records=len(done))
            write_json(manifest_path, manifest)
            raise


def run_command(args):
    return run_experiment(_load_data(args.data, args.limit_families), read_json(args.config),
                          read_json(args.model_config), args.out, args.resume, args.audit_report)


def summarize_command(args):
    path = Path(args.run)
    manifest = read_json(path / "manifest.json")
    records = read_jsonl(path / "results.jsonl")
    if manifest.get("status") != "complete" or len(records) != manifest["expected_records"]:
        raise ValueError("Only complete runs can be summarized; resume this run first")
    result = save_summary(args.out, records, manifest)
    print(f"Saved {len(result['metrics'])} clustered summaries -> {args.out}")


def smoke_command(args):
    path = Path(args.out)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Smoke output must be new/empty: {path}")
    config = {"name": "smoke", "dataset": {"count": 2, "seed": 123, "split": "smoke", "depth": 2},
              "formats": ["answer_only", "with_intermediate"],
              "layouts": ["AB_AB", "AB_BA", "BA_AB", "BA_BA"],
              "policy": "confidence", "interventions": True,
              "bootstrap_seed": 123, "bootstrap_samples": 200}
    data = generate_families(**config["dataset"])
    path.mkdir(parents=True, exist_ok=True)
    write_jsonl(path / "dataset.jsonl", data)
    manifest = run_experiment(data, config, {"backend": "mock", "family": "mock", "seed": 123}, path / "run")
    save_summary(path / "report", read_jsonl(path / "run" / "results.jsonl"), manifest)
    print("Smoke completed. Mock outputs verify plumbing only and must not be reported as model results.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("generate")
    p.add_argument("--config", required=True); p.add_argument("--out", required=True)
    p.set_defaults(func=generate_command)
    p = commands.add_parser("validate")
    p.add_argument("--data", required=True); p.add_argument("--against", action="append", default=[])
    p.set_defaults(func=validate_command)
    p = commands.add_parser("pin-model")
    p.add_argument("--config", required=True); p.add_argument("--out", required=True)
    p.set_defaults(func=pin_command)
    p = commands.add_parser("audit")
    p.add_argument("--data", required=True); p.add_argument("--config", required=True)
    p.add_argument("--model-config", action="append", required=True); p.add_argument("--out", required=True)
    p.add_argument("--limit-families", type=int)
    p.set_defaults(func=audit_command)
    p = commands.add_parser("run")
    p.add_argument("--data", required=True); p.add_argument("--config", required=True)
    p.add_argument("--model-config", required=True); p.add_argument("--out", required=True)
    p.add_argument("--audit-report"); p.add_argument("--limit-families", type=int)
    p.add_argument("--resume", action="store_true")
    p.set_defaults(func=run_command)
    p = commands.add_parser("summarize")
    p.add_argument("--run", required=True); p.add_argument("--out", required=True)
    p.set_defaults(func=summarize_command)
    p = commands.add_parser("smoke")
    p.add_argument("--out", default="runs/smoke")
    p.set_defaults(func=smoke_command)
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Error: {exc}\n")
