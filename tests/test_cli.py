import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dlm_coupling.cli import run_experiment, _verify_audit
from dlm_coupling.storage import read_json, read_jsonl, write_json
from dlm_coupling.tasks.arithmetic import generate_families


class CLITests(unittest.TestCase):
    def test_complete_run_resume_and_crash_after_last_record(self):
        config = {"formats": ["answer_only"], "layouts": ["AB_AB"], "interventions": True}
        data = generate_families(1, 17)
        model = {"backend": "mock", "family": "mock"}
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            run = Path(directory) / "run"
            manifest = run_experiment(data, config, model, run)
            rows = read_jsonl(run / "results.jsonl")
            self.assertEqual(len(rows), 10)
            self.assertEqual(sum(r["trace"]["nfe"] for r in rows), manifest["expected_nfe"])
            self.assertTrue(manifest["is_mock"])
            manifest["status"] = "failed"
            write_json(run / "manifest.json", manifest)
            run_experiment(data, config, model, run, resume=True)
            self.assertEqual(read_json(run / "manifest.json")["status"], "complete")
            self.assertEqual(read_jsonl(run / "results.jsonl"), rows)
            self.assertFalse((run / ".run.lock").exists())
            with self.assertRaisesRegex(ValueError, "Resume mismatch"):
                run_experiment(data, {**config, "policy": "alternating"}, model, run, resume=True)
            with self.assertRaisesRegex(ValueError, "Nonempty"):
                run_experiment(data, config, model, run)

    def test_real_models_require_exact_shared_audit_before_loading(self):
        data = generate_families(1, 17)
        config = {"formats": ["answer_only"], "layouts": ["AB_AB"]}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "shared tokenizer audit"):
                run_experiment(data, config, {"backend": "hf", "family": "dream"}, Path(directory) / "run")
            self.assertFalse((Path(directory) / "run").exists())

    def test_existing_lock_prevents_any_manifest_access_or_model_loading(self):
        data = generate_families(1, 17)
        config = {"formats": ["answer_only"], "layouts": ["AB_AB"]}
        model = {"backend": "mock", "family": "mock"}
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            run.mkdir()
            lock = run / ".run.lock"
            lock.write_text("another active process", encoding="utf-8")
            with patch("dlm_coupling.cli.read_json") as reader, \
                    patch("dlm_coupling.cli.write_json") as writer, \
                    patch("dlm_coupling.cli.make_backend") as loader:
                with self.assertRaisesRegex(ValueError, "is locked"):
                    run_experiment(data, config, model, run)
                reader.assert_not_called()
                writer.assert_not_called()
                loader.assert_not_called()
            self.assertEqual(lock.read_text(encoding="utf-8"), "another active process")
            self.assertEqual({path.name for path in run.iterdir()}, {".run.lock"})

    def test_complete_resume_also_respects_an_existing_lock(self):
        data = generate_families(1, 17)
        config = {"formats": ["answer_only"], "layouts": ["AB_AB"]}
        model = {"backend": "mock", "family": "mock"}
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            run = Path(directory) / "run"
            run_experiment(data, config, model, run)
            before = {name: (run / name).read_bytes() for name in ("manifest.json", "results.jsonl")}
            lock = run / ".run.lock"
            lock.write_text("another resume process", encoding="utf-8")
            with patch("dlm_coupling.cli.read_json") as reader, patch("dlm_coupling.cli.write_json") as writer:
                with self.assertRaisesRegex(ValueError, "is locked"):
                    run_experiment(data, config, model, run, resume=True)
                reader.assert_not_called()
                writer.assert_not_called()
            for name, contents in before.items():
                self.assertEqual((run / name).read_bytes(), contents)
            self.assertEqual(lock.read_text(encoding="utf-8"), "another resume process")

    def test_backend_failure_releases_own_lock_and_allows_resume(self):
        data = generate_families(1, 17)
        config = {"formats": ["answer_only"], "layouts": ["AB_AB"]}
        model = {"backend": "mock", "family": "mock"}
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            run = Path(directory) / "run"
            with patch("dlm_coupling.cli.make_backend", side_effect=RuntimeError("injected load failure")):
                with self.assertRaisesRegex(RuntimeError, "injected load failure"):
                    run_experiment(data, config, model, run)
            self.assertEqual(read_json(run / "manifest.json")["status"], "failed")
            self.assertFalse((run / ".run.lock").exists())
            self.assertEqual(run_experiment(data, config, model, run, resume=True)["status"], "complete")
            self.assertFalse((run / ".run.lock").exists())

    def test_cleanup_does_not_remove_a_replaced_lock(self):
        data = generate_families(1, 17)
        config = {"formats": ["answer_only"], "layouts": ["AB_AB"]}
        model = {"backend": "mock", "family": "mock"}
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"

            def change_owner_and_fail(_):
                (run / ".run.lock").write_text("replacement owner", encoding="utf-8")
                raise RuntimeError("injected failure after lock replacement")

            with patch("dlm_coupling.cli.make_backend", side_effect=change_owner_and_fail):
                with self.assertRaisesRegex(RuntimeError, "after lock replacement"):
                    run_experiment(data, config, model, run)
            self.assertEqual((run / ".run.lock").read_text(encoding="utf-8"), "replacement owner")


if __name__ == "__main__":
    unittest.main()
