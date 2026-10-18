"""Run one bounded iLLaDA calibration and preserve partial data on failure."""
import hashlib
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-archive', default='/content/answer-graphs-v1-code.zip')
    parser.add_argument('--result-prefix', default='/content/answer-graphs-v1-private-results')
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0):
        raise RuntimeError('This calibration requires the requested A100')
    (ROOT / 'runs').mkdir(exist_ok=True)
    if Path(args.result_prefix + '.zip').exists():
        raise FileExistsError('Preserve prior archives; use a fresh result prefix')
    shutil.copyfile(args.code_archive, ROOT / 'runs/code.zip')
    (ROOT / 'runs/nvidia-smi.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True))
    status = dict(status='running', seed=123, gpu=torch.cuda.get_device_name(0), torch=torch.__version__)
    started = time.monotonic()
    def save():
        status['elapsed_seconds'] = time.monotonic() - started
        (ROOT / 'runs/suite_status.json').write_text(json.dumps(status, indent=2))
    save()
    try:
        command_args = [sys.executable, 'scripts/run_answer_graphs.py', '--out', 'runs/illada-answer-graphs-v1',
                '--seed', '123', '--per-cell', '10', '--max-seconds', '3600']
        proc = subprocess.Popen(command_args, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        timer = threading.Timer(3660, proc.kill); timer.start()
        try:
            with (ROOT / 'runs/inference-stdout.txt').open('w') as log:
                for line in proc.stdout:
                    log.write(line); log.flush(); print(line, end='', flush=True)
            if proc.wait() != 0:
                raise RuntimeError('Calibration failed or exceeded wall-time budget')
        finally:
            timer.cancel()
        manifest = json.loads((ROOT / 'runs/illada-answer-graphs-v1/manifest.json').read_text())
        status.update(status=manifest['status'], rows=manifest['rows'], compute=manifest['compute'])
    except Exception as exc:
        status.update(status='failed', error=repr(exc))
    finally:
        save()
        archive = Path(shutil.make_archive(args.result_prefix, 'zip', ROOT, 'runs'))
        print('SUITE_STATUS', json.dumps(status), flush=True)
        print('ARCHIVE_METADATA', archive.stat().st_size, hashlib.sha256(archive.read_bytes()).hexdigest(), flush=True)
        print('COMPUTE_COMPLETE: verify download, then release A100.', flush=True)


if __name__ == '__main__':
    main()
