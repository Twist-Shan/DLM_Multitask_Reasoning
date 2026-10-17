"""Bounded GPU run with archive preservation on failure."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    import torch
    if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0):
        raise RuntimeError('This run requires the requested A100')
    (ROOT/'runs').mkdir(exist_ok=True)
    shutil.copyfile('/content/precommit-v24-code.zip',ROOT/'runs/code-v24.zip')
    (ROOT/'runs/nvidia-smi.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
    suite=dict(status='running',seed=123,gpu=torch.cuda.get_device_name(0),torch=torch.__version__)
    started=time.monotonic()
    def save():
        suite['elapsed_seconds']=time.monotonic()-started
        (ROOT/'runs/suite_status.json').write_text(json.dumps(suite,indent=2))
    save();print('GPU',suite['gpu'],suite['torch'],flush=True)
    try:
        args=[sys.executable,'scripts/run_precommit_v24.py','--seed','123','--out','runs/illada-v24','--max-seconds','5400']
        proc=subprocess.Popen(args,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        timer=threading.Timer(5460,proc.kill);timer.start()
        try:
            with (ROOT/'runs/illada-stdout.txt').open('w') as log:
                for line in proc.stdout:log.write(line);log.flush();print(line,end='',flush=True)
            if proc.wait()!=0:raise RuntimeError('Run failed or exceeded budget')
        finally:timer.cancel()
        meta=json.loads((ROOT/'runs/illada-v24/manifest.json').read_text())
        suite.update(status=meta['status'],outputs=meta['outputs'],compute=meta['compute'])
    except Exception as exc:
        suite.update(status='failed',error=repr(exc))
    finally:
        save()
        archive=Path(shutil.make_archive('/content/precommit-v24-private-results','zip',ROOT,'runs'))
        print('SUITE_STATUS',json.dumps(suite),flush=True)
        print('ARCHIVE_METADATA',archive.stat().st_size,hashlib.sha256(archive.read_bytes()).hexdigest(),flush=True)
        print('COMPUTE_COMPLETE: export and verify locally, then release A100.',flush=True)


if __name__=='__main__':main()
