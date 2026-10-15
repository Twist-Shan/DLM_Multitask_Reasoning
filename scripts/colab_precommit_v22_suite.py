"""A100 supervisor; archive partial results on failure and preserve frozen code."""
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
    assert torch.cuda.is_available() and 'A100' in torch.cuda.get_device_name(0)
    (ROOT/'runs').mkdir(exist_ok=True)
    shutil.copyfile('/content/precommit-v22-code.zip',ROOT/'runs/code-v22.zip')
    (ROOT/'runs/nvidia-smi.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
    suite=dict(status='running',seed=123,models={},gpu=torch.cuda.get_device_name(0),torch=torch.__version__)
    started=time.monotonic()
    def save():
        suite['elapsed_seconds']=time.monotonic()-started
        (ROOT/'runs/suite_status.json').write_text(json.dumps(suite,indent=2))
    save();print('GPU',suite['gpu'],suite['torch'],flush=True)
    for model in ('illada','dream'):
        out=Path('runs')/f'{model}-v22'
        args=[sys.executable,'scripts/run_precommit_v22.py','--model',model,'--seed','123',
              '--out',str(out),'--max-seconds','3600']
        print('START_MODEL',model,flush=True)
        try:
            proc=subprocess.Popen(args,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            timer=threading.Timer(3660,proc.kill);timer.start()
            try:
                with (ROOT/'runs'/f'{model}-stdout.txt').open('w') as log:
                    for line in proc.stdout:log.write(line);log.flush();print(line,end='',flush=True)
                if proc.wait()!=0:raise RuntimeError('Model process failed or exceeded budget')
            finally:timer.cancel()
            meta=json.loads((ROOT/out/'manifest.json').read_text())
            suite['models'][model]=dict(status=meta['status'],outputs=meta['outputs'],baseline=meta['baseline_diagnostics'])
        except Exception as exc:
            suite['models'][model]=dict(status='failed',error=repr(exc))
        save();print('MODEL_STATUS',model,json.dumps(suite['models'][model]),flush=True)
        if suite['models'][model]['status']=='failed':break
    suite['status']='finished' if len(suite['models'])==2 and all(x['status']=='complete' for x in suite['models'].values()) else 'finished_with_errors'
    save()
    archive=Path(shutil.make_archive('/content/precommit-v22-private-results','zip',ROOT,'runs'))
    print('SUITE_STATUS',json.dumps(suite),flush=True)
    print('ARCHIVE_METADATA',archive.stat().st_size,hashlib.sha256(archive.read_bytes()).hexdigest(),flush=True)
    print('COMPUTE_COMPLETE: verify local archive, then release runtime.',flush=True)


if __name__=='__main__':main()
