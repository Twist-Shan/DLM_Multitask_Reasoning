"""Validate private archive bytes and safely extract before analysis."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('archive',type=Path);ap.add_argument('--sha256',required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    raw=a.archive.read_bytes();assert hashlib.sha256(raw).hexdigest()==a.sha256
    a.out.mkdir(parents=True,exist_ok=False)
    dest=a.out/'artifacts';dest.mkdir()
    with zipfile.ZipFile(a.archive) as z:
        assert len(z.namelist())==len(set(z.namelist()))
        for item in z.infolist():
            p=PurePosixPath(item.filename)
            if p.is_absolute() or '..' in p.parts or '\\' in item.filename or ':' in item.filename:
                raise ValueError('Unsafe path')
            if ((item.external_attr>>16)&0o170000)==0o120000:raise ValueError('Symlink')
            target=dest.joinpath(*p.parts)
            if item.is_dir():target.mkdir(parents=True,exist_ok=True)
            else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(item))
    shutil.copyfile(a.archive,a.out/'results.zip')
    runs=dest/'runs';suite=json.loads((runs/'suite_status.json').read_text())
    assert suite['status']=='finished',suite
    counts={}
    for model,splits in [('illada',['calibration','discovery']),('dream',['discovery'])]:
        for split in splits:
            names={'factorial':240,**({'contexts':660,'patch':7440} if model=='illada' else {})}
            for name,n in names.items():
                path=runs/(model+'-v22')/split/(name+'.jsonl')
                lines=path.read_text(encoding='utf-8').splitlines();assert len(lines)==n
                for line in lines:json.loads(line)
                counts[f'{model}/{split}/{name}']=n
    record=dict(sha256=a.sha256,bytes=len(raw),counts=counts,suite_status=suite['status'])
    (a.out/'download-validation.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
