"""Download public benchmark inputs, verify pinned bytes, preserve provenance.

MuSiQue uses a community mirror because the official Drive archive was quota
blocked. Equality to that official archive has NOT been independently verified.
The external semantic mapping is read as literal data, never executed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

SOURCES = {
    'musique-dev.jsonl': (
        'https://huggingface.co/datasets/dgslibisey/MuSiQue/resolve/c8f4f8c9465fb69d31a8eae894c3fd509c4ca321/musique_ans_v1.0_dev.jsonl',
        '15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b'),
    'stepgame-valid.jsonl': (
        'https://huggingface.co/datasets/ZhengyanShi/StepGame/resolve/6d859381dfd518cae3f073b268aaa323bf4dcf04/valid.jsonl',
        '90c7efa0de6bd2a03c9813beaf94e910be803da0e1f08e38557c21b1ad49b5a9'),
    'stepgame-semantic-mapping.py': (
        'https://raw.githubusercontent.com/Fangjun-Li/SpatialLM-StepGame/main/sentence_to_relation.py',
        '7a2d0e269aeb0080e52e2d69b83a02b53d291e569c2bad926ad7ffeddb6cdac2'),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=Path('data/external/realistic-v1'))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, (url, sha) in SOURCES.items():
        target = args.out / name
        if target.exists():
            content = target.read_bytes()
        else:
            with urllib.request.urlopen(url, timeout=120) as response:
                content = response.read()
        if hashlib.sha256(content).hexdigest() != sha:
            raise ValueError(f'Hash mismatch: {name}. Refusing changed dataset/template bytes.')
        if not target.exists():
            with target.open('xb') as f:
                f.write(content)
        target.with_name(name + '.provenance.json').write_text(json.dumps(
            dict(url=url, sha256=sha, bytes=len(content)), indent=2), encoding='utf-8')
        print(name, len(content), sha)


if __name__ == '__main__':
    main()
