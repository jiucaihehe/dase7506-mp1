"""Check fixed benchmark hashes and frozen inference assets without training."""
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    release = json.loads(Path('PACKAGE_MANIFEST.json').read_text())
    fixed = ['model.py','train.py','common.py','evaluate.py','configs/baseline.json',
             'tests/test_contract.py','requirements.txt']
    for name in fixed:
        assert sha(name)==release['code/'+name], f'Fixed file modified: {name}'
    data = json.loads(Path('data/manifest.json').read_text())
    for name,digest in data['sha256'].items():
        assert sha(Path('data')/name)==digest, f'Dataset mismatch: {name}'
    frozen = Path('artifacts/FREEZE.json')
    if frozen.exists():
        metadata = json.loads(frozen.read_text())
        for name,digest in metadata['code_sha256'].items():
            assert sha(name)==digest, f'Frozen code mismatch: {name}'
        for name,digest in metadata['checkpoint_sha256'].items():
            path=Path('artifacts')/name
            if name in ['final.pt','baseline.pt','baseline_cache.pt'] or path.exists():
                assert sha(path)==digest, f'Checkpoint mismatch: {name}'
        for name,digest in metadata.get('asset_sha256',{}).items():
            assert sha(name)==digest, f'Inference asset mismatch: {name}'
    print('Fixed benchmark and all available frozen assets: verified.')


if __name__ == '__main__':
    main()
