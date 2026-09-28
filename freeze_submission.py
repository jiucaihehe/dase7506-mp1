"""Select by VALIDATION and freeze all predictors before test evaluation."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import torch
from common import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=Path('artifacts'))
    args = p.parse_args()
    if args.output_dir.exists():
        p.error('Refusing to overwrite a frozen bundle.')
    args.output_dir.mkdir()
    searches = {name: json.loads(Path(f'runs/{name}/search.json').read_text())
                for name in ['baseline-cache-s17', 'wide-cache-s17']}
    selected_run = min(searches, key=lambda name: searches[name]['selected']['bpb'])
    selected_checkpoint = Path(f'runs/{selected_run}/checkpoint.pt')
    shutil.copy2(selected_checkpoint, args.output_dir/'final.pt')
    shutil.copy2('runs/baseline-s17/checkpoint.pt', args.output_dir/'baseline.pt')
    shutil.copy2('runs/baseline-cache-s17/checkpoint.pt', args.output_dir/'baseline_cache.pt')
    source = torch.load(selected_checkpoint, map_location='cpu', weights_only=True)
    variants = {
        'no_cache': dict(cache_weight=0.0),
        'uniform_cache': dict(cache_theta=0.0),
        'uncalibrated_neural': dict(cache_weight=0.0, temperature=1.0),
    }
    for name, changes in variants.items():
        torch.save(source | dict(config=source['config'] | changes), args.output_dir/f'{name}.pt')
    files = ['student.py','model.py','common.py','evaluate.py','train.py',
             'train_experiment.py','tune_cache.py','requirements.txt',
             'configs/baseline.json','configs/wide.json','data/tokenizer.json']
    frozen = dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
                  selected_on='validation', selected_run=selected_run,
                  validation_bpb=searches[selected_run]['selected']['bpb'],
                  config=source['config'], train_tokens=source['train_tokens'],
                  seed=source['seed'],
                  inference_assets=['artifacts/final.pt','student.py','model.py'],
                  code_sha256={path:sha(Path(path)) for path in files},
                  checkpoint_sha256={path.name:sha(path) for path in sorted(args.output_dir.glob('*.pt'))},
                  candidate_validation_bpb={name:r['selected']['bpb'] for name,r in searches.items()},
                  notes='All variants frozen together before any full-test scoring. The uniform-cache ablation changes only theta; no-cache changes only lambda.')
    (args.output_dir/'FREEZE.json').write_text(json.dumps(frozen, indent=2)+'\n')
    print(json.dumps(frozen, indent=2))


if __name__ == '__main__':
    main()
