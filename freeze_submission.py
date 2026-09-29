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
    searches = {path.parent.name:json.loads(path.read_text()) for path in sorted(Path('runs').glob('*/search.json'))}
    selected_run = min(searches, key=lambda name: searches[name]['selected']['bpb'])
    selected_checkpoint = Path(f'runs/{selected_run}/checkpoint.pt')
    source = torch.load(selected_checkpoint, map_location='cpu', weights_only=True)
    # Preserve tied weight storage in the bundle without changing any numbers.
    if not torch.equal(source['model']['head.weight'],source['model']['token.weight']):
        raise ValueError('Expected tied input/output embeddings.')
    source['model']['head.weight']=source['model']['token.weight']
    torch.save(source,args.output_dir/'final.pt')
    shutil.copy2('runs/baseline-s17/checkpoint.pt', args.output_dir/'baseline.pt')
    shutil.copy2('runs/baseline-cache-s17/checkpoint.pt', args.output_dir/'baseline_cache.pt')
    variants = {
        'neural_only': dict(cache_weight=0.0,ngram_weight=0.0),
        'no_cache': dict(cache_weight=0.0),
        'no_ngram': dict(ngram_weight=0.0),
        'uniform_cache': dict(cache_theta=0.0),
        'uncalibrated_neural': dict(cache_weight=0.0,ngram_weight=0.0,temperature=1.0),
    }
    for name, changes in variants.items():
        torch.save(source | dict(config=source['config'] | changes), args.output_dir/f'{name}.pt')
    constant_config=dict(source['config'])
    constant_config.pop('ngram_weights_by_order',None)
    torch.save(source|dict(config=constant_config),args.output_dir/'constant_ngram.pt')
    files = ['student.py','model.py','ngram.py','common.py','evaluate.py','train.py',
             'train_experiment.py','train_continue.py','train_mps.py','tune_cache.py',
             'tune_hybrid.py','tune_gate.py','build_ngram.py','measure_eval.py',
             'benchmark_frozen.py','freeze_submission.py','verify_package.py']
    files+=['materialize_ablations.py']
    files += ['requirements.txt','configs/baseline.json','configs/wide.json',
              'configs/regularized256.json','data/tokenizer.json']
    inference_assets=['artifacts/final.pt','student.py','model.py']
    asset_hashes={}
    if float(source['config'].get('ngram_weight',0))>0:
        inference_assets+=['ngram.py','assets/ngram.npz']
        asset_hashes['assets/ngram.npz']=sha(Path('assets/ngram.npz'))
    frozen = dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
                  selected_on='validation', selected_run=selected_run,
                  validation_bpb=searches[selected_run]['selected']['bpb'],
                  config=source['config'], train_tokens=source['train_tokens'],
                  seed=source['seed'],
                  ngram_training_targets=source.get('ngram_training_targets',0),
                  inference_assets=inference_assets,asset_sha256=asset_hashes,
                  ablation_names=['baseline_cache','neural_only','no_cache','no_ngram',
                                  'uniform_cache','uncalibrated_neural','constant_ngram'],
                  code_sha256={path:sha(Path(path)) for path in files},
                  checkpoint_sha256={path.name:sha(path) for path in sorted(args.output_dir.glob('*.pt'))},
                  candidate_validation_bpb={name:r['selected']['bpb'] for name,r in searches.items()},
                  notes='All variants frozen together before any test scoring. Ablations reuse the same fitted components and disable one inference mechanism; shared n-gram fitting cost is disclosed separately from gradient training.')
    (args.output_dir/'FREEZE.json').write_text(json.dumps(frozen, indent=2)+'\n')
    print(json.dumps(frozen, indent=2))


if __name__ == '__main__':
    main()
