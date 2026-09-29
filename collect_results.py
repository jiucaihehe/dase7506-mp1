"""Collect measured evidence without selecting on test or modifying predictors."""
import csv
import json
from pathlib import Path


def main():
    load=lambda p:json.loads(Path(p).read_text())
    frozen=load('artifacts/FREEZE.json');benchmark=load('results/benchmark.json')
    validation=load('results/validation_ablations.json')
    training=[]
    for path in sorted(Path('runs').glob('*/metrics.json')):
        row=load(path)
        training.append(dict(run=path.parent.name,device=row.get('device','cpu'),seed=row['seed'],
                             parameters=row['parameters'],train_seconds=row['train_seconds'],
                             cumulative_targets=row['train_tokens'],
                             new_targets=row.get('new_train_tokens',row['train_tokens']),
                             best_step=row.get('best_step',1200),
                             validation_bpb=row.get('best_validation_bpb',row['validation']['bpb']),
                             parent_checkpoint_sha256=row.get('parent_checkpoint_sha256')))
    searches=[]
    for path in sorted(Path('runs').glob('*/search.json')):
        row=load(path)
        searches.append(dict(run=path.parent.name,seconds=row['seconds'],selected=row['selected'],
                             train_tokens=row['train_tokens'],candidate_count=row.get('candidate_count'),
                             scalar_settings_tested=row.get('scalar_settings_tested')))
    tests={r['name']:r for r in benchmark['records']}
    comparisons=[]
    for name in ['baseline','baseline_cache','uncalibrated_neural','neural_only','no_ngram',
                 'no_cache','uniform_cache','constant_ngram','final']:
        comparisons.append(dict(name=name,validation_bpb=validation[name]['bpb'],test_bpb=tests[name]['bpb'],
                                seconds=tests[name]['seconds']))
    result=dict(frozen=frozen,benchmark=benchmark,training=training,searches=searches,
                comparisons=comparisons,total_new_gradient_targets=sum(r['new_targets'] for r in training),
                total_train_seconds=sum(r['train_seconds'] for r in training),
                total_search_seconds=sum(r['seconds'] for r in searches),ngram_build=load('assets/ngram_build.json'),
                discarded_ngram_build_seconds=13.267145249992609,
                note='Ablations reuse all fitted components and vary inference settings. Statistical fitting scans and gradient targets are reported separately; no test-based selection.')
    Path('results/summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with Path('results/comparisons.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(comparisons[0]));writer.writeheader();writer.writerows(comparisons)
    with Path('results/training_costs.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(training[0]));writer.writeheader();writer.writerows(training)


if __name__=='__main__':main()
