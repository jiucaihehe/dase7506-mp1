"""Recreate inference-only ablations from the final weights; no training occurs."""
import json
from pathlib import Path
import torch


def main():
    root=Path('artifacts')
    source=torch.load(root/'final.pt',map_location='cpu',weights_only=True)
    changes={'neural_only':dict(cache_weight=0.0,ngram_weight=0.0),
             'no_cache':dict(cache_weight=0.0),'no_ngram':dict(ngram_weight=0.0),
             'uniform_cache':dict(cache_theta=0.0),
             'uncalibrated_neural':dict(cache_weight=0.0,ngram_weight=0.0,temperature=1.0)}
    for name,delta in changes.items():
        output=root/f'{name}.pt'
        if not output.exists(): torch.save(source|dict(config=source['config']|delta),output)
    constant=dict(source['config']);constant.pop('ngram_weights_by_order',None)
    if not (root/'constant_ngram.pt').exists():
        torch.save(source|dict(config=constant),root/'constant_ngram.pt')
    print('Ablation checkpoint configurations materialized from the frozen weights; no optimization performed.')


if __name__=='__main__':main()
