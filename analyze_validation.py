"""Aggregate validation diagnostics; no token-level answers are persisted."""
import json
from pathlib import Path
import torch
from common import load_data, make_model, setup, windows


@torch.no_grad()
def main():
    setup('cpu','fp32',4)
    checkpoint = torch.load('artifacts/final.pt',map_location='cpu',weights_only=True)
    model, _ = make_model('student',checkpoint['config'],torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    model.eval()
    weight = model.cache_weight
    tokens, _ = load_data()['validation']
    buckets = {name:dict(targets=0,neural_nll=0.,cache_nll=0.) for name in
               ['seen_in_cache','not_seen_in_cache','positions_0_31','positions_32_63',
                'positions_64_127','positions_128_255']}
    for x,y in windows(tokens,32):
        model.cache_weight = weight
        cached = -model.predict_log_probs(x).gather(-1,y.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        model.cache_weight = 0.0
        neural = -model.predict_log_probs(x).gather(-1,y.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        positions = torch.arange(x.shape[1])
        allowed = positions[:,None] > positions[None,:]
        next_ids = torch.cat((x[:,1:],x[:,-1:]),1)
        seen = ((next_ids[:,None,:] == y[:,:,None]) & allowed).any(-1)
        valid = y != -100
        masks = dict(seen_in_cache=seen,not_seen_in_cache=~seen)
        for low,high in [(0,31),(32,63),(64,127),(128,255)]:
            masks[f'positions_{low}_{high}'] = ((positions>=low)&(positions<=high))[None,:]
        for name,mask in masks.items():
            mask = mask & valid
            buckets[name]['targets'] += int(mask.sum())
            buckets[name]['neural_nll'] += float(neural[mask].double().sum())
            buckets[name]['cache_nll'] += float(cached[mask].double().sum())
    for row in buckets.values():
        row['nats_saved_per_target'] = (row['neural_nll']-row['cache_nll'])/row['targets']
    result = dict(split='validation',buckets=buckets,
                  interpretation='Positive nats_saved_per_target means the continuous cache improves this group while the statistical component and all other settings remain fixed. neural_nll names the no-cache comparator, which can still include the statistical mixture. Categories are descriptive, not additional selection criteria.')
    Path('results').mkdir(exist_ok=True)
    Path('results/validation_diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
