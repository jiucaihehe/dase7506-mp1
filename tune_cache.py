"""Validation-only cache/temperature search; writes aggregate scores, never answers.

Each neural forward pass is shared across all candidates. Only true-token
probabilities are gathered for accumulating validation NLL. This is equivalent
to scoring full distributions, checked using the unmodified official scorer.
"""
import argparse
import itertools
import json
import math
from pathlib import Path
import time
import torch
from common import load_data, make_model, setup, sha, windows
from student import cache_attention


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        p.error('Use a new output directory.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device, _ = setup('cpu', 'fp32', args.threads)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model, _ = make_model('model', checkpoint['config'], device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    tokens, byte_count = load_data()['validation']
    temperatures = [0.9, 1.0, 1.1]
    thetas = [0.0, 4.0, 8.0, 16.0, 32.0]
    decays = [0.0, 0.01]
    weights = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]
    candidates = [(temp, theta, decay, weight) for temp in temperatures
                  for theta, decay, weight in itertools.product(thetas, decays, weights)
                  if weight != 0 or (theta == 0 and decay == 0)]
    totals = torch.zeros(len(candidates), dtype=torch.float64)
    started = time.perf_counter()
    for x, y in windows(tokens, batch_size=32):
        h = model.features(x)
        logits = model.head(h).float()
        valid = y != -100
        base = {temp: torch.softmax(logits / temp, -1).gather(
            -1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1) for temp in temperatures}
        next_ids = torch.cat((x[:, 1:], x[:, -1:]), dim=1)
        matching = next_ids[:, None, :] == y[:, :, None]
        cached = {(theta, decay): (cache_attention(h, theta, decay) * matching).sum(-1)
                  for theta, decay in itertools.product(thetas, decays)}
        for i, (temp, theta, decay, weight) in enumerate(candidates):
            probability = (1 - weight) * base[temp] + weight * cached[theta, decay]
            probability[:, 0] = base[temp][:, 0]
            totals[i] += -probability[valid].clamp_min(torch.finfo(torch.float32).tiny).log().double().sum()
    rows = [dict(temperature=temp, cache_theta=theta, cache_decay=decay,
                 cache_weight=weight, bpb=float(total) / math.log(2) / byte_count)
            for (temp, theta, decay, weight), total in zip(candidates, totals)]
    rows.sort(key=lambda row: row['bpb'])
    selected = rows[0]
    config = checkpoint['config'] | {k: v for k, v in selected.items() if k != 'bpb'}
    result = dict(split='validation', selected=selected, candidates=rows,
                  candidate_count=len(rows), seconds=time.perf_counter()-started,
                  source_checkpoint=str(args.checkpoint), source_sha256=sha(args.checkpoint),
                  train_tokens=checkpoint['train_tokens'], threads=args.threads,
                  selection_script_sha256=sha(Path(__file__)))
    (args.output_dir/'search.json').write_text(json.dumps(result, indent=2)+'\n')
    output = checkpoint | dict(implementation='student', config=config,
                               parent_checkpoint_sha256=sha(args.checkpoint),
                               selection_split='validation')
    torch.save(output, args.output_dir/'checkpoint.pt')
    print(json.dumps({k: v for k, v in result.items() if k != 'candidates'}, indent=2))


if __name__ == '__main__':
    main()
