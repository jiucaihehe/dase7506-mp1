"""Validation-only selection of four interpolation weights by matched n-gram order."""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from common import load_data, make_model, setup, sha, windows
from ngram import NgramModel


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    if args.output_dir.exists(): p.error('Use a new output directory.')
    args.output_dir.mkdir(parents=True)
    setup('cpu','fp32',4)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    config=checkpoint['config']
    model,_=make_model('student',config|{'ngram_weight':0.0},torch.device('cpu'))
    model.load_state_dict(checkpoint['model']);model.eval()
    gram=NgramModel('assets/ngram.npz',config['ngram_discount'],config['ngram_order'])
    weights=np.arange(0,0.81,0.1).tolist()
    totals=np.zeros((4,len(weights)),np.float64)
    counts=np.zeros(4,np.int64)
    tokens,byte_count=load_data()['validation']
    started=time.perf_counter()
    for x,y in windows(tokens,32):
        valid=(y!=-100).numpy()
        neural=model.predict_log_probs(x).gather(-1,y.clamp_min(0).unsqueeze(-1)).squeeze(-1).exp().numpy()
        full,buckets=gram.probabilities(x.numpy(),return_buckets=True)
        probability=np.take_along_axis(full,y.clamp_min(0).numpy()[...,None],-1).squeeze(-1)
        for order in range(2,6):
            mask=valid&(buckets==order)
            counts[order-2]+=mask.sum()
            for index,weight in enumerate(weights):
                mixture=(1-weight)*neural[mask]+weight*probability[mask]
                totals[order-2,index]+=-np.log(np.maximum(mixture,np.finfo(np.float32).tiny)).astype(np.float64).sum()
    best=totals.argmin(axis=1)
    chosen=[0.0,0.0]+[round(weights[i],4) for i in best]
    bpb=float(sum(totals[i,j] for i,j in enumerate(best)))/math.log(2)/byte_count
    result=dict(split='validation',selected={'ngram_weights_by_order':chosen,'bpb':bpb},
                seconds=time.perf_counter()-started,weights_searched=weights,bucket_target_counts=counts.tolist(),
                bucket_nll_nats=totals.tolist(),scalar_settings_tested=36,
                note='Four independent bucket settings, each selected from nine fixed values; no training or test targets are fitted.',
                source_checkpoint=str(args.checkpoint),source_sha256=sha(args.checkpoint),
                train_tokens=checkpoint['train_tokens'])
    (args.output_dir/'search.json').write_text(json.dumps(result,indent=2)+'\n')
    torch.save(checkpoint|dict(config=config|{'ngram_weights_by_order':chosen},parent_checkpoint_sha256=sha(args.checkpoint)),
               args.output_dir/'checkpoint.pt')
    print(json.dumps({k:v for k,v in result.items() if k!='bucket_nll_nats'},indent=2))


if __name__=='__main__': main()
