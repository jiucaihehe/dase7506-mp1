"""Validation-only selection of train-derived KN / GPT-cache interpolation."""
import argparse
import itertools
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
    model,_=make_model('student',checkpoint['config'],torch.device('cpu'))
    model.load_state_dict(checkpoint['model']);model.eval()
    discounts=[0.5,0.75,0.9]
    orders=[3,4,5]
    weights=[0.0,0.1,0.2,0.3,0.4,0.5,0.6]
    candidates=[(d,n,w) for d,n,w in itertools.product(discounts,orders,weights)
                if w!=0 or (d==0.75 and n==5)]
    totals=np.zeros(len(candidates),np.float64)
    grams={d:NgramModel('assets/ngram.npz',d,5) for d in discounts}
    tokens,byte_count=load_data()['validation']
    started=time.perf_counter()
    for x,y in windows(tokens,32):
        valid=(y!=-100).numpy()
        neural=model.predict_log_probs(x).gather(-1,y.clamp_min(0).unsqueeze(-1)).squeeze(-1).exp().numpy()
        gather=y.clamp_min(0).numpy()[...,None]
        probabilities={}
        for discount,gram in grams.items():
            full=gram.probabilities(x.numpy(),return_orders=True)
            for order in orders:
                probabilities[discount,order]=np.take_along_axis(full[order],gather,axis=-1).squeeze(-1)
            del full
        for i,(discount,order,weight) in enumerate(candidates):
            mixture=(1-weight)*neural+weight*probabilities[discount,order]
            totals[i]+=-np.log(np.maximum(mixture[valid],np.finfo(np.float32).tiny)).astype(np.float64).sum()
    rows=[dict(ngram_discount=d,ngram_order=n,ngram_weight=w,bpb=float(loss)/math.log(2)/byte_count)
          for (d,n,w),loss in zip(candidates,totals)]
    rows.sort(key=lambda row:row['bpb'])
    selected=rows[0]
    config=checkpoint['config']|{k:v for k,v in selected.items() if k!='bpb'}|{'ngram_asset':'assets/ngram.npz'}
    result=dict(split='validation',selected=selected,candidates=rows,candidate_count=len(rows),
                seconds=time.perf_counter()-started,source_checkpoint=str(args.checkpoint),
                source_sha256=sha(args.checkpoint),train_tokens=checkpoint['train_tokens'],
                ngram_training_targets=json.loads(Path('assets/ngram_build.json').read_text())['train_targets'],
                asset_sha256=sha(Path('assets/ngram.npz')))
    (args.output_dir/'search.json').write_text(json.dumps(result,indent=2)+'\n')
    torch.save(checkpoint|dict(implementation='student',config=config,parent_checkpoint_sha256=sha(args.checkpoint),
                              selection_split='validation',ngram_training_targets=result['ngram_training_targets']),
               args.output_dir/'checkpoint.pt')
    print(json.dumps({k:v for k,v in result.items() if k!='candidates'},indent=2),flush=True)


if __name__=='__main__': main()
