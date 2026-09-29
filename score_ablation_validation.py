"""Post-freeze validation reporting only; does not select or change any model."""
import json
from pathlib import Path
import torch
from common import load_data, make_model, setup
from evaluate import score


def main():
    setup('cpu','fp32',4)
    data=load_data()['validation']
    frozen=json.loads(Path('artifacts/FREEZE.json').read_text())
    results={}
    for name in ['baseline','final']+frozen['ablation_names']:
        checkpoint=torch.load(f'artifacts/{name}.pt',map_location='cpu',weights_only=True)
        model,_=make_model(checkpoint['implementation'],checkpoint['config'],torch.device('cpu'))
        model.load_state_dict(checkpoint['model'])
        result=score(model,*data,torch.device('cpu'),'fp32');result.pop('window_nll_nats')
        results[name]=result
        print(json.dumps(dict(name=name,bpb=result['bpb'])),flush=True)
        del model,checkpoint
    Path('results/validation_ablations.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
