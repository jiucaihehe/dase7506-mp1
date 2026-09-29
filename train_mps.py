"""Train on local Apple GPU; select checkpoints with the unchanged CPU scorer.

The benchmark's common.py and evaluate.py remain unchanged. This training-only
entry point enables the local MPS backend and adds dropout through student.py.
"""
import argparse
import json
import math
from pathlib import Path
import platform
import time
import torch
from torch.nn import functional as F
from common import PROTOCOL, load_data, make_model, sha
from evaluate import score


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/regularized256.json'))
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--steps',type=int,default=6000)
    p.add_argument('--eval-every',type=int,default=600)
    p.add_argument('--seed',type=int,default=17)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--learning-rate',type=float,default=0.001)
    args=p.parse_args()
    if args.run_dir.exists(): p.error('Use a new output directory.')
    if not torch.backends.mps.is_available(): raise SystemExit('MPS unavailable; use CPU trainer with the same config.')
    args.run_dir.mkdir(parents=True)
    started=time.perf_counter()
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.manual_seed(args.seed)
    data=load_data();config=json.loads(args.config.read_text())
    device=torch.device('mps')
    model,implementation_sha=make_model('student',config,device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=0.1)
    tokens=data['train'][0].to(device)
    generator=torch.Generator().manual_seed(args.seed)
    torch.mps.synchronize()
    preparation=time.perf_counter()-started
    history=[];validation_history=[];best_bpb=float('inf');best_step=0
    train_seconds=0.0;segment=time.perf_counter()
    for step in range(1,args.steps+1):
        starts=torch.randint(len(tokens)-257,(args.batch_size,),generator=generator).to(device)
        batch=tokens[starts[:,None]+torch.arange(257,device=device)]
        lr=args.learning_rate*min(1.,step/100)*(.1+.9*.5*(1+math.cos(math.pi*(step-1)/args.steps)))
        for group in optimizer.param_groups: group['lr']=lr
        optimizer.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(batch[:,:-1]).flatten(0,1),batch[:,1:].flatten())
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
        if step%100==0 or step==args.steps:
            torch.mps.synchronize()
            row=dict(step=step,loss=loss.item(),seconds=train_seconds+time.perf_counter()-segment)
            history.append(row);print(json.dumps(row),flush=True)
        if step%args.eval_every==0 or step==args.steps:
            torch.mps.synchronize();train_seconds+=time.perf_counter()-segment
            cpu_model,_=make_model('student',config,torch.device('cpu'))
            cpu_model.load_state_dict({k:v.detach().cpu() for k,v in model.state_dict().items()})
            validation=score(cpu_model,*data['validation'],torch.device('cpu'),'fp32')
            validation.pop('window_nll_nats');validation_history.append(dict(step=step,**validation))
            print(json.dumps({'validation':validation_history[-1]}),flush=True)
            checkpoint=dict(protocol=PROTOCOL,implementation='student',config=config,model=cpu_model.state_dict(),
                            seed=args.seed,train_tokens=step*args.batch_size*256,selected_step=step,
                            selection_split='validation',training_device='mps')
            if validation['bpb']<best_bpb:
                best_bpb=validation['bpb'];best_step=step
                torch.save(checkpoint,args.run_dir/'best.pt')
            if step==args.steps: torch.save(checkpoint,args.run_dir/'checkpoint.pt')
            del cpu_model,checkpoint
            segment=time.perf_counter()
    result=dict(protocol=PROTOCOL,implementation='student',config=config,seed=args.seed,device='mps',precision='fp32',
                parameters=sum(p.numel() for p in model.parameters()),train_tokens=args.steps*args.batch_size*256,
                train_seconds=train_seconds,preparation_seconds=preparation,process_seconds=time.perf_counter()-started,
                validation=validation_history[-1],history=history,validation_history=validation_history,
                best_validation_bpb=best_bpb,best_step=best_step,torch_version=str(torch.__version__),
                platform=platform.platform(),implementation_sha256=implementation_sha,trainer_sha256=sha(Path(__file__)),
                best_checkpoint_sha256=sha(args.run_dir/'best.pt'),
                arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    (args.run_dir/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result|{'history':[]},indent=2),flush=True)


if __name__=='__main__': main()
