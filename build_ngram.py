"""Build an exact-key, pruned interpolated Kneser-Ney model from train ONLY."""
import json
from pathlib import Path
import time
import zipfile
import numpy as np
from common import load_data, sha


def main():
    started=time.perf_counter()
    tokens=load_data()['train'][0].numpy().astype(np.uint64)
    raw={}
    for order in range(2,6):
        codes=np.zeros(len(tokens)-order+1,dtype=np.uint64)
        for offset in range(order):
            codes=(codes << np.uint64(11)) | tokens[offset:offset+len(codes)]
        key,count=np.unique(codes,return_counts=True)
        raw[order]=(key,count.astype(np.uint32))
        print(json.dumps(dict(order=order,unique_ngrams=len(key))),flush=True)
    uni=np.bincount((raw[2][0] % 2048).astype(np.int64),minlength=2048).astype(np.float64)+0.01
    arrays={'unigram':(uni/uni.sum()).astype(np.float32)}
    details=[]
    for order in range(2,6):
        if order==5:
            keys,counts=raw[order]
        else:
            suffix=raw[order+1][0] & np.uint64((1 << (11*order))-1)
            keys,counts=np.unique(suffix,return_counts=True)
            counts=counts.astype(np.uint32)
        contexts=keys >> np.uint64(11)
        all_contexts,inverse=np.unique(contexts,return_inverse=True)
        totals=np.bincount(inverse,weights=counts).astype(np.uint32)
        # Lower orders retain all events; high orders retain repeated raw events.
        minimum=1 if order<=4 else 2
        raw_keys,raw_counts=raw[order]
        keep=raw_counts[np.searchsorted(raw_keys,keys)]>=minimum
        keys,counts,contexts=keys[keep],counts[keep],contexts[keep]
        retained_contexts,starts=np.unique(contexts,return_index=True)
        prefix=f'n{order}_'
        arrays[prefix+'context']=retained_contexts
        arrays[prefix+'ptr']=np.r_[starts,len(keys)].astype(np.uint32)
        arrays[prefix+'word']=(keys & np.uint64(2047)).astype(np.uint16)
        if counts.max() > 65535:
            raise ValueError('Count exceeds exact uint16 storage; increase dtype explicitly.')
        arrays[prefix+'count']=counts.astype(np.uint16)
        arrays[prefix+'total']=totals[np.searchsorted(all_contexts,retained_contexts)]
        details.append(dict(order=order,entries=len(keys),contexts=len(retained_contexts),raw_minimum=minimum))
    out=Path('assets');out.mkdir(exist_ok=True)
    path=out/'ngram.npz'
    np.savez_compressed(path,**arrays)
    with zipfile.ZipFile(path) as archive:
        uncompressed_bytes=sum(info.file_size for info in archive.infolist())
    result=dict(source_split='train',train_targets=len(tokens)-1,seconds=time.perf_counter()-started,
                bytes=path.stat().st_size,uncompressed_bytes=uncompressed_bytes,asset_sha256=sha(path),orders=details,
                smoothing='Interpolated Kneser-Ney with one absolute discount; unigram continuation counts + 0.01 pseudocount',
                pruning='Order 5 retains raw-count >=2 events; orders 2-4 retain all events. Removed discounted mass backs off to the lower order.',
                train_text_sha256=sha(Path('data/wikitext_train.txt')))
    (out/'ngram_build.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
