"""Compact interpolated Kneser-Ney probabilities derived only from training text.

Higher-order entries are stored as sorted, exact integer context keys and sparse
continuations. Pruned discounted mass backs off, preserving normalization.
Reference: Kneser & Ney (1995); no external code or weights are used.
"""
from pathlib import Path
import numpy as np

VOCAB = 2048


class NgramModel:
    def __init__(self, path, discount=0.75, max_order=5):
        if not 0 < discount < 1:
            raise ValueError('discount must be strictly between zero and one')
        self.max_order = max_order
        self.tables = {}
        with np.load(path, allow_pickle=False) as archive:
            self.unigram = archive['unigram'].copy()
            for order in range(2, max_order + 1):
                prefix = f'n{order}_'
                context = archive[prefix+'context'].copy()
                ptr = archive[prefix+'ptr'].copy()
                word = archive[prefix+'word'].copy()
                count = archive[prefix+'count'].astype(np.float32)
                total = archive[prefix+'total'].astype(np.float32)
                discounted = np.maximum(count - discount, 0)
                row = np.repeat(np.arange(len(context)), np.diff(ptr.astype(np.int64)))
                values = discounted / total[row]
                mass = np.bincount(row, weights=values, minlength=len(context)).astype(np.float32)
                backoff = np.maximum(1.0 - mass, 0.0)
                self.tables[order] = (context, ptr, word, values, backoff)
        context, ptr, word, values, backoff = self.tables[2]
        self.bigram = np.broadcast_to(self.unigram,(VOCAB,VOCAB)).copy()
        self.bigram[context.astype(np.int64)] *= backoff[:,None]
        row = np.repeat(context.astype(np.int64),np.diff(ptr.astype(np.int64)))
        self.bigram[row,word] += values

    def probabilities(self, ids, return_orders=False, return_buckets=False):
        """Independent causal windows in [batch,time]; outputs float32 probabilities."""
        ids = np.asarray(ids,dtype=np.int64)
        batch,length=ids.shape
        probability=self.bigram[ids].reshape(-1,VOCAB).copy()
        buckets=np.full(batch*length,2,dtype=np.int64)
        output={2:probability.reshape(batch,length,VOCAB).copy()} if return_orders else None
        context_codes=ids.astype(np.uint64).copy()
        positions=np.tile(np.arange(length),batch)
        for order in range(3,self.max_order+1):
            # Extend the context to the left without accessing future positions.
            context_length=order-1
            lag=context_length-1
            if lag < length:
                context_codes[:,lag:] |= ids[:,:length-lag].astype(np.uint64) << np.uint64(11*lag)
            keys,ptr,word,values,backoff=self.tables[order]
            query=context_codes.reshape(-1)
            row=np.searchsorted(keys,query)
            valid=(row<len(keys)) & (positions>=lag)
            row=np.minimum(row,len(keys)-1)
            valid &= keys[row]==query
            locations=np.flatnonzero(valid)
            buckets[locations]=order
            matched=row[valid]
            probability[locations] *= backoff[matched,None]
            starts=ptr[matched].astype(np.int64)
            lengths=ptr[matched+1].astype(np.int64)-starts
            repeated=np.repeat(locations,lengths)
            if len(repeated):
                relative=np.arange(len(repeated))-np.repeat(np.cumsum(lengths)-lengths,lengths)
                entries=np.repeat(starts,lengths)+relative
                probability[repeated,word[entries]] += values[entries]
            if return_orders:
                output[order]=probability.reshape(batch,length,VOCAB).copy()
        if return_orders: return output
        if return_buckets: return probability.reshape(batch,length,VOCAB),buckets.reshape(batch,length)
        return probability.reshape(batch,length,VOCAB)


def asset_path(config):
    return Path(__file__).resolve().parent/config.get('ngram_asset','assets/ngram.npz')
