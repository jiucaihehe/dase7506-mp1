"""GPT with a strictly within-window continuous cache.

Adapted conceptually from Grave, Joulin & Usunier (ICLR 2017),
https://arxiv.org/abs/1612.04426. Implementation written for this assignment.
No external weights, stored corpus, persistent evaluation state, or network I/O.
"""
import torch
from torch.nn import functional as F
from model import GPT


class RegularizedBlock(torch.nn.Module):
    """Original GPT block with dropout on attention and residual branches."""
    def __init__(self, block, dropout):
        super().__init__()
        self.heads = block.heads
        self.norm1, self.norm2 = block.norm1, block.norm2
        self.qkv, self.proj, self.mlp = block.qkv, block.proj, block.mlp
        self.dropout = dropout

    def forward(self, x):
        batch,length,width=x.shape
        q,k,v=self.qkv(self.norm1(x)).view(batch,length,3,self.heads,width//self.heads).permute(2,0,3,1,4)
        attention=F.scaled_dot_product_attention(q,k,v,is_causal=True,
                                                dropout_p=self.dropout if self.training else 0.0)
        x=x+F.dropout(self.proj(attention.transpose(1,2).reshape(batch,length,width)),
                      p=self.dropout,training=self.training)
        return x+F.dropout(self.mlp(self.norm2(x)),p=self.dropout,training=self.training)


def cache_attention(features, theta=16.0, decay=0.0):
    """At query t, key s < t predicts the already observed token x[s+1]."""
    length = features.shape[1]
    unit = F.normalize(features.float(), dim=-1)
    scores = theta * (unit @ unit.transpose(1, 2))
    positions = torch.arange(length, device=features.device)
    age = positions[:, None] - positions[None, :]
    scores = scores - decay * age.clamp_min(0)
    allowed = age > 0
    scores = scores.masked_fill(~allowed, -1e9)
    weights = F.softmax(scores, dim=-1) * allowed
    return weights


class CacheGPT(GPT):
    def __init__(self, config):
        super().__init__(config)
        dropout=float(config.get('dropout',0.0))
        if dropout:
            self.blocks=torch.nn.ModuleList([RegularizedBlock(block,dropout) for block in self.blocks])
        self.cache_weight = float(config.get('cache_weight', 0.20))
        self.cache_theta = float(config.get('cache_theta', 16.0))
        self.cache_decay = float(config.get('cache_decay', 0.0))
        self.temperature = float(config.get('temperature', 1.0))
        if not 0 <= self.cache_weight < 1:
            raise ValueError('cache_weight must be in [0, 1).')
        if self.temperature <= 0 or self.cache_theta < 0 or self.cache_decay < 0:
            raise ValueError('Invalid temperature or cache settings.')

    def predict_log_probs(self, ids):
        features = self.features(ids)
        logits = self.head(features).float() / self.temperature
        if self.cache_weight == 0 or ids.shape[1] <= 1:
            return F.log_softmax(logits, dim=-1)
        probabilities = F.softmax(logits, dim=-1)
        weights = cache_attention(features, self.cache_theta, self.cache_decay)
        # The last placeholder is always masked: s must be strictly less than t.
        next_ids = torch.cat((ids[:, 1:], ids[:, -1:]), dim=1)
        mixture = torch.full_like(probabilities[..., :1], self.cache_weight)
        mixture[:, 0] = 0.0
        probabilities = probabilities * (1.0 - mixture)
        probabilities.scatter_add_(
            2, next_ids[:, None, :].expand(-1, ids.shape[1], -1), weights * mixture
        )
        return probabilities.clamp_min(torch.finfo(torch.float32).tiny).log()


def build_model(config):
    if float(config.get('ngram_weight', 0.0)) > 0:
        return HybridGPT(config)
    return CacheGPT(config)


class HybridGPT(CacheGPT):
    """Mix the neural/cache distribution with a compact train-only KN model."""
    def __init__(self, config):
        super().__init__(config)
        from ngram import NgramModel, asset_path
        self.ngram_weight = float(config['ngram_weight'])
        self.ngram_weights_by_order = config.get('ngram_weights_by_order')
        if self.ngram_weights_by_order is not None:
            if len(self.ngram_weights_by_order)!=6 or any(not 0 <= x < 1 for x in self.ngram_weights_by_order):
                raise ValueError('Expected six interpolation weights in [0, 1).')
        if not 0 < self.ngram_weight < 1:
            raise ValueError('ngram_weight must be in (0, 1).')
        self.ngram = NgramModel(asset_path(config), config.get('ngram_discount',0.75),
                                config.get('ngram_order',5))

    def predict_log_probs(self, ids):
        neural = super().predict_log_probs(ids).exp()
        probabilities,buckets=self.ngram.probabilities(ids.detach().cpu().numpy(),return_buckets=True)
        gram = torch.from_numpy(probabilities).to(ids.device)
        weight=self.ngram_weight
        if self.ngram_weights_by_order is not None:
            import numpy as np
            weight=torch.from_numpy(np.asarray(self.ngram_weights_by_order,dtype=np.float32)[buckets]).to(ids.device)[...,None]
        mixture = (1.0-weight)*neural + weight*gram
        return mixture.clamp_min(torch.finfo(torch.float32).tiny).log()
