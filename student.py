"""GPT with a strictly within-window continuous cache.

Adapted conceptually from Grave, Joulin & Usunier (ICLR 2017),
https://arxiv.org/abs/1612.04426. Implementation written for this assignment.
No external weights, stored corpus, persistent evaluation state, or network I/O.
"""
import torch
from torch.nn import functional as F
from model import GPT


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
    return CacheGPT(config)
