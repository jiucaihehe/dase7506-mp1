"""Mechanism tests on genuine train-derived tables, when the asset is present."""
from pathlib import Path
import unittest
import numpy as np
import torch
from common import load_data
from student import build_model
from ngram import NgramModel


@unittest.skipUnless((Path(__file__).resolve().parents[1]/'assets/ngram.npz').exists(),
                     'Build the training-only n-gram asset before hybrid tests.')
class NgramTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(23)
        cls.ids=load_data()['train'][0][100:124].reshape(2,12)
        cls.model=build_model(dict(vocab=2048,width=32,heads=4,depth=2,context=256,
                                   cache_weight=0.1,ngram_weight=0.3,
                                   ngram_discount=0.75,ngram_order=5,dropout=0.1,
                                   ngram_weights_by_order=[0,0,0.2,0.3,0.3,0.5])).eval()

    def test_full_hybrid_normalization_and_each_prefix(self):
        with torch.no_grad():
            full=self.model.predict_log_probs(self.ids)
            self.assertTrue(torch.isfinite(full).all())
            torch.testing.assert_close(full.logsumexp(-1),torch.zeros(2,12),atol=2e-6,rtol=0)
            for length in range(1,13):
                prefix=self.model.predict_log_probs(self.ids[:,:length])
                torch.testing.assert_close(full[:,length-1],prefix[:,-1],atol=2e-6,rtol=2e-6)

    def test_history_and_other_batch_examples_do_not_leak(self):
        with torch.no_grad():
            first=self.model.predict_log_probs(self.ids[:1])
            self.model.predict_log_probs((self.ids+19)%2048)
            together=self.model.predict_log_probs(self.ids)
        torch.testing.assert_close(first,together[:1],atol=2e-6,rtol=2e-6)

    def test_ngram_pruning_preserves_normalization(self):
        orders=self.model.ngram.probabilities(self.ids.numpy(),return_orders=True)
        for probability in orders.values():
            np.testing.assert_allclose(probability.sum(-1),1,atol=2e-6,rtol=0)
            self.assertTrue(np.all(probability>0))


if __name__=='__main__': unittest.main()
