"""Additional checks for the cache's most important leakage boundary."""
import unittest
import torch
from student import build_model, cache_attention
from model import GPT


class CacheMechanismTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(23)
        self.config = dict(vocab=2048, width=32, heads=4, depth=2, context=256,
                           cache_weight=0.3, cache_theta=8.0)
        self.model = build_model(self.config).eval()

    def test_each_prefix_matches_batched_causal_prediction(self):
        x = torch.randint(0, 2048, (2, 9))
        with torch.no_grad():
            full = self.model.predict_log_probs(x)
            for length in range(1, x.shape[1] + 1):
                prefix = self.model.predict_log_probs(x[:, :length])
                torch.testing.assert_close(full[:, length - 1], prefix[:, -1],
                                           atol=2e-6, rtol=2e-6)

    def test_cache_keys_are_strictly_past_and_start_is_empty(self):
        h = torch.randn(2, 7, 32)
        attention = cache_attention(h, theta=8.0, decay=0.01)
        self.assertEqual(attention.triu().count_nonzero().item(), 0)
        torch.testing.assert_close(attention[:, 1:].sum(-1), torch.ones(2, 6))
        self.assertEqual(attention[:, 0].count_nonzero().item(), 0)

    def test_zero_weight_is_exact_neural_ablation(self):
        self.model.cache_weight = 0.0
        x = torch.randint(0, 2048, (2, 12))
        with torch.no_grad():
            expected = torch.log_softmax(self.model(x).float(), dim=-1)
            actual = self.model.predict_log_probs(x)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_regularization_is_disabled_during_evaluation(self):
        regularized=build_model(self.config|{'dropout':0.1,'cache_weight':0.0}).eval()
        reference=GPT(self.config).eval()
        reference.load_state_dict(regularized.state_dict())
        x=torch.randint(0,2048,(2,12))
        with torch.no_grad():
            torch.testing.assert_close(regularized(x),reference(x),atol=0,rtol=0)


if __name__ == '__main__':
    unittest.main()
