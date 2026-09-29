# DASE7506 MP1 - Resource-Bounded Hybrid Small Language Model

Student ID: **3036707472** · GitHub: **jiucaihehe**

**Full-test CPU FP32 BPB: 1.539148051** (lower is better).
Protocol: `7506-mp1-wt2-v2`. The selected validation BPB is **1.522428**.
All model and mixture selection occurred on validation before any test scoring.

The predictor combines a small GPT, a strictly causal within-window continuous
cache, and compact train-only interpolated Kneser-Ney statistics. The cache uses
only previously observed tokens in the current independent 256-token window.
The statistical component uses only the supplied training text. No external
training data, pretrained weights, evaluation network access, persistent
cross-window state or cached validation/test answers are used.

## Evaluate the submitted checkpoint (no retraining)

Download the exact submitted repository version, including `data/`, `assets/`
and `artifacts/`. The immutable repository ZIP also serves as the complete checkpoint bundle.
`artifacts/final.pt` is the matching checkpoint. The required
statistical asset is `assets/ngram.npz`, included in this code repository. Both
files are needed; a checkpoint alone does not contain the statistical tables.

Use **Python 3.12** and run all commands from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python verify_package.py
python -m unittest discover -s tests -v
python evaluate.py --checkpoint artifacts/final.pt --device cpu --precision fp32 --threads 4 --split test
```

On Windows activate with `.venv\Scripts\Activate.ps1`. On Linux, CPU-only
PyTorch can be installed first with `python -m pip install torch==2.7.1 --index-url
https://download.pytorch.org/whl/cpu` (one command). macOS uses the default PyPI
wheel. Evaluation and training are offline after installing dependencies.

The scorer writes `artifacts/test_cpu_fp32.json`. Report its **bpb** field, not
`token_ppl` or validation BPB. It covers all **428,405 test targets** and
**1,292,013 raw UTF-8 bytes**, including the final short window. The scorer,
common benchmark code, data and tokenizer retain the course release hashes.

Final checkpoint SHA-256:

```text
132eb36bf132406709b4610473b39ff9ae7400fb40520bffcc922f8dfb495cb2
```

Asset SHA-256 and source hashes are in `artifacts/FREEZE.json`. Evaluation was
performed on an Apple M5 CPU, macOS ARM64, Python 3.12.14, PyTorch 2.7.1,
NumPy 2.5.3, tokenizers 0.21.4, FP32 and four PyTorch threads.
`environment-lock.txt` records all installed packages. Small numerical differences
across platforms can occur; the supplied weights avoid any need to retrain.

## Method, results and controls

Selected configuration:

```json
{
  "vocab": 2048,
  "width": 256,
  "heads": 4,
  "depth": 4,
  "context": 256,
  "dropout": 0.1,
  "cache_weight": 0.05,
  "temperature": 1.1,
  "cache_theta": 16.0,
  "cache_decay": 0.0,
  "ngram_discount": 0.5,
  "ngram_order": 5,
  "ngram_weight": 0.2,
  "ngram_asset": "assets/ngram.npz",
  "ngram_weights_by_order": [
    0.0,
    0.0,
    0.2,
    0.2,
    0.2,
    0.3
  ]
}
```

The cache pairs a past hidden state h[s] with the already observed token x[s+1].
Only s < t is eligible at query position t, so x[s+1] is never a future token.
Cosine similarity weights are normalized over eligible keys and aggregated by
next-token ID. The first position has no cache. Every call reconstructs all
prefix state, independently for each window and batch example.

The statistical model uses raw highest-order counts and lower-order left
continuation counts. Orders 2-4 retain all events; order 5 retains repeated raw
events. Pruned discounted probability mass backs off, preserving normalization.
Exact integer context keys avoid hash collisions. A validation-selected mixture
combines its distribution with GPT/cache; optional weights by longest matched
order adjust the mixture using only the observed input prefix.

| Frozen predictor | Validation BPB | Test BPB |
|---|---:|---:|
| Initial classroom GPT | 2.071087 | 2.101265 |
| Same baseline + cache | 1.999506 | 2.012973 |
| Selected GPT, temperature 1 | 1.595861 | 1.614482 |
| Selected GPT, same temperature | 1.588980 | 1.606800 |
| Remove statistical mixture | 1.566724 | 1.578519 |
| Remove continuous cache | 1.542126 | 1.564512 |
| Uniform cache | 1.547729 | 1.569081 |
| Constant statistical mixture | 1.523537 | 1.540046 |
| Final selected hybrid | 1.522428 | 1.539148 |

The original baseline/cache pair shares exactly the same weights and 9,830,400
training targets. Final ablations reuse the same fitted neural and statistical
components; only one inference setting changes. Statistical fitting cost is
shared and disclosed separately from gradient targets. Disabling its contribution
does not erase that preprocessing cost. Larger-model versus initial-baseline
comparisons also change training duration and cannot isolate architecture alone.
Only seed 17 was used; no between-seed significance is claimed.

`artifacts/baseline.pt` and `artifacts/baseline_cache.pt` support the initial
comparison. Other ablation checkpoints can be recreated directly from the final
weights, with **no training**:

```bash
python materialize_ablations.py
python verify_package.py
python evaluate.py --checkpoint artifacts/no_ngram.pt --device cpu --precision fp32 --split test
```

## Resource limits

| Measurement | Result | Limit |
|---|---:|---:|
| Median final CPU scoring time | 10.384 s | - |
| Median baseline CPU scoring time | 4.232 s | - |
| CPU time ratio | 2.454x | 5x |
| Peak whole-process evaluation RAM | 2.197 GiB | 4 GiB |
| Uncompressed inference assets | 60.219 MiB | 64 MiB |

Times are medians of three fresh processes per predictor, alternating baseline
and final on the same CPU with four threads and no concurrent training. The
unmodified scorer's seconds exclude loading. Peak RAM is whole-process
`resource.getrusage` maximum RSS, including imports/loading. NPZ contents are
counted **uncompressed**, not by downloaded ZIP size. The final checkpoint,
statistical asset and predictor modules are counted. The fixed benchmark,
framework installation and optional ablation/experiment files are not required
inference assets of the final predictor and are excluded.

```bash
python measure_eval.py --checkpoint artifacts/final.pt --device cpu --precision fp32 --threads 4 --split test --output my-evaluation.json
```

The sidecar `my-evaluation.resources.json` contains peak RAM. Resource ratios
can differ on the instructor's reference Xeon; this report gives measured results
on the available Mac and complete reproduction commands.

## Complete experiment evidence

`report.pdf` is the English report (under ten pages). `experiment_evidence.zip`
contains the complete `runs/` metrics/search results, `logs/` console records,
and `results/` official scores, per-window losses, resource measurements and
comparison tables. It contains no model-answer cache. Extract it for inspection:

```bash
python -m zipfile -e experiment_evidence.zip .
```

`results/summary.json` collects all measured evidence. Source/configuration hashes
were frozen in `artifacts/FREEZE.json` before testing. Test losses are audit outputs
from the official scorer and are never read by the predictor.

## Reproduce training and validation selection

Use a fresh working directory or move existing generated `runs/`, `artifacts/`
and `results/` aside. Training scripts refuse nonempty output directories; the
freeze script refuses to overwrite a bundle. The selected checkpoint was produced
by the following recorded sequence (all commands from repository root):

```bash
python train.py --implementation model --device cpu --precision fp32 --threads 4 --seed 17 --eval-every 300 --run-dir runs/baseline-s17
python tune_cache.py --checkpoint runs/baseline-s17/checkpoint.pt --output-dir runs/baseline-cache-s17
python train_experiment.py --implementation student --config configs/wide.json --device cpu --precision fp32 --threads 4 --seed 17 --steps 3600 --eval-every 600 --run-dir runs/wide-s17
python tune_cache.py --checkpoint runs/wide-s17/best.pt --output-dir runs/wide-cache-s17
python build_ngram.py
python tune_hybrid.py --checkpoint runs/wide-cache-s17/checkpoint.pt --output-dir runs/wide-hybrid-s17
python tune_gate.py --checkpoint runs/wide-hybrid-s17/checkpoint.pt --output-dir runs/wide-gated-s17
python train_continue.py --implementation student --config configs/wide.json --init-checkpoint runs/wide-s17/best.pt --learning-rate 0.0002 --device cpu --precision fp32 --threads 4 --seed 17 --steps 2400 --eval-every 400 --run-dir runs/continued-s17
python tune_cache.py --checkpoint runs/continued-s17/best.pt --output-dir runs/continued-cache-s17
python tune_hybrid.py --checkpoint runs/continued-cache-s17/checkpoint.pt --output-dir runs/continued-hybrid-s17
python tune_gate.py --checkpoint runs/continued-hybrid-s17/checkpoint.pt --output-dir runs/continued-gated-s17
```

The next experiment uses an Apple GPU with the MPS backend. Its validation and
all final scoring use CPU FP32. This training requirement does not apply to direct
evaluation of the provided checkpoint:

```bash
python train_mps.py --steps 20 --eval-every 20 --run-dir runs/mps-smoke
python train_mps.py --steps 6000 --eval-every 600 --run-dir runs/regularized256-s17
python tune_cache.py --checkpoint runs/regularized256-s17/best.pt --output-dir runs/regularized256-cache-s17
python tune_hybrid.py --checkpoint runs/regularized256-cache-s17/checkpoint.pt --output-dir runs/regularized256-hybrid-s17
python tune_gate.py --checkpoint runs/regularized256-hybrid-s17/checkpoint.pt --output-dir runs/regularized256-gated-s17
python freeze_submission.py
python benchmark_frozen.py
python score_ablation_validation.py
python analyze_validation.py
python collect_results.py
```

`freeze_submission.py` selects the lowest full validation BPB among saved searches,
then fixes the predictor and ablations before testing. Re-training on different
hardware need not produce bitwise identical weights. For CPU-only experimentation,
`train_experiment.py` supports `configs/regularized256.json` with 6,000 steps; that
is an alternate training backend, not a claim of identical MPS-trained weights.

Each cache search evaluates 153 deduplicated settings; each statistical search
55 settings. Gating chooses four weights independently from nine values each.
These settings are selected on validation, never on test. Full candidate results
and costs are retained in the evidence archive.

| Training run | Device | New gradient targets | Cumulative targets | Train seconds |
|---|---|---:|---:|---:|
| baseline-s17 | cpu | 9,830,400 | 9,830,400 | 191.19 |
| continued-s17 | cpu | 19,660,800 | 49,152,000 | 1072.83 |
| mps-smoke | mps | 163,840 | 163,840 | 14.28 |
| regularized256-s17 | mps | 49,152,000 | 49,152,000 | 1676.06 |
| wide-s17 | cpu | 29,491,200 | 29,491,200 | 1236.83 |

Total new gradient targets: **108,298,240**. Logged optimization
time: **4191.19 seconds**, excluding measured validation.
Validation search time: **228.09 seconds**. The retained
statistical model scans **3,613,342** targets in
**13.07 seconds**. An initial, more pruned asset build
cost **13.27 seconds** and was replaced based on
size before statistical-model scoring. The 20-step GPU smoke run is disclosed in
costs and was never a final candidate. Ancestry is explicitly retained.

To regenerate the English PDF after extracting evidence, install `reportlab`
separately and run `python build_report.py`. It reads measured JSON and is not
needed for evaluation. To repeat the resource audit, materialize ablations, move
the existing `results/` aside, and run `python benchmark_frozen.py`.

## Reuse, attribution and substantive AI disclosure

The classroom GPT, original trainer, fixed scorer, benchmark, tokenizer and
original contract tests are reused from the DASE7506 starter. Their fixed hashes
are verified. The additional trainers adapt its optimization recipe. Continuous
cache inspiration: Grave, Joulin & Usunier,
[Improving Neural Language Models with a Continuous Cache](https://arxiv.org/abs/1612.04426),
ICLR 2017. Statistical smoothing: Kneser & Ney,
[Improved Backing-Off for M-Gram Language Modeling](https://www-i6.informatik.rwth-aachen.de/publications/download/951/Kneser-ICASSP-1995.pdf),
ICASSP 1995; Chen & Goodman, An Empirical Study of Smoothing Techniques for Language
Modeling, Computer Speech and Language 13(4), 1999. This work does not claim to
invent neural caching or Kneser-Ney smoothing.

**OpenAI Codex provided substantive AI assistance** with planning, implementation,
experiment execution, testing, validation selection, analysis, report writing,
reproduction documentation and submission preparation. The student is responsible
for reviewing, understanding and explaining the implementation. This disclosure
does not assert that independent student review has already occurred.

WikiText-2 was introduced by Merity et al.,
[Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843).
The text is by Wikipedia contributors. Retain the supplied
[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) and
[GNU Free Documentation License](https://www.gnu.org/licenses/fdl-1.3.html)
notices. Revision and hashes are in `data/manifest.json`. These dataset licenses
do not relicense the classroom source. `STARTER_README.md` and
`ASSIGNMENT_GUIDE.md` preserve the supplied instructions.
