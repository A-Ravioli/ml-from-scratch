# 25 · capstone: label efficiency

## goal

the case for self-supervised learning is economic: labels are expensive, images are cheap. so the question that matters is not "how good is my SimCLR model?" but **"how many labels does pretraining save me?"** in this capstone you answer it on CIFAR-10. you pretrain a ResNet with SimCLR on all 50,000 training images *without their labels*, then evaluate it with 1%, 10% and 100% of the labels (500, 5,000 and 50,000 images), using every protocol from topic 01: a linear probe and kNN on the frozen features, and fine-tuning the whole network. the baseline is the honest one: the same architecture trained from scratch on the same labelled subset, with the same number of steps. random-init baselines for the frozen protocols tell you how much of each number is the architecture rather than the pretraining.

this pulls in the whole chapter: the evaluation harness (topic 01), InfoNCE, the augmentations and the projection head (topic 02), and the habits of every lab: report spread, compare with a baseline, measure collapse-free training rather than assume it. topics 03–05 supply the stretch goals.

## data

CIFAR-10 through `mlfs.datasets.cifar10()` (cached under `~/.cache/mlfs`): 50,000 training and 10,000 test images, 32×32×3, kept as uint8 (NCHW) in memory and converted to floats one batch at a time. with `MLFS_OFFLINE=1` (tests and `verify.py`) the loader returns a small synthetic stand-in, on which even a random encoder scores near 100%: the small config checks that the pipeline runs, nothing more.

## configurations

| config | where | sizes | runtime |
|---|---|---|---|
| **small** | `MLFS_OFFLINE=1` (tests, verify) | ResNet-18 layout at base width 8; 1 pretraining epoch; 10 / 20 / 30 supervised steps; 2 label subsets | about 3½ minutes on a laptop CPU |
| **full** (your write-up) | default | ResNet-18 layout at base width 32 (half the standard width, feature dim 256); SimCLR for 12 epochs (batch 256, AdamW lr $10^{-3}$, 1 warmup epoch then cosine, $T = 0.2$, strength 0.5); fine-tuning (SGD lr 0.01) and from scratch (SGD lr 0.05) for 150 / 300 / 1,000 steps of batch 128 at 1% / 10% / 100%; 3 label subsets at 1% and 10% | measured once: 70 minutes on an Apple-silicon GPU (MPS): 19 min pretraining, 50 min evaluation. the machine was shared with other jobs (load average 86 when the run started, 6 when it ended), so treat this as an upper bound rather than a typical time |

the notebook picks the config from `datasets.is_offline()`.

## deliverables

1. `capstone.py` with the required functions implemented and their checks green:
   - `label_subset(labels, fraction, *, seed=0)` → sorted int64 indices: for each class, in sorted order, the first $\max(1, \operatorname{round}(\text{fraction} \cdot n_c))$ entries of `rng.permutation(that class's indices)`, with one `rng = np.random.default_rng(seed)`. subsets for one seed are therefore class-balanced and **nested** (the 1% subset is inside the 10% one). `ValueError` unless $0 < \text{fraction} \le 1$.
   - `pretrain_simclr(encoder, feature_dim, images, *, epochs=1, batch_size=256, lr=1e-3, temperature=0.2, strength=0.5, warmup_epochs=0, weight_decay=1e-6, seed=0, device='cpu')` → `(head, history)`. `torch.manual_seed(seed)` and `rng = np.random.default_rng(seed)` first; then `ProjectionHead(feature_dim)`, `SimCLRAugment(H, strength=strength)`, `AdamW` over encoder and head (`lr`, `weight_decay`), and `CosineSchedule(optimizer, epochs * steps_per_epoch, warmup_steps=warmup_epochs * steps_per_epoch)` stepped after every optimizer step. each epoch: both in train mode, a fresh permutation, `N // bs` full batches (`bs = min(batch_size, N)`); uint8 batches are divided by 255 (floats are used as they are); the loss is `info_nce_loss` of view 1 then view 2. `history` has `'loss'` and `'lr'` (the rate each step used).
   - `train_classifier(encoder, feature_dim, n_classes, images, labels, *, steps, batch_size=128, lr=0.05, weight_decay=5e-4, augment=True, seed=0, device='cpu')` → `nn.Sequential(encoder, nn.Linear(feature_dim, n_classes))` in eval mode, after **exactly** `steps` SGD steps (momentum 0.9, `weight_decay`, `CosineSchedule(optimizer, steps)`). `torch.manual_seed(seed)` before creating the head; batches cycle through fresh permutations from `np.random.default_rng(seed)`, `N // bs` full batches per pass, stopping mid-pass when the budget runs out. with `augment`, each batch goes through the provided `flip_crop` (flip, pad 4, random crop).
   - `accuracy(model, images, labels, *, batch_size=512, device='cpu')` → float, in eval mode under `no_grad`, the model's modes restored.
   - `summarize(records)` → `{(method, fraction): {'mean', 'std', 'n'}}` from a list of `{'method', 'fraction', 'accuracy'}` dicts; `std` with `ddof=1`, `nan` for a single run.
2. the full-config study in the notebook (it runs on your functions).
3. the write-up at the end of the notebook, answering its five prompts, including one ablation whose hypothesis you write down *before* running it.

## requirements

| check | threshold |
|---|---|
| tests in `test_capstone.py` | all pass (`python -m pytest test_capstone.py`) |
| `label_subset`, `pretrain_simclr`, `train_classifier` | exact against replays of the specified draws and loops |
| learning | SimCLR loss falls by > 0.3 on toy data (median of 3 seeds); `train_classifier` reaches > 0.9 test accuracy on a toy task (median of 3 seeds) |
| the study | a SimCLR row and a from-scratch row at all three fractions, with the spread over label subsets |
| the write-up | every number with its spread and number of runs; the ablation's hypothesis stated before its result |

for reference, the reference solution's full config measured (test accuracy on the 10,000 test images; mean ± std over 3 label subsets at 1% and 10%, a single run at 100%; one pretraining run, seed 0):

| method | 1% labels | 10% labels | 100% labels |
|---|---|---|---|
| fine-tune SimCLR | 0.5217 ± 0.0061 | 0.6281 ± 0.0006 | 0.7177 |
| from scratch | 0.3970 ± 0.0185 | 0.5257 ± 0.0093 | 0.7155 |
| linear probe (SimCLR) | 0.4164 ± 0.0110 | 0.5289 ± 0.0010 | 0.5543 |
| linear probe (random init) | 0.2842 ± 0.0102 | 0.3898 ± 0.0038 | 0.4242 |
| kNN (SimCLR) | 0.3734 ± 0.0067 | 0.4493 ± 0.0063 | 0.5145 |
| kNN (random init) | 0.2546 ± 0.0028 | 0.3146 ± 0.0026 | 0.3817 |

(raw-pixel kNN, for scale: 0.244, 0.299 and 0.364.) SimCLR pretraining is worth about 12 points of accuracy at 1% of the labels and 10 at 10%, far more than the spread over label subsets. at 100% the two are level (0.718 against 0.716, a single run each). an earlier run with a larger budget (15 pretraining epochs; 400 and 1,500 steps at 10% and 100%) gave the same picture at 1% and 10% (0.541 ± 0.007 vs 0.397 ± 0.018; 0.656 vs 0.570) and had from-scratch *ahead* at 100% (0.783 vs 0.745). so the results suggest that the advantage of pretraining shrinks to nothing somewhere between 10% and 100% of the labels at this budget; where exactly, and whether it reverses, depends on the budget. two confounds to keep in mind: the 100% runs use under 3 epochs of supervised training, which favours the pretrained start, and the learning rates (0.01 for fine-tuning, 0.05 from scratch) were chosen once and never tuned, so either side could be handicapped. frozen SimCLR features beat frozen random features by 12–14 points under both protocols. the spread shown is only over label subsets: one pretraining seed and one fine-tuning seed per subset, so it understates the real uncertainty; a second pretraining run is a good first stretch goal.

## stretch goals

- **a second source of variance.** repeat pretraining with seed 1 (or the 100% runs with three seeds) and see whether the 100% result (level, in the reference run) holds up.
- **SimSiam or BYOL instead of SimCLR** with your topic 03 code, monitoring the std of $z$ throughout; or **an MAE** with a ViT encoder (topic 04): MAE is known for weak probes and strong fine-tuning, does that show at 1%?
- **longer pretraining.** 12 epochs is very short for SimCLR (the paper used hundreds). plot the 1% fine-tuning accuracy against pretraining epochs (5, 12, 30) and extrapolate cautiously.
- **semi-supervised baselines.** pseudo-labelling or FixMatch-style consistency at 1% (Sohn et al., 2020), which uses the unlabelled images *during* supervised training rather than before it.
- **zero-shot with captions.** use topic 05's mini-CLIP recipe with generated captions of CIFAR-10 classes, and compare its zero-shot accuracy with the 1% row.

## self-assessment

grade your write-up with [`rubric.md`](rubric.md).
