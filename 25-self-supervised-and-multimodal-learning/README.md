# 25 · self-supervised & multimodal learning

> learn representations without labels, and connect modalities.

labels are the expensive part of machine learning; images and text are nearly free. this chapter is about learning from the free part. you'll train networks with objectives that come from the data itself: make two distorted views of a photo agree (and disagree with other photos), predict a slowly moving copy of yourself, fill in three quarters of an image from the quarter you can see, match an image with its caption. none of these objectives is the task you care about, so the chapter starts with the question every self-supervised paper has to answer first: *how do you know the representation is any good?* and it keeps asking it. on the way you'll see a network that has never been trained let a linear classifier reach 86% on new digits, a loss value that means nothing without its batch size, a model reach the best possible loss by mapping every image to the same point, a masked autoencoder that learns more from hiding half an image than a quarter, a zero-shot classifier drop to chance because its prompt used "7" instead of "seven", and the payoff in the capstone: what pretraining without labels buys you when you have 1%, 10% or all of CIFAR-10's labels.

**you will build:** an evaluation harness (feature extraction in eval mode, a standardised linear probe, cosine-weighted kNN in memory-bounded chunks, fine-tuning with a frozen or trainable encoder); NT-Xent / InfoNCE, SimCLR's batched augmentations (random resized crops and flips through one affine resample, colour jitter, grayscale), a projection head and the SimCLR loop; the SimSiam and BYOL losses with their stop-gradients, collapse metrics (normalised std, effective rank), SimSiam with a switchable stop-gradient, BYOL with an EMA target, and a shared training loop; patchify/unpatchify in MAE order, per-sample random masking with its inverse permutation, and a masked autoencoder from your transformer blocks; the symmetric CLIP loss, a padded text transformer, mini-CLIP with a learnable temperature, zero-shot classifiers with prompt ensembles, and the CLIP loop. module: **`mlfs.ssl`** (topics 01–05), whose `linear_probe` and `knn_eval` chapter 28 reuses.
**prerequisites:** chapter 18 (`TransformerBlock` with masks, `sinusoidal_positions`, BERT's masking, ViT patches), chapter 16 (`ResNet`, transfer learning), chapter 15 (`make_loader`, `ema_update`, `CosineSchedule`, train/eval mode and BatchNorm), chapter 08 (`SoftmaxRegression`), chapter 09 (kNN). a GPU (Apple-silicon MPS or CUDA) helps from topic 04 on and is needed for the capstone's full config.
**time · scaffolding:** about 65 hours · L4 (spec-driven): each topic's lesson has **spec** boxes that fix names, arguments and conventions (MAE's patch order, the order of random draws where checks replay your loop exactly, what may and may not receive gradients); the notebooks give signatures with one-line docstrings and the checks, and the internal structure is yours to design. one hint per stage. the capstone gives you a brief and checks.
**newly unlocked library components:** after topic 02 you may use torchvision's `RandomResizedCrop`, `RandomHorizontalFlip`, `ColorJitter` and `RandomGrayscale` (or `transforms.v2`) in later work; after topic 04, library MAE/ViT implementations (e.g. `timm`) for comparison; after topic 05, pretrained CLIP models (e.g. `open_clip`) where a later chapter needs real image–text embeddings. inside `mlfs`, later chapters use only the contract in [`CONTRACTS.md`](../CONTRACTS.md).

## topics

each topic directory has a `lesson.md` to read and an `exercise.py` notebook to build in (`marimo edit exercise.py`), then a `lab.py` notebook for experiments.

| # | topic | you build | the lab's surprise |
|---|---|---|---|
| 01 | [evaluating representations](01-evaluating-representations/) | `extract_features`, `linear_probe`, `knn_eval` → `mlfs.ssl`, `fine_tune` | a never-trained CNN lets a linear probe reach 0.86 on digits 5–9 (chance 0.20), while raw pixels win kNN outright (0.94); fine-tuning a network pretrained on 0–4 is at or above the probe at every budget (at 5 labels per class the gap is within the spread over label subsets), and its advantage over training from scratch (0.10 at ≤ 20 labels per class) is gone by 500 |
| 02 | [contrastive learning](02-contrastive-learning/) | `info_nce_loss`, `SimCLRAugment`, `ProjectionHead` → `mlfs.ssl`, `train_simclr` | random embeddings score *above* chance, $\log(2B - 1)$, at small temperatures, and perfect positives can't get below 5.2 at $T = 1$ ($B = 256$); in training, $T = 0.2$ beats 0.05 and 1.0 by about a point of kNN (2 seeds), and $T = 0.05$ leaves the embeddings the least uniform |
| 03 | [non-contrastive methods](03-non-contrastive-methods/) | `simsiam_loss`, `byol_loss` → `mlfs.ssl`, `collapse_metrics`, `SimSiam`, `BYOL`, `train_ssl` | without the stop-gradient the loss reaches its best value, −1, while the std of $z$ falls from 0.125 to 0.006, and in one of two seeds kNN drops below an untrained network (0.63 vs 0.77); removing the predictor collapses $z$ too, yet the encoder below keeps most of its kNN accuracy (0.80) |
| 04 | [masked modeling](04-masked-modeling/) | `patchify`, `unpatchify`, `random_masking`, `MAE` → `mlfs.ssl`, `train_mae` | on MNIST with 4×4 patches, 50% masking gives the best features and 90% clearly the worst, with 25% and 75% close (the paper's 75% isn't universal: with 7×7 patches, 25% won, in one seed); reconstructions are blurry conditional means, and at 90% every digit becomes the same smudge |
| 05 | [multimodal learning](05-multimodal-learning/) | `clip_loss`, `MiniCLIP`, `zero_shot_predict`, `class_embeddings` → `mlfs.ssl`, `TextTransformer`, `train_clip` | on a generated captioned-digit dataset (and it is very artificial), prompts with numerals instead of words collapse to the class-0 base rate (0.090) on all five models; "a red {d}" helps red digits on every model, by 0.03 to 0.29; colour–digit combinations never seen in training get their colour right, while their digit accuracy varies widely between models (0.24 to 0.98 across runs) |

## capstone

*label efficiency*: you build class-balanced nested label subsets, a SimCLR pretraining loop with warmup and a cosine schedule (on uint8 images), a fixed-step supervised trainer with flip-and-crop augmentation, and a summary over runs; then you pretrain a half-width ResNet-18 on CIFAR-10 with no labels and compare linear probes, kNN, fine-tuning and training from scratch at 1%, 10% and 100% of the labels, with random-init baselines and the spread over label subsets. in the reference run, fine-tuning from SimCLR beats training from scratch by 12 points at 1% of the labels (0.522 ± 0.006 vs 0.397 ± 0.019, 3 label subsets) and 10 at 10%, while at 100% the two are level (0.718 vs 0.716, one run each); a run with a larger budget had from-scratch ahead at 100%, so the crossover depends on the budget. small CPU config (synthetic stand-in data): about 3½ minutes; full config: 70 minutes measured on a shared Apple-silicon machine.

see [`capstone/brief.md`](capstone/brief.md).

## checkpoint

[`checkpoint.md`](checkpoint.md): pass it to finish the chapter, or to skip it. it has 15 concept questions, three by-hand problems (weighted kNN and the probe's standardisation; NT-Xent and an ambiguous CLIP batch; MAE masking and an EMA step), and a 60-minute blind rebuild of `info_nce_loss`, `train_simclr`, the SimSiam/BYOL losses and the collapse metrics.

## notation

| symbol | meaning |
|---|---|
| $f$, $h = f(x)$ | encoder and its representation (the features you evaluate) |
| $g$, $z = g(h)$ | projection head and the embedding the loss sees |
| $q$, $p = q(z)$ | predictor (SimSiam, BYOL) and its output |
| $x_1, x_2$; $z_1, z_2$ | two augmented views of one image and their embeddings |
| $s_{ij}$, $\cos(a, b)$ | cosine similarity $a^\top b / (\|a\|\|b\|)$ |
| $T$, $s = 1/T$ | temperature; logit scale (CLIP stores $\log s$) |
| $B$, $2B - 1$ | batch size; the number of candidates per anchor in NT-Xent |
| $\operatorname{sg}(\cdot)$ | stop-gradient (`.detach()`) |
| $\xi$, $\theta$, $m$ | target and online weights, EMA momentum: $\xi \leftarrow m\xi + (1 - m)\theta$ |
| $\operatorname{erank}(Z)$ | effective rank $\exp(-\sum_kp_k\log p_k)$, $p_k = \sigma_k/\sum_j\sigma_j$ |
| $p$, $N$, $r$ | patch size, number of patches, mask ratio; $N_\text{keep} = \operatorname{round}(N(1 - r))$ |
| $u_i$, $v_j$ | normalised image and text embeddings (CLIP) |
| $k$ | number of neighbours in kNN evaluation |

## how this connects

- builds on: softmax regression (ch 08), kNN (ch 09), autoencoders and PCA (ch 11, ch 14), BatchNorm, train/eval mode, `make_loader`, `ema_update` and schedules (ch 15), ResNets and transfer learning (ch 16), word embeddings (ch 17), transformer blocks, positional encodings, BERT and ViT (ch 18), the EMA target networks of deep RL (ch 24).
- used later in: linear probes of what networks represent (ch 28 · topic 01, with this chapter's `linear_probe`); embedding search for retrieval-augmented generation (ch 27 · topic 07); text-conditioned generation and guidance (ch 22).

## core references

- Chen, Kornblith, Norouzi & Hinton (2020), *A simple framework for contrastive learning of visual representations* (SimCLR). [arXiv:2002.05709](https://arxiv.org/abs/2002.05709)
- Grill et al. (2020), *Bootstrap your own latent* (BYOL). [arXiv:2006.07733](https://arxiv.org/abs/2006.07733)
- Chen & He (2021), *Exploring simple Siamese representation learning* (SimSiam). [arXiv:2011.10566](https://arxiv.org/abs/2011.10566)
- He et al. (2022), *Masked autoencoders are scalable vision learners* (MAE). [arXiv:2111.06377](https://arxiv.org/abs/2111.06377)
- Radford et al. (2021), *Learning transferable visual models from natural language supervision* (CLIP). [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
- Balestriero et al. (2023), *A cookbook of self-supervised learning*: a practical survey of the whole field. [arXiv:2304.12210](https://arxiv.org/abs/2304.12210)
