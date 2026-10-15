# modern convnets & transfer · lab notes

numbers are for real CIFAR-10 (downsampled to $16 \times 16$ for experiments 2–3), measured with the reference solution on an Apple M2 laptop CPU while other jobs were running (load average about 10; the lab took 2.5 minutes). experiment 1's times depend on the machine and its load, so compare the ratios: three runs gave 1.2×, 1.4× and 2.0× speedups for the depthwise block, and 0.2–0.5× for the ConvNeXt block. experiment 2 uses one draw of the labelled subset per budget, so its accuracies carry about ±0.011 from the test set and a few points from the draw at $n = 5$. experiment 3 was repeated with two more seeds (pretraining, subset and head); its conclusions held every time. offline (the synthetic stand-in that `verify.py` uses) the lab runs end to end but says nothing about CIFAR.

## experiment 1: fewer multiplications, faster network?

**what people usually predict:** 8× fewer MACs, about 8× faster; and the ConvNeXt block, with about as many MACs as the standard conv, about as fast.

**what happens:** at $C = 128$ on $32 \times 32$ maps, batch 32:

| block | parameters | MACs per image | fewer MACs than standard | time | faster than standard |
|---|---|---|---|---|---|
| standard 3x3 conv (+BN, ReLU) | 147,712 | 151.0 M | 1.0× | 91.5 ms | 1.0× |
| depthwise separable | 18,048 | 18.0 M | 8.4× | 44.8 ms | 2.0× |
| ConvNeXt block | 138,496 | 140.6 M | 1.1× | 172.1 ms | 0.5× |

the depthwise separable block does 8.4× fewer MACs but runs only 1.2–2× faster (across three runs). the ConvNeXt block does slightly fewer MACs than the standard conv and runs **2–5 times more slowly**.

**why:** MACs count arithmetic; time on a CPU (or GPU) is often set by memory traffic. a standard convolution uses each input value $9C$ times (high arithmetic intensity) and runs as a well-optimized matrix product near the hardware's peak. a depthwise convolution uses each value only 9 times (on paper 2: about 1 MAC per byte), so it waits on memory, and its kernels are less optimized; the pointwise half is a fast matrix product, but the depthwise half and the two batch norms (each a full pass over the activations) eat most of the saving. the ConvNeXt block adds more memory-bound work on top: two permutes of the whole tensor, a layer norm, a GELU over a 4× wider tensor, and the layer-scale multiply; its $7 \times 7$ depthwise conv is cheap in MACs but not in time. MobileNets still win on phones because mobile accelerators and runtimes are built for these operations (fused depthwise kernels, int8 arithmetic) and because parameter count and energy matter there; ConvNeXt's authors accepted the cost because at ImageNet scale on GPUs, accuracy per FLOP and the large receptive field were worth it. the lesson for your own work: profile the time, not just the MACs.

## experiment 2: linear probe, fine-tune, or start from scratch?

**what people usually predict:** with 5 images per class the linear probe wins (fine-tuning overfits); with 500 fine-tuning wins: a crossover.

**what happens:** target test accuracy (pretrained for 2 epochs on airplane/automobile/bird/cat/deer, source test accuracy 0.591; 60 optimizer steps per run):

| images per class | linear probe | fine-tune (backbone lr × 0.1) | fine-tune, one lr | from scratch |
|---|---|---|---|---|
| 5 | 0.519 | **0.523** | 0.497 | 0.386 |
| 50 | 0.550 | 0.592 | **0.611** | 0.508 |
| 500 | 0.558 | 0.620 | **0.676** | 0.638 |

with 5 images per class the probe and the gentle fine-tune tie (0.519 vs 0.523, within noise), and fine-tuning with one learning rate everywhere is worse than both: that's the small-data end of the expected crossover, but it's a tie rather than a clear probe win. from 50 images per class on, fine-tuning wins clearly, and the one-lr version overtakes the gentle one. the bigger crossover is with **training from scratch**: pretraining is worth 13 points at 5 images per class and 10 at 50, and only 4 at 500, where from scratch (0.638) already beats the gentle fine-tune. the linear probe saturates around 0.55: more labels don't help a classifier on fixed features. (an earlier version of this lab with 3 pretraining epochs and 100 steps gave the same picture, with fine-tuning ahead of the probe by 3 points even at 5 images per class.)

**why:** with 5 images per class the probe fits 645 parameters on 25 labels; fine-tuning fits 308,181 but starts from the pretrained solution and takes only 60 small steps, which keeps it near that solution: early stopping by budget acts as regularization (on paper 4). a full learning rate on the backbone moves it too far for 25 labels, which is the regime where a smaller backbone learning rate pays off. the probe's ceiling is the quality of the frozen features: they were learned to separate airplanes from cats, not dogs from horses, and only fine-tuning can change them. as labels grow, the target data can teach the network what pretraining did, so from-scratch training catches up (here by 500 per class, at $16 \times 16$ with a small network and a short pretraining; with a stronger pretrained model and a more different target, pretraining keeps helping much longer). a clear win for linear probing appears in other settings: with very few labels and a large backbone, or when you care about inputs unlike the training data, where fine-tuning distorts features that a probe keeps (Kumar et al., 2022).

## experiment 3: what does adapting a backbone cost the old task?

**what people usually predict:** the probes leave the old task untouched; fine-tuning on 500 images costs it a little.

**what happens:** after adapting to the dusk-domain target task (100 images per class, the same batches for all three), the original source head is put back on each backbone and tested on the original (daylight) source task:

| method | backbone parameters unchanged? | target (dusk) accuracy | source accuracy before → after |
|---|---|---|---|
| your `fine_tune(mode='linear')` | yes | 0.561 | 0.591 → **0.591** |
| hand-written probe in `train()` mode | yes | 0.575 | 0.591 → **0.412** |
| your `fine_tune(mode='full')` | no | **0.645** | 0.591 → **0.408** |

with seeds 1 and 2: your probe 0.648 → 0.648 and 0.658 → 0.658; the train-mode probe → 0.500 and 0.520; full fine-tuning → 0.522 and 0.486. your linear probe leaves the backbone exactly as it was. the hand-written probe changed no weight and saw exactly the same batches, yet it wrecked the source task about as badly as full fine-tuning (14–18 points), while gaining a point or two on the target.

**why:** the hand-written probe ran the backbone in training mode, so every batch-norm layer replaced its running statistics with estimates from the dusk images (on paper 3: after 320 steps at momentum 0.1, nothing of the originals remains). dusk images have a much lower mean and half the spread, so afterwards the backbone normalizes *daylight* images with dusk statistics: every activation is shifted and scaled wrongly, and the old head receives features it was never trained on. the same change helps the target a little, because normalizing dusk images with dusk statistics undoes part of the domain shift (a known adaptation trick, "AdaBN"). full fine-tuning gets the most target accuracy, because it adapts the weights as well as the statistics, and forgets the old task (**catastrophic forgetting**) by both routes. the lesson: `requires_grad=False` is not "frozen". a linear probe in eval mode is the only one of the three that is safe to share: one frozen backbone can serve many tasks, each with its own 645-parameter head.
