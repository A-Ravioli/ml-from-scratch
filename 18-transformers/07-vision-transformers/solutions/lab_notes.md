# vision transformers · lab notes

numbers below are from the reference solution on an Apple-silicon laptop GPU (`mps`, shared with other jobs: 263 s for the whole lab, 103 s more for the optional 20,000-image runs), real CIFAR-10, seed 0, one run per cell. 300 steps of batches of 128 is a deliberately short budget, chosen to keep the lab to a few minutes (chapter 16's capstone trained a full-width ResNet-18 for about 7,800 steps to 92.9%), so every number here is far from converged; the comparison is between models given the same small budget. an earlier version of this lab used 800 steps and found the same pattern with higher accuracies (ResNet 0.485 / 0.596 / 0.625 and ViT 0.378 / 0.520 / 0.543 at 1,000 / 5,000 / 20,000 images; without positions 0.496 at 20,000).

## experiment 1: ViT against ResNet, by amount of data

**what people predict.** having read the ViT paper's headline, some expect the ViT to win or tie. others expect it to lose badly at 1,000 images and catch up by 5,000.

**what happens.** test accuracy on 2,000 images (chance 0.10), models of 0.70M (ResNet) and 0.81M (ViT) parameters:

| training images | ResNet | ViT | gap |
|---|---|---|---|
| 1,000 | 0.452 | 0.364 | 8.8 points |
| 5,000 | 0.500 | 0.422 | 7.8 points |
| 20,000 (optional button) | 0.505 | 0.409 | 9.6 points |

the ResNet wins at every size, by 8–10 points, and the gap doesn't close. from 5,000 to 20,000 images neither model improves much: both see only 38,400 augmented images in 300 steps, so at 20,000 each image is seen about twice and the models are step-limited, not data-limited.

**why.** a convolution assumes locality (a 3×3 filter combines neighbouring pixels) and translation equivariance (the same filter everywhere): both true of natural images, and both free. the ViT has to learn from examples that neighbouring patches matter more than distant ones and that a wheel is a wheel wherever it is. in a short budget it learns less of either, so it trails at every data size here. the priors are worth less as data grows, which is the trend Dosovitskiy et al. report (their figure 3: ViTs lose to ResNets when pretrained on ImageNet's 1.3M images, tie around 14M, and win at 300M); at laptop scale and a few hundred steps you don't reach the regime where that trend shows, and nothing here would let you extrapolate to the crossover. the recipe is shared, and ViTs typically need stronger augmentation and regularization and longer training than this; a tuned ViT recipe (DeiT, Touvron et al., 2021) would close part of the gap.

## experiment 2: does the ViT use its positions?

**what people predict.** a large drop, often to near chance: "without positions, it can't see shapes".

**what happens.** without position embeddings, the ViT trained on 5,000 images reaches 0.406, against 0.422 with them: it loses 1.6 points (in the 800-step version, 4.7 points at 20,000 images).

**why.** with no positions the ViT is a set model (lesson, on paper 3): it sees 64 patches of 4×4 pixels with no arrangement. but a CIFAR-10 class is largely identifiable from local evidence: colours (green and brown for frogs and deer, blue for ships and planes), textures (fur, feathers, metal) and small parts. a bag of local features is a strong classifier, which is also why "bag of visual words" models worked before deep learning. what the positions add is arrangement: sky above ground, the outline of a car. a model trained this briefly has learned little arrangement to lose, which is why the drop is smaller than in the longer run. Dosovitskiy et al. found the same on ImageNet (appendix D.4: no position embedding costs about 3 points, and 1-D, 2-D and relative encodings perform about equally); the classes that suffer most are those defined by shape rather than texture (cats vs dogs, automobiles vs trucks).

## experiment 3: how far does each head look?

**what people predict.** that training makes early heads local, like convolutions, because that's what the ViT paper's famous figure shows.

**what happens.** untrained, every head's mean attention distance is about 16.2–16.8 pixels (read from the figure), essentially the 16.5 of uniform attention over the 8×8 grid. trained (5,000 images, 300 steps), the heads are at 15.7–17.5 pixels: layer 1's four heads at 15.7–16.2, the deeper layers spread slightly in both directions. no head has become local; a 3×3 convolution's neighbours are 4 pixels away.

**why.** at initialization the queries and keys are small and nearly uncorrelated with position, so the softmax is nearly flat and every head averages the whole image. learning to attend to neighbours requires the position embeddings to become similar for nearby patches and the query–key projections to exploit that, a slow, data-hungry process. in 300 steps this ViT hasn't done it: it classifies with global averages of patch features, close to the bag-of-patches model of experiment 2 (and that's why removing positions cost so little). in the ViT paper, trained on 300 million images, some first-layer heads attend locally (20–40 pixels on 224×224 images, comparable to a convolution's early receptive field) while others stay global from the very first layer; that locality is learned, and learning it is exactly the data cost experiment 1 measures. the convolution has it from step 0.
