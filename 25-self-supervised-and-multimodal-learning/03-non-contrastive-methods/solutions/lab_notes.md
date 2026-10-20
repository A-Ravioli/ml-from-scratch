# non-contrastive methods · lab notes

numbers are from the reference solution with the lab's defaults (real MNIST, laptop CPU, 4 threads). the lab took about three minutes on a busy machine. yours will differ in detail.

## experiment 1: remove the stop-gradient

**what people usually predict:** without the stop-gradient training is "less stable" or "a bit worse"; few people predict that the loss gets *better*.

**what happens:** (seed 0 / seed 1; the random untrained encoder scores kNN 0.769 and its features have effective rank 6.3)

| model | final loss | final std of z | kNN accuracy | effective rank of h (of 64) |
|---|---|---|---|---|
| SimSiam (stop-gradient) | −0.778 / −0.768 | 0.121 / 0.123 | 0.836 / 0.829 | 14.7 / 15.1 |
| no stop-gradient | −0.998 / −0.997 | 0.006 / 0.025 | 0.630 / 0.833 | 5.4 / 13.9 |

without the stop-gradient, both seeds drive the loss to its global minimum, $-1$, within about 80 steps, and the std of $z$ falls steadily from 0.125 towards 0 (the figure): complete collapse of the projections. what that does to the *encoder* depends on the seed: in seed 0 its features end at effective rank 5.4 and kNN 0.630, well below the never-trained network; in seed 1 the encoder is still at 0.833 when training stops, although its $z$ is nearly constant. with the stop-gradient, the std stays at 0.12 (near $1/\sqrt{64} = 0.125$), the loss plateaus around $-0.77$, and both seeds improve kNN by 6–7 points over the random encoder while more than doubling the effective rank of the features.

**why:** a constant output makes every pair of views agree, so it's the loss's global minimum (on paper 1), and without the stop-gradient the gradient pulls both branches together until nothing about the input survives. the loss curve looks like excellent training. the projections collapse first; the damage then propagates into the encoder as the encoder stops receiving useful gradient, at a speed that varies from run to run, which is why two seeds tell such different stories about kNN at step 156. that's also why you need both monitors: the std of $z$ catches the method failing early and cheaply, and the encoder's effective rank (or a kNN probe) tells you how much of the representation has been lost. with the stop-gradient, each step moves the predictions towards targets held fixed for that step (the alternating-optimisation view), so the constant solution is no longer a direction the gradient can take for free.

## experiment 2: is the stop-gradient the only ingredient?

**what people usually predict:** the predictor is a detail (it's "just an MLP"); BYOL's momentum target is essential, since it's BYOL's headline idea.

**what happens:** one seed (seed 0) each; same settings as experiment 1. the BYOL rows were measured by running the same configuration outside the notebook.

| model | final loss | final std of z | kNN accuracy | effective rank of h |
|---|---|---|---|---|
| SimSiam (from experiment 1) | −0.778 | 0.121 | 0.836 | 14.7 |
| no predictor | −0.999 | 0.014 | 0.797 | 13.2 |
| BYOL, momentum 0.99 | 0.64 (of 0–8) | 0.089 | 0.808 | 13.8 |
| BYOL, momentum 0 | 0.71 | 0.094 | 0.831 | 16.0 |

without a predictor, the projections collapse just as they did without the stop-gradient (loss −0.999, std 0.014), even though the stop-gradient is in place. the encoder, however, keeps most of what it learned (kNN 0.797, above the random network's 0.769 but below SimSiam's). neither BYOL variant collapses; at this budget the momentum-0 target does at least as well as the momentum-0.99 one. with one seed, the kNN gaps among the non-collapsed runs (a few points) are not conclusive.

**why:** with the predictor replaced by the identity, the loss compares $z_1$ directly with a detached $z_2$ from the same network. Chen & He (2021, §4.2) point out that in this case the symmetric loss's gradient has the same direction as the gradient *without* the stop-gradient, just halved: the stop-gradient is effectively removed, and the run is experiment 1's collapse again. the network can satisfy the loss by making the projector's output constant, because the stop-gradient doesn't stop the *shared weights* from moving both branches over successive steps. in the reference run the scale parameters of the projector's final BatchNorm shrank from 1 to about 0.10, squashing every $z$ towards the same vector (BatchNorm's bias); the encoder below it was much less affected, which is why kNN on $h$ survives. the predictor breaks the symmetry between the branches: one side must learn a mapping that predicts the other, and Tian, Chen & Ganguli (2021) show that in linear models the predictor's dynamics are what keep the eigen-directions of the representation alive. the momentum target is a stabiliser rather than a necessity here, consistent with SimSiam's claim (BYOL's own ablation found collapse at momentum 0, under different settings). note also how slowly the 0.99 target moves: after 156 steps it still carries about 21% of its initial weights (on paper 2), which may be why it learns more slowly in such a short run.
