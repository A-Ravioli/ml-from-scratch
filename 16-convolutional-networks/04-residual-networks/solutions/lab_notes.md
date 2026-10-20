# residual networks · lab notes

numbers are for real CIFAR-10 (5,000 training images, of which experiment 2 trains on the first 3,000, and 2,000 test images, all downsampled to $16 \times 16$), measured with the reference solution on an Apple M2 laptop CPU while other jobs were running (load average about 11; the lab took 134 s). experiment 2 was repeated with two more seeds (initialization and data order); the ordering of the three networks was the same every time. offline (the synthetic stand-in that `verify.py` uses) the lab runs end to end, but its numbers say nothing about CIFAR.

## experiment 1: gradients at initialization

**what people usually predict:** batch norm keeps everything at unit scale, so the gradients are about the same everywhere; or, from chapter 15, that the plain network's early gradients *vanish*.

**what happens:** the gradient norm of each block's first convolution, first block / last block:

| network | first block | last block | first / last |
|---|---|---|---|
| residual-18 | 5.1e-01 | 3.5e-01 | 1.5 |
| plain-18 | 5.2e-01 | 2.4e-01 | 2.2 |
| residual-34 | 2.0e+00 | 2.9e-01 | 6.9 |
| plain-34 | 1.0e+01 | 2.3e-01 | **44.9** |

the plain 34-layer network's gradients don't vanish: they **explode** towards the input, growing steadily block by block until the first block's gradient is 45 times the last one's. the residual networks' profiles are much flatter (7× for residual-34, 1.5× for residual-18).

**why:** batch norm fixes each layer's forward scale, but the backward pass multiplies the Jacobians of 33 layers, and a batch-norm layer's Jacobian isn't the identity: it divides by the batch standard deviation and projects out the mean and the direction of $\hat x$ (chapter 15, topic 02). in randomly initialized deep batch-norm networks, that product grows geometrically with depth: gradient explosion at initialization is a known property of plain batch-norm networks (Yang et al., 2019), and it's one way the degradation problem shows up. the early layers get huge updates relative to the late ones, which makes a single learning rate wrong for most of the network. in a residual network, each block's Jacobian is $I + \partial F/\partial x$, and the identity term passes the gradient through unchanged, so the product stays much closer to the identity: the growth is far slower (and with zero-initialized residual branches it would be flat).

## experiment 2: the degradation problem

**what people usually predict:** the deeper network fits the training data at least as well (it has twice the parameters), maybe overfitting more.

**what happens:** after 4 epochs on 3,000 images:

| network | parameters | training loss (last epoch) | training accuracy | test accuracy |
|---|---|---|---|---|
| plain-18 | 173,490 | 1.843 | 0.316 | 0.295 |
| plain-34 | 332,242 | **2.165** | 0.197 | 0.188 |
| residual-34 | 335,154 | **1.734** | 0.403 | 0.355 |

(seeds 1 and 2: training losses 1.876 / 2.085 / 1.794 and 1.875 / 2.040 / 1.839, test accuracies 0.265 / 0.177 / 0.299 and 0.289 / 0.207 / 0.312. an earlier, longer version with 5 epochs on 5,000 images gave the same ordering on three seeds.) the plain 34-layer network fits the *training* data much worse than the plain 18-layer one, despite having twice the parameters and containing it as a special case. adding the shortcuts to the same 34 layers turns it into the best of the three.

**why:** this is He et al.'s degradation problem in miniature: not overfitting (the training loss is worse, and the train and test accuracies are close), but optimization. the 34-layer plain network *could* compute what the 18-layer one does by turning its extra layers into identities (on paper 2), but that's a very particular point in weight space, and gradient descent starting from random weights, with the exploding gradients of experiment 1, doesn't find it in this budget. a residual block's default behaviour is the identity: with small weights, $x + F(x) \approx x$, so extra blocks start harmless and can only add useful corrections. the curves show the plain-34 network's loss falling slowest from the very first steps. at a longer budget the plain networks keep improving, but He et al. found the gap persists to convergence (their 56-layer plain net stayed behind the 20-layer one).

## experiment 3: delete a block

**what people usually predict:** deleting one of 16 blocks costs both networks a lot, or both a little.

**what happens:** test accuracy after replacing one non-downsampling block (13 of them) with the identity:

| network | intact | mean after removing one block | worst | best |
|---|---|---|---|---|
| residual-34 | 0.355 | 0.310 | 0.215 | 0.356 |
| plain-34 | 0.188 | 0.130 | 0.082 | 0.183 |

removing a single block from the residual network costs 4.5 points on average, and the late blocks almost nothing (the last six cost under 2 points each); the very first block (which works directly on the stem's output) and the block just before a downsampling hurt most. removing a block from the plain network drops it most of the way to chance (0.13 on average, and below chance twice), from an accuracy that was low to begin with. (with the longer budget of an earlier version, residual-34 lost 3 points on average and plain-34 fell to 0.11.)

**why:** in a plain network each layer's output is the next layer's only input, so deleting one feeds every later layer a representation it has never seen: the whole chain breaks. in a residual network, $x_{l+1} = x_l + F_l(x_l)$ unrolls into $2^n$ paths of different lengths (on paper 4); deleting a block removes only the paths through it, and every later block still receives $x_l$ plus the other corrections it was trained with, slightly perturbed. the network behaves like an ensemble of many shallower networks sharing weights (Veit et al., 2016), and its accuracy degrades smoothly when members are removed. that's also why ResNets tolerate stochastic depth (randomly dropping blocks during training, Huang et al., 2016) and why their effective depth is much smaller than their nominal depth.

## references

- Yang, Pennington, Rao, Sohl-Dickstein & Schoenholz (2019), "A mean field theory of batch normalization", *ICLR*. gradient explosion in deep batch-norm networks at initialization. [arXiv:1902.08129](https://arxiv.org/abs/1902.08129)
- Huang, Sun, Liu, Sedra & Weinberger (2016), "Deep networks with stochastic depth", *ECCV*. [arXiv:1603.09382](https://arxiv.org/abs/1603.09382)
