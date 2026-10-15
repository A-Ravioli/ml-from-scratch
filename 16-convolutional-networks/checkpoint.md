# 16 · convolutional networks · checkpoint

use this to finish the chapter, or to test out of it before starting. sections 1 and 2 take about an hour with pen and paper. section 3 takes 60 minutes at a computer.

## 1. concept check

1. what is the difference between the convolution of signal processing and the "convolution" of deep-learning libraries, and why doesn't it matter for a learned layer?

<details><summary>answer</summary>

the mathematical convolution flips the kernel ($x[r - i, c - j]$); libraries compute cross-correlation ($x[r + i, c + j]$). a learned kernel can absorb the flip: if the flipped kernel is better, gradient descent learns that one. it matters only when you compare against an oracle or reuse hand-designed kernels.

</details>

2. give the output size of a convolution along one axis with padding $p$, stride $s$, dilation $d$ and kernel $k$, and the size for $n = 32$, $k = 3$, $p = 1$, $s = 2$.

<details><summary>answer</summary>

$\lfloor (n + 2p - d(k - 1) - 1)/s \rfloor + 1$; for the example $\lfloor 31/2 \rfloor + 1 = 16$.

</details>

3. what does im2col compute, what is its memory cost, and why is col2im its transpose rather than its inverse?

<details><summary>answer</summary>

it lays every receptive field out as a column, $(N, C k_h k_w, H_\text{out} W_\text{out})$, so the convolution becomes one matrix product with the flattened weight. each pixel is copied into up to $k_h k_w$ columns, so the matrix is about $k^2$ times the input. col2im *adds* every column entry back to its pixel, which is the adjoint of copying; `col2im(im2col(x))` multiplies each pixel by the number of windows covering it.

</details>

4. write the three gradients of a convolutional layer in terms of im2col, and say which sums over what.

<details><summary>answer</summary>

with $X_n = \operatorname{im2col}(x)_n$ and $G_n$ the upstream gradient as $C_\text{out} \times L$: $\partial L/\partial W = \sum_n G_n X_n^\top$ (over images and positions), $\partial L/\partial b = \sum_n G_n \mathbf 1$ (over images and positions), $\partial L/\partial x = \operatorname{col2im}(W^\top G)$ (each pixel collects from the windows that cover it).

</details>

5. where does max pooling send its gradient, and what fraction of a $2 \times 2$ pool's inputs get a nonzero gradient on one step?

<details><summary>answer</summary>

the whole gradient of each window goes to its maximum (the first one in case of ties); the other inputs get zero. one input per window: a quarter. overlapping windows add their contributions.

</details>

6. state the receptive field recursion, and explain why downsampling early grows the field cheaply.

<details><summary>answer</summary>

$r_l = r_{l-1} + d_l(k_l - 1) j_{l-1}$ and $j_l = j_{l-1} s_l$, from $r_0 = j_0 = 1$. each layer's contribution is multiplied by the product of all strides below it, so a stride-2 layer early on doubles what every later layer adds, at the cost of resolution.

</details>

7. what is the effective receptive field, and how does it grow with depth compared with the theoretical one?

<details><summary>answer</summary>

the map of $|\partial y/\partial x_{ij}|$ for a unit $y$: how much each input pixel actually influences it. it's roughly Gaussian, because the paths from the input to the unit sum many small random offsets, and its width grows like $\sqrt{L}$ while the theoretical field grows like $L$, so deep networks use a shrinking fraction of their nominal field.

</details>

8. a CNN with three stride-2 steps and global average pooling: is it translation-invariant? explain.

<details><summary>answer</summary>

only approximately. convolutions are equivariant and the global pool is invariant, but each stride-2 step keeps one sample per $2 \times 2$ block, so shifts that aren't multiples of the total stride (8) change the feature map in ways that aren't shifts (aliasing), and objects near the border meet zero padding. one-pixel shifts change a few percent of predictions; invariance in practice comes as much from augmentation as from the architecture.

</details>

9. VGG-16 has 138 million parameters. where are most of them, and where is most of its compute? why the difference?

<details><summary>answer</summary>

89% of the parameters are in the three fully connected layers (74% in the first), while 99% of the multiply–accumulates are in the 13 convolutions. a convolutional weight is used at every output position, a fully connected weight once per image.

</details>

10. what is the degradation problem, and why is it evidence of an optimization failure rather than overfitting?

<details><summary>answer</summary>

deeper plain networks reach *higher training error* than shallower ones, although they can represent everything the shallower ones can (extra layers as identities). overfitting would mean lower training error and higher test error. residual blocks make the identity the default ($x + F(x)$ with $F \approx 0$), which removes the problem.

</details>

11. why does a residual network survive the deletion of a single block, while a plain network collapses?

<details><summary>answer</summary>

unrolled, $x_{l+1} = x_l + F_l(x_l)$ gives $2^n$ paths of different lengths; deleting a block removes only the paths through it, and later blocks still receive $x_l$ plus most of the corrections they were trained with. a plain network has one path, and deleting a layer feeds every later layer an input it has never seen.

</details>

12. how many fewer multiply–accumulates does a $3 \times 3$ depthwise separable convolution do than a standard one, and why is it usually less than that much faster?

<details><summary>answer</summary>

the ratio is $\frac{1}{C_\text{out}} + \frac19$, about 8–9× fewer for wide layers. the depthwise step does only 9 MACs per input value it reads, so it's limited by memory bandwidth rather than arithmetic, and runs far below the hardware's peak; measured speedups are often 1.5–3×.

</details>

13. linear probing versus fine-tuning: what does each train, and what does each cost the original task?

<details><summary>answer</summary>

a linear probe trains only a new head on a frozen backbone (in eval mode), so the backbone and its old tasks are untouched, but the features can't adapt. fine-tuning trains everything from the pretrained weights (often with a smaller backbone learning rate), adapts the features and usually wins on the new task, but moves the backbone away from the old one (catastrophic forgetting).

</details>

14. why can a "frozen" backbone still change, and how do you prevent it?

<details><summary>answer</summary>

`requires_grad=False` stops weight updates, but batch norm in training mode still updates its running mean and variance on every forward pass, changing the features. run frozen parts in eval mode.

</details>

15. why does a detector need NMS, why does it fail in crowds, and why does a U-Net need skip connections?

<details><summary>answer</summary>

dense predictions fire many overlapping boxes on each object, so NMS keeps the best of each cluster. it judges duplicates only by IoU, and neighbours in a crowd overlap as much as duplicates do, so no threshold separates them. a U-Net's decoder knows *what* is where only coarsely after downsampling; the skip connections hand it the encoder's full-resolution maps, so the masks' edges land on the right pixels.

</details>

## 2. by hand (no computer)

1. **a convolution and its gradients.** $x = \begin{pmatrix} 1 & 2 & 0 \\ 0 & 1 & 3 \\ 2 & 1 & 1 \end{pmatrix}$, $w = \begin{pmatrix} 1 & -1 \\ 2 & 0 \end{pmatrix}$, no bias, stride 1, no padding (cross-correlation). (a) compute the output. (b) with $\partial L/\partial y = \mathbf{1}$ (all ones), compute $\partial L/\partial w$, $\partial L/\partial b$ and $\partial L/\partial x$.

<details><summary>solution</summary>

(a) $y[0, 0] = 1 \cdot 1 + 2 \cdot (-1) + 0 \cdot 2 + 1 \cdot 0 = -1$, $y[0, 1] = 2 - 0 + 2 + 0 = 4$, $y[1, 0] = 0 - 1 + 4 + 0 = 3$, $y[1, 1] = 1 - 3 + 2 + 0 = 0$: $y = \begin{pmatrix} -1 & 4 \\ 3 & 0 \end{pmatrix}$.
(b) $\partial L/\partial w[i, j] = \sum_{r, c} x[r + i, c + j]$, the sum of the $2 \times 2$ window starting at $(i, j)$: $\begin{pmatrix} 4 & 6 \\ 4 & 6 \end{pmatrix}$. $\partial L/\partial b = 4$ (one per output). $\partial L/\partial x[a, b]$ sums $w[i, j]$ over the windows that read pixel $(a, b)$ at tap $(i, j)$: corners get one tap each ($1, -1, 2, 0$), edges two, the centre all four: $\begin{pmatrix} 1 & 0 & -1 \\ 3 & 2 & -1 \\ 2 & 2 & 0 \end{pmatrix}$. (it's the zero-padded `dout` cross-correlated with the flipped kernel.)

</details>

2. **ResNet-18's receptive field.** the CIFAR ResNet-18 has a $3 \times 3$ stride-1 stem, then 16 $3 \times 3$ convolutions on its main path, with stride 2 in the first convolution of stages 2, 3 and 4. (a) what are the spatial sizes of the four stages' outputs for a $32 \times 32$ input? (b) what is the theoretical receptive field (and the jump) of a unit in the last stage, along the main path? (c) is it larger than the image, and what does topic 02 say about how much of it matters?

<details><summary>solution</summary>

(a) $32, 16, 8, 4$. (b) stem: $r = 3$; stage 1, 4 convs: $+2$ each, $r = 11$; stage 2: the strided conv adds $2 \cdot 1$ ($r = 13$, then $j = 2$), three more add $2 \cdot 2$ each: $r = 25$; stage 3: $+2 \cdot 2 = 29$ ($j = 4$), then $+8$ three times: $53$; stage 4: $+2 \cdot 4 = 61$ ($j = 8$), then $+16$ three times: $r = 109$, $j = 8$. (c) 109 pixels is more than three times the 32-pixel image (the rest is padding), so every last-stage unit can see the whole image; the effective field is a Gaussian-like blob much smaller than 109, but at this depth it still covers most of a $32 \times 32$ image.

</details>

3. **IoU and NMS.** boxes (xyxy) with scores: $A = (0, 0, 4, 4)$, 0.9; $B = (1, 0, 5, 4)$, 0.8; $C = (0, 0, 4, 2)$, 0.7; $D = (6, 6, 8, 8)$, 0.6. (a) compute the IoUs of $A$ with $B$, $A$ with $C$, and $B$ with $C$. (b) which boxes does NMS keep at threshold 0.5, in what order? (c) the Dice loss of a mask predicted with probability 0.5 on all 16 pixels of a $4 \times 4$ image whose true mask has 4 foreground pixels ($\epsilon = 1$).

<details><summary>solution</summary>

(a) $A \cap B = (1, 0, 4, 4)$, area 12, union $16 + 16 - 12 = 20$: 0.6. $C \subset A$: $8/16 = 0.5$. $B \cap C = (1, 0, 4, 2)$, area 6, union $16 + 8 - 6 = 18$: $1/3$. (b) keep $A$; it suppresses $B$ ($0.6 > 0.5$) but not $C$ (0.5 is not *greater* than 0.5); keep $C$; keep $D$ (disjoint): $[A, C, D]$, i.e. indices $[0, 2, 3]$. (c) $\sum p t = 2$, $\sum p = 8$, $\sum t = 4$: Dice $= \frac{4 + 1}{12 + 1} = \frac{5}{13}$, loss $\frac{8}{13} \approx 0.615$.

</details>

## 3. blind rebuild

with no notes and no looking at your old code, in 60 minutes: rebuild **convolution** (topic 01, stages 1–4): `conv_output_size`, `conv2d_naive`, `im2col`, `col2im`, `conv2d` and `conv2d_backward`. work in a scratch directory, starting from the *untouched* stub notebook and the topic's checks, so your own solved `exercise.py` stays where it is. from the repo root:

```bash
T=16-convolutional-networks/01-convolution
mkdir -p /tmp/rebuild-16
git show HEAD:$T/exercise.py > /tmp/rebuild-16/exercise.py   # the stubs as shipped
cp $T/test_exercise.py /tmp/rebuild-16/
```

(if you've committed your own solutions, regenerate the stubs instead: copy `$T/solutions/` to `/tmp/rebuild-16/solutions/` and run `python scripts/stubify.py /tmp/rebuild-16`.) open `/tmp/rebuild-16/exercise.py` with `marimo edit`, implement stages 1–4 from memory, and run the checks against your copy:

```bash
pytest /tmp/rebuild-16/test_exercise.py
```

you pass if every check is green within the hour, including the line-count checks that your im2col version doesn't loop over pixels.

**harder variant** (for testing out of the chapter with confidence): within 100 minutes in total, also rebuild topic 04's `BasicBlock` and `ResNet` (stages 1–2) and topic 06's `box_iou` and `nms` (stages 1–2), each in its own scratch directory the same way:

```bash
pytest /tmp/rebuild-16-res/test_exercise.py -k "stage1 or stage2"
pytest /tmp/rebuild-16-det/test_exercise.py -k "stage1 or stage2"
```

(run them from the repo root: the notebooks import earlier `mlfs` modules, which the tests resolve to the reference solutions, so only the code you rebuilt is being tested.)

## 4. readiness

move on if you got at least 80% of sections 1–2 (15 concept questions + 3 problems, counting each part of a problem separately) and passed section 3. otherwise, revisit the topics behind the questions you missed: each answer above names the idea, and the topic lessons are organized by those ideas.
