# detection & segmentation

## why this matters

a self-driving car doesn't need to know that a photo "contains a pedestrian". it needs to know *where* every pedestrian is, how big, and which pixels are road. classification networks answer one question per image; **object detection** answers "what, and where?" for every object (a class and a box), and **semantic segmentation** answers "what?" for every pixel (a mask). both reuse everything you've built (convolutional backbones, receptive fields, residual blocks) and add a handful of ideas of their own, which are surprisingly small and surprisingly subtle.

for detection: how to measure the overlap of two boxes (IoU), how to turn thousands of overlapping guesses into one box per object (non-maximum suppression), and how a network decides which object each of its outputs is responsible for (anchors and matching). for segmentation: an architecture that sees the whole image *and* outputs full-resolution masks (the U-Net), and a loss that doesn't let a tiny object drown in a sea of background (Dice). you'll build all of them. in the lab you'll find that no NMS threshold works in a crowd, that a standard anchor grid can't see objects under 16 pixels at all, and that the loss that finds the most small objects isn't the one with the best overlap.

## warm-up: predict first

1. two $2 \times 2$ squares overlap in a $1 \times 1$ corner. what is their intersection over union?
2. a detector outputs 5 boxes on one person, all with high scores. what should be reported, and how would you decide which box?
3. a detector predicts relative to "anchor" boxes of sizes 32, 64 and 128 pixels. what happens to an object 10 pixels across?
4. 2% of an image's pixels belong to the object. what accuracy does a segmenter that predicts "background" everywhere get?
5. a segmentation network downsamples by 8 to see context. how can it output a mask at full resolution with sharp edges?

## prerequisites check

- [ ] broadcasting pairwise computations, `(N, 1, d)` against `(1, M, d)` (→ ch 00 · topic 02)
- [ ] receptive fields and downsampling; the stride as the jump between output positions (→ topic 02)
- [ ] the transposed convolution as the backward pass of a strided convolution (→ topic 01)
- [ ] VGG-style blocks of $3 \times 3$ convolutions with batch norm (→ topic 03); class imbalance and precision/recall (→ ch 07 · topic 03)

## core ideas

### idea 1: intersection over union

boxes are given as $(x_1, y_1, x_2, y_2)$ ("xyxy"): the top-left and bottom-right corners, with $x_2 \ge x_1$ and $y_2 \ge y_1$. the standard measure of how well two boxes agree is their **intersection over union**:

$$\operatorname{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}.$$

it's 1 for identical boxes, 0 for disjoint ones, and doesn't depend on the boxes' scale: two boxes off by 5 pixels have a high IoU if they're 200 pixels wide and a low one if they're 10 pixels wide, which is what "agree" should mean. the intersection of two axis-aligned boxes is itself a box, from $\max(x_1, x_1')$ to $\min(x_2, x_2')$ (and the same in $y$); if those limits cross, the boxes don't overlap, so clamp its width and height at 0. detection benchmarks call a predicted box correct if its IoU with an unmatched ground-truth box of the same class is at least 0.5 (or averaged over thresholds 0.5–0.95, COCO's AP).

**`box_iou(boxes1, boxes2)`** computes every pair at once: with `boxes1` of shape $(N, 4)$ and `boxes2` of shape $(M, 4)$, the corners `torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])` broadcast to $(N, M, 2)$, and the result is $(N, M)$, with no Python loop. guard the one division that can fail: two zero-area boxes have an empty union, and their IoU is defined as 0.

> **▶ build: stage 1.** implement `box_iou`.

### idea 2: non-maximum suppression

a detector makes a prediction at every position of its feature map (and for several anchors per position, idea 3). an object is seen by many neighbouring positions, so it gets many boxes, all slightly different, all with high scores. **non-maximum suppression** (NMS) keeps the best one of each cluster:

1. sort the boxes by score, highest first;
2. take the best remaining box and keep it;
3. discard every remaining box whose IoU with it is **greater than** the threshold (typically 0.5);
4. repeat from step 2 until no boxes remain.

the result is a list of indices in decreasing score order. it's greedy and simple, which is why it survived a decade of detectors (and real pipelines also drop boxes below a score threshold first, and run NMS separately per class). its weakness is built in: it can't tell a duplicate of the same object from a different object that overlaps it. two people standing close together have boxes with IoU of 0.5–0.75, as high as the overlap between two guesses at the *same* person, so whatever threshold you pick either deletes real people or keeps duplicates. the lab measures this.

implementation: compute the full IoU matrix once with your `box_iou`, then loop over the sorted indices with a boolean `suppressed` mask; when you keep box $i$, mark `suppressed |= iou[i] > threshold`. the loop runs at most $N$ times and does a vectorized update each time. return `torch.int64` indices (an empty tensor for no boxes).

> **▶ build: stage 2.** implement `nms(boxes, scores, iou_threshold=0.5)`. it must agree with `torchvision.ops.nms`.

> **🔮 predict:** objects in a crowd overlap their neighbours with IoU around 0.6. what does NMS at the default threshold of 0.5 do to them?

### idea 3: anchors, and who is responsible for which object

how does a convolutional network output a variable number of boxes? one answer (Faster R-CNN, SSD, RetinaNet, YOLO v2–v3) is to fix a large set of **anchors**, reference boxes laid out on a grid, and turn detection into a dense prediction problem: for every anchor, a classifier says "object of class $c$ or background", and a regressor says how to shift and stretch the anchor to fit the object. the anchors are tied to the feature map: a map of $H \times W$ positions computed with a total stride $s$ has a centre at $\big((j + \tfrac12)s, (i + \tfrac12)s\big)$ for position $(i, j)$, and at each centre several anchors of different **sizes** and **aspect ratios**. for size $a$ and ratio $r = \text{height}/\text{width}$, the anchor has width $a/\sqrt{r}$ and height $a\sqrt{r}$, so its area is $a^2$ whatever the ratio.

**`make_anchors(feature_size, stride, sizes, ratios)`** returns all of them as xyxy boxes of shape $(H W \cdot |\text{sizes}| \cdot |\text{ratios}|, 4)$, ordered by row, then column, then size, then ratio, so that anchor $k$ lines up with output channel $k$ of the network's prediction head at that position (`feature_size` is an int or a pair $(H, W)$).

training needs a target for every anchor, and that's **matching**. with the IoU matrix between anchors and ground-truth boxes:

- an anchor whose best IoU with any object is $\ge$ `pos_threshold` (0.5) is **positive** (label 1), responsible for that object;
- an anchor whose best IoU is $<$ `neg_threshold` (0.4) is **background** (label 0);
- anchors in between are **ignored** (label $-1$): too ambiguous to train on either way;
- every object's single best anchor is made positive even if its IoU is below `pos_threshold`, so no object is left without a responsible anchor.

**`match_anchors(anchors, gt_boxes, *, pos_threshold=0.5, neg_threshold=0.4)`** returns `(labels, matched)`: the label of every anchor and the index of the object it's matched to: its best object, except that a forced anchor is matched to the object that forced it (with $-1$ everywhere when there are no objects, all labelled background). two consequences shape detector design. first, an image with ten objects and 20,000 anchors has a few dozen positives and nearly 20,000 negatives, a severe class imbalance (RetinaNet's focal loss exists to fix it). second, an object much smaller than the smallest anchor has a low IoU with *every* anchor, so it only ever gets its one forced match; feature pyramids (FPN) add finer feature maps with smaller anchors for exactly that reason.

> **▶ build: stage 3.** implement `make_anchors` and `match_anchors`.

### idea 4: the Dice loss

semantic segmentation is classification at every pixel: the network outputs a logit per pixel (per class) and the mask is where it's positive. the obvious loss is per-pixel binary cross-entropy. the obvious problem is imbalance: if the object covers 2% of the image, predicting "background" everywhere is 98% accurate and already has a low average cross-entropy, and the pixels of a small object contribute almost nothing to the loss.

the **Dice coefficient** measures overlap instead, like IoU: for a predicted mask $P$ and the true mask $T$, $\operatorname{Dice} = \frac{2|P \cap T|}{|P| + |T|}$, which is 0 for "background everywhere" no matter how small the object is. to train with it, replace the hard mask by the predicted probabilities $p = \sigma(\text{logits})$ ("soft Dice"):

$$\operatorname{Dice}_\text{soft} = \frac{2\sum_i p_i t_i + \epsilon}{\sum_i p_i + \sum_i t_i + \epsilon}, \qquad \mathcal{L}_\text{Dice} = 1 - \operatorname{Dice}_\text{soft}.$$

the sums run over the pixels of **one image**, and the loss averages $1 - \operatorname{Dice}$ over the batch: summing over the whole batch would let the images with large objects dominate the ones with small objects. the smoothing $\epsilon$ (1 by default) keeps the ratio defined for an empty mask predicted empty (Dice $= \epsilon/\epsilon = 1$, loss 0) and smooths the gradient for tiny objects. Dice is less well-behaved than cross-entropy (its gradient for one pixel depends on all the others, and it tends to over-segment), so the common default is the sum of the two.

> **▶ build: stage 4.** implement `dice_loss(logits, targets, *, eps=1.0)`.

### idea 5: the U-Net

a segmentation network needs two things that pull in opposite directions. to decide *what* a pixel is, it needs a large receptive field (context): downsampling, as in every classifier. to say *exactly which* pixels, it needs full resolution: no downsampling. the **U-Net** (Ronneberger, Fischer & Brox, 2015) does both. an **encoder** (the left side of the "U") is a VGG-like stack: at each of `depth` levels, a `DoubleConv` (two $3 \times 3$ conv → BN → ReLU layers) followed by $2 \times 2$ max pooling, with the channels doubling from `base_channels`. a **bottleneck** `DoubleConv` works at the lowest resolution with $2^\text{depth}$ times the base width. the **decoder** (the right side) goes back up: at each level, a $2 \times 2$ **transposed convolution** with stride 2 doubles the resolution and halves the channels (topic 01: it's the backward pass of a strided convolution, used forward), the result is **concatenated** with the encoder's feature map at the same resolution (the **skip connection**), and a `DoubleConv` merges them. a final $1 \times 1$ convolution outputs the logits.

the skip connections are what make the edges sharp: the decoder's upsampled maps know *what* is there (from the bottleneck's large receptive field) but only roughly where, and the encoder's maps at the same resolution know exactly where the edges are but not what they belong to. concatenation gives the decoder both. a U-Net with `depth` levels halves the resolution `depth` times, so the input's height and width must be divisible by $2^\text{depth}$ (raise a `ValueError` otherwise).

store the parts as `self.downs` (an `nn.ModuleList` of `DoubleConv`s), `self.bottleneck`, `self.ups` (the transposed convolutions, deepest first), `self.up_convs` (the decoder's `DoubleConv`s, each taking twice its level's width) and `self.out_conv`. in `forward`, keep each encoder `DoubleConv`'s output as a skip *before* pooling, and concatenate `[skip, upsampled]` in that order along the channels.

> **🔮 predict:** a U-Net trained with cross-entropy and one trained with Dice on images whose objects cover 3% of the pixels: which one predicts *more* foreground than there is?

> **▶ build: stage 5.** implement `DoubleConv(in_channels, out_channels)` and `UNet(in_channels=1, out_channels=1, *, base_channels=32, depth=3)`.

## worked example

**IoU.** $A = (0, 0, 2, 2)$, $B = (1, 1, 3, 3)$: the intersection is $(1, 1, 2, 2)$, area 1; the union is $4 + 4 - 1 = 7$: IoU $= 1/7 \approx 0.143$. $(0, 0, 1, 1)$ inside $(0, 0, 2, 2)$: $1/4$.

**NMS.** four boxes: $b_0 = (0, 0, 10, 10)$ with score 0.9, $b_1 = (1, 1, 11, 11)$ with 0.95, $b_2 = (20, 20, 30, 30)$ with 0.5, $b_3 = (0, 0, 10, 9)$ with 0.3. IoUs: $b_0$–$b_1$ $= 81/119 = 0.68$, $b_1$–$b_3$ $= 72/118 = 0.61$, $b_0$–$b_3$ $= 90/100 = 0.9$, $b_2$ overlaps nothing. at threshold 0.5: keep $b_1$ (best score), which suppresses $b_0$ and $b_3$; keep $b_2$: result $[1, 2]$. at threshold 0.7: $b_1$ suppresses nothing (0.68 and 0.61 are below 0.7), so $b_0$ is kept next and suppresses $b_3$ (0.9): result $[1, 0, 2]$.

**anchors.** a $2 \times 2$ feature map with stride 16, size 32, ratios $\tfrac12, 1, 2$: 12 anchors. the first is centred at $(8, 8)$ with width $32/\sqrt{0.5} = 45.25$ and height $22.63$; the second is the $32 \times 32$ square $(-8, -8, 24, 24)$; anchors 3–5 sit at $(24, 8)$, the next column.

**Dice.** a $4 \times 4$ mask whose top half is foreground (8 pixels), with all logits 0 ($p = \tfrac12$ everywhere): $\sum p t = 4$, $\sum p = 8$, $\sum t = 8$, so $\operatorname{Dice} = \frac{2 \cdot 4 + 1}{8 + 8 + 1} = \frac{9}{17}$ and the loss is $\frac{8}{17} \approx 0.47$ (exactly $\frac12$ with $\epsilon = 0$).

**a U-Net's parameters.** `UNet(1, 2, base_channels=4, depth=2)`: a `DoubleConv(i, o)` has $9 i o + 9 o^2$ weights and $4 o$ batch-norm parameters. encoder `DoubleConv(1, 4)` and `(4, 8)`, bottleneck `(8, 16)`, transposed convolutions $16 \cdot 8 \cdot 4 + 8$ and $8 \cdot 4 \cdot 4 + 4$, decoder `DoubleConv(16, 8)` and `(8, 4)`, and the $1 \times 1$ output $4 \cdot 2 + 2$: $196 + 896 + 3{,}520 + 520 + 132 + 1{,}760 + 448 + 10 = 7{,}482$ parameters. the default `UNet()` has 1,926,433.

the tests check every number in this section.

## on paper

1. **IoU and the shift that keeps it above 0.5.** two $w \times w$ squares, one shifted horizontally by $d$. find their IoU, and the largest shift (as a fraction of $w$) that keeps IoU $\ge 0.5$. what does it mean for a 16-pixel object?

<details><summary>solution</summary>

the intersection is $(w - d) w$, the union $2w^2 - (w - d)w$, so $\operatorname{IoU} = \frac{w - d}{w + d} = \frac{1 - \delta}{1 + \delta}$ with $\delta = d/w$. IoU $\ge 0.5$ needs $1 - \delta \ge \frac12(1 + \delta)$, i.e. $\delta \le \frac13$. a 16-pixel object must be localized to within 5 pixels in each direction; a 200-pixel one to within 66. the crowded scenes in the lab have neighbours shifted by $\delta = 0.15$–$0.3$, so their IoUs are $0.54$–$0.74$.

</details>

2. **the IoU of a small object with a large anchor.** a $b \times b$ object sits centred inside an $a \times a$ anchor with $b < a$. what is the IoU, and how small can $b$ be for the IoU to reach 0.5?

<details><summary>solution</summary>

the intersection is the object, $b^2$, the union the anchor, $a^2$: IoU $= (b/a)^2$. it reaches 0.5 only for $b \ge a/\sqrt 2 \approx 0.71a$. with a smallest anchor of 32, an object under 23 pixels can never be a regular positive, however well it's centred; a $16 \times 16$ object has IoU at most $1/4$. (the lab's coverage curve shows exactly this edge.)

</details>

3. **the gradient of soft Dice.** with $S = \sum p_i t_i$, $U = \sum p_i + \sum t_i$, show that $\partial \mathcal{L}/\partial p_k = -\frac{2 t_k (U + \epsilon) - (2S + \epsilon)}{(U + \epsilon)^2}$. compare the size of this gradient for a foreground pixel when the object has 4 pixels and when it has 400 (prediction $p = \frac12$ everywhere on a $64 \times 64$ image).

<details><summary>solution</summary>

$\mathcal{L} = 1 - \frac{2S + \epsilon}{U + \epsilon}$; $\partial S/\partial p_k = t_k$ and $\partial U/\partial p_k = 1$, so the quotient rule gives the formula. with $p = \frac12$ on $n = 4{,}096$ pixels and $m$ foreground pixels: $S = m/2$, $U = n/2 + m$. for a foreground pixel the numerator is $2(U + \epsilon) - (m + \epsilon) = n + m + \epsilon$, so the gradient magnitude is $\frac{n + m + \epsilon}{(n/2 + m + \epsilon)^2}$: $9.7 \times 10^{-4}$ for $m = 4$ and $7.5 \times 10^{-4}$ for $m = 400$, about the same per pixel (slightly larger for the small object). cross-entropy's gradient per pixel is also similar, but its *total* pull towards foreground scales with $m$, against $n - m$ pixels pulling towards background; Dice's normalization by $U$ makes the whole object count as much as the whole background, whatever its size.

</details>

4. **why skip connections.** a U-Net of depth 3 without skip connections must reconstruct a $64 \times 64$ mask from an $8 \times 8$ bottleneck map. how many numbers per output pixel does the bottleneck provide (with $8 \cdot 32 = 256$ channels), and what is lost?

<details><summary>solution</summary>

$8 \times 8 \times 256 = 16{,}384$ numbers for $4{,}096$ output pixels: 4 per pixel, enough capacity in principle, but each bottleneck position summarizes an $8 \times 8$ patch (jump 8), and its features are invariant to small shifts within it (topic 02): the precise position of an edge inside the patch has been pooled away, and the transposed convolutions can only produce blurry, blocky boundaries. the skip connections hand the decoder the encoder's full-resolution maps, where each position still knows its exact location.

</details>

## experiment

`lab.py` puts the pieces to work on synthetic scenes:

1. NMS thresholds against precision and recall, in sparse and in crowded scenes.
2. which object sizes a standard anchor grid can match at IoU 0.5.
3. a U-Net segmenting small shapes, trained with cross-entropy, Dice, or both.

## check your understanding

1. why is IoU, and not the distance between box centres, the standard measure of a detection's quality?

<details><summary>answer</summary>

IoU is scale-invariant and accounts for size and shape as well as position: a 5-pixel offset is negligible for a large box and fatal for a small one, and a box with the right centre but twice the width is wrong. a centre distance would need a per-size tolerance and ignore the extent entirely.

</details>

2. NMS visits boxes in decreasing score order. what would go wrong if it visited them in index order?

<details><summary>answer</summary>

a low-scoring, badly placed box that happened to come first would be kept and would suppress the better boxes overlapping it. sorting guarantees that each kept box is the best remaining one in its neighbourhood.

</details>

3. why does anchor matching have an "ignore" band between the two thresholds?

<details><summary>answer</summary>

an anchor with IoU 0.45 with an object is neither a good example of that object (its box is off) nor a good example of background (it contains much of the object). forcing a label on it adds noisy, contradictory targets; ignoring it lets the classifier learn from clear cases only.

</details>

4. why must the Dice loss be computed per image before averaging, rather than over the whole batch?

<details><summary>answer</summary>

summed over the batch, the large objects' pixels dominate both $\sum p t$ and $\sum p + \sum t$, so a missed small object in one image barely changes the batch's Dice. per image, every image's Dice is a number between 0 and 1, and a missed small object costs its image the full amount.

</details>

5. a U-Net of depth 4 is given a $100 \times 100$ image. what goes wrong, and what are two fixes?

<details><summary>answer</summary>

100 isn't divisible by $2^4 = 16$: pooling floors $100 \to 50 \to 25 \to 12 \to 6$, and upsampling gives $12 \to 24$, which can't be concatenated with the 25-pixel skip. fixes: pad (or crop) the input to a multiple of 16 (e.g. 112) and crop the output back, or crop/pad each skip to the upsampled map's size (the original U-Net cropped, because it used unpadded convolutions).

</details>

6. what's the difference between semantic segmentation and instance segmentation, and which of this topic's tools would you combine to get the second?

<details><summary>answer</summary>

semantic segmentation labels every pixel with a class, so two touching people form one "person" region. instance segmentation separates objects of the same class: one mask per object. Mask R-CNN combines detection (anchors, matching, NMS to get one box per object) with a small segmentation head that predicts a mask inside each box.

</details>

## common bugs

- **unclamped intersections.** symptom: disjoint boxes get positive (or negative) IoU when both widths are negative and their product is positive. clamp the intersection's width and height at 0.
- **`>=` instead of `>` in NMS.** symptom: disagrees with torchvision only on exact ties; boxes with IoU exactly at the threshold must survive.
- **returning kept indices in index order.** symptom: the right set of boxes in the wrong order; NMS returns indices by decreasing score.
- **ratio as width / height.** symptom: anchors transposed (tall where they should be wide). in this course, ratio = height / width, and the area stays size².
- **global Dice over the batch.** symptom: small objects are ignored in batches that also contain large ones. compute Dice per image, then average.
- **no sigmoid in Dice, or skip and upsampled maps concatenated in the wrong order.** symptom: losses outside $[0, 1]$, or a U-Net whose decoder weights don't match the spec. apply `torch.sigmoid` to the logits, and concatenate `[skip, upsampled]`.

## going deeper

- Ren, He, Girshick & Sun (2015), "Faster R-CNN: towards real-time object detection with region proposal networks", *NeurIPS*. anchors and matching. [arXiv:1506.01497](https://arxiv.org/abs/1506.01497)
- Lin, Goyal, Girshick, He & Dollár (2017), "Focal loss for dense object detection", *ICCV*. RetinaNet and the imbalance problem. [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)
- Lin et al. (2017), "Feature pyramid networks for object detection", *CVPR*. [arXiv:1612.03144](https://arxiv.org/abs/1612.03144)
- Bodla, Singh, Chellappa & Davis (2017), "Soft-NMS: improving object detection with one line of code", *ICCV*. [arXiv:1704.04503](https://arxiv.org/abs/1704.04503)
- Ronneberger, Fischer & Brox (2015), "U-Net: convolutional networks for biomedical image segmentation", *MICCAI*. [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)
- Milletari, Navab & Ahmadi (2016), "V-Net: fully convolutional neural networks for volumetric medical image segmentation", *3DV*. the soft Dice loss. [arXiv:1606.04797](https://arxiv.org/abs/1606.04797)
- Carion et al. (2020), "End-to-end object detection with transformers", *ECCV*. DETR: detection without anchors or NMS. [arXiv:2005.12872](https://arxiv.org/abs/2005.12872)

## connections

- ← builds on: transposed convolutions (topic 01), receptive fields, strides and dilation (topic 02), VGG blocks with batch norm (topic 03), skip connections (topic 04), backbones and transfer (topic 05), class imbalance and precision/recall (ch 07 · topic 03).
- → used in: DETR's set prediction and vision transformers (ch 18), U-Nets as the denoising network of diffusion models (ch 22), segmentation and detection as downstream tasks for self-supervised backbones (ch 25), robustness and attribution maps (ch 28).
