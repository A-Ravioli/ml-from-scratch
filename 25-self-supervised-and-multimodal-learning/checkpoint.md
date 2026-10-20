# 25 · self-supervised & multimodal learning · checkpoint

use this to check that you've finished the chapter, or to test out of it. work without notes, then compare with the answers. section 3 is checked by the topics' own tests.

## 1. concept check

1. why can't the training loss of a self-supervised method tell you whether its representation is any good?

<details><summary>answer</summary>

the loss measures the pretext task (match two views, reconstruct patches), not usefulness downstream. worse, some failures *are* the loss's optimum: a collapsed SimSiam encoder reaches the minimum, $-1$, while its features are useless. you need an evaluation protocol (probe, kNN, fine-tuning) or a direct measure of the embeddings (std, effective rank). (topics 01 and 03)

</details>

2. name the three standard protocols for scoring a representation and the question each one asks.

<details><summary>answer</summary>

linear probe: is the task's information linearly accessible in the frozen features? kNN: are same-class examples close together (local neighbourhoods class-pure)? fine-tuning: is the network a good starting point once every weight may adapt? (topic 01)

</details>

3. a linear probe standardises its features. with which statistics, and what goes wrong otherwise?

<details><summary>answer</summary>

the training set's mean and std, applied to both train and test features (a zero std replaced by 1). using the test set's own statistics leaks information and breaks on shifted or single-class test sets; skipping standardisation makes fixed-learning-rate gradient descent diverge or crawl on badly scaled features, and makes the L2 penalty uneven. (topic 01)

</details>

4. in weighted kNN with weights $e^{s/T}$, what happens as $T \to 0$ and $T \to \infty$?

<details><summary>answer</summary>

$T \to 0$: the most similar neighbour dominates (1-NN). $T \to \infty$: every weight tends to 1, a majority vote among the $k$. $T = 0.07$ is close to the 1-NN end: a similarity gap of 0.07 is a factor $e$. (topic 01)

</details>

5. write NT-Xent for one anchor and say what is excluded from the denominator and why.

<details><summary>answer</summary>

$\ell_i = -\log\big(e^{s_{i,p(i)}/T}/\sum_{j \ne i}e^{s_{ij}/T}\big)$ with cosine similarities over all $2B$ views. the anchor's similarity with itself ($s_{ii} = 1$) is excluded: it would be a free, maximal logit that keeps the loss above about $\log 2$ even for perfect embeddings. (topic 02)

</details>

6. an InfoNCE loss of 5.0 is reported. what else must you know to interpret it?

<details><summary>answer</summary>

the batch size (chance level is $\log(2B - 1)$) and the temperature (at small $T$ random embeddings score above chance; at large $T$ even perfect positives can't approach 0). (topic 02, lab)

</details>

7. why does SimCLR need strong colour distortion?

<details><summary>answer</summary>

two crops of one image share their colour histogram; without colour jitter the network can match them by colour statistics alone and never learn shape or object features. (topic 02)

</details>

8. why evaluate $h = f(x)$ rather than the projection $z = g(h)$?

<details><summary>answer</summary>

the loss makes $z$ invariant to the augmentations, discarding information (colour, orientation) that downstream tasks may need; the projection head absorbs that invariance and leaves more in $h$. SimCLR measured more than 10 points of linear-probe accuracy between them. (topic 02)

</details>

9. what are the two ingredients that keep SimSiam from collapsing, and what does each do?

<details><summary>answer</summary>

the stop-gradient on the target branch (each step moves predictions towards targets held fixed, instead of pulling both branches together) and the predictor (an asymmetric mapping, so the network can't satisfy the loss by making the projector's output constant). removing either collapses $z$ in the lab. (topic 03)

</details>

10. what's the difference between complete and dimensional collapse, and which metric catches each?

<details><summary>answer</summary>

complete: every embedding is the same vector (per-dimension std of the normalised embeddings → 0). dimensional: embeddings vary but only within a low-dimensional subspace (effective rank, $\exp$ of the entropy of the normalised singular values, far below $d$), which the std can miss. (topic 03)

</details>

11. why does MAE mask 75% of patches while BERT masks 15% of words?

<details><summary>answer</summary>

images are spatially redundant: with few patches hidden, each can be interpolated from its neighbours without understanding. masking most of the image forces global reasoning. language is dense, and predicting 15% of words is already a semantic task. (topic 04; in the lab's tiny setting the best ratio was 50%, with 25% and 75% close.)

</details>

12. why does MAE's encoder never see the mask tokens?

<details><summary>answer</summary>

the encoder processes only the visible patches: 4× cheaper MLPs and 16× cheaper attention at 75% masking, and no train/test mismatch from a `[MASK]` token that never appears at test time. the decoder inserts mask tokens, unshuffles, and predicts every patch. (topic 04)

</details>

13. why are MAE's reconstructions blurry, and why doesn't that matter?

<details><summary>answer</summary>

the per-pixel squared error is minimised by the conditional mean of the plausible completions, a blur where several are possible. the encoder still has to know which completions are plausible, which is what the representation needs. (topic 04)

</details>

14. how does CLIP's loss differ from SimCLR's?

<details><summary>answer</summary>

it's symmetric cross-entropy over a $B \times B$ image–text similarity matrix: each image picks its caption among $B$ captions and each caption its image among $B$ images. negatives are only from the other modality ($B - 1$ per anchor, not $2B - 2$), and the temperature is learned (stored as $\log s$, clamped at $s \le 100$). (topic 05)

</details>

15. a zero-shot classifier with prompts "the digit 0", …, "the digit 9" scores exactly chance on the lab's model. why?

<details><summary>answer</summary>

the training captions spell digits as words, so every numeral is `<unk>`: all ten prompts become the same token sequence, all class embeddings are equal, and the argmax always returns class 0. a prompt only works in the language the text encoder learned. (topic 05)

</details>

## 2. by hand

1. **kNN and the probe's standardisation.** training points on the unit circle $(1, 0)$ and $(-1, 0)$ of class 0, $(0.8, 0.6)$ and $(0.6, 0.8)$ of class 1, query $(0.96, 0.28)$. find the prediction for $k = 3$ at $T = 0.1$ and at $T = 1$. then standardise the one-feature training set $\{0, 2, 4\}$ and the test point $6$.

<details><summary>answer</summary>

similarities $0.96, 0.936, 0.8, -0.96$; the top three are $(1, 0)$ (class 0) and the two class-1 points. $T = 0.1$: class 0 scores $e^{9.6} = 14{,}765$, class 1 $e^{9.36} + e^{8} = 14{,}595$, so class 0. $T = 1$: $e^{0.96} = 2.61$ against $e^{0.936} + e^{0.8} = 4.78$, so class 1. standardising: mean 2, std $\sqrt{8/3} = 1.633$; the training set becomes $\{-1.22, 0, 1.22\}$ and the test point $(6 - 2)/1.633 = 2.45$.

</details>

2. **two contrastive losses.** for $z_1 = z_2 = ((1, 0), (0, 1))$ and $T = 1$, compute NT-Xent. then compute the CLIP loss with logit scale 1 for images $((1, 0), (1, 0))$ and captions $((1, 0), (0, 1))$.

<details><summary>answer</summary>

NT-Xent: every view sees its partner at similarity 1 and two negatives at 0: $\log(1 + 2/e) = 0.5514$. CLIP: logits $\begin{pmatrix}1 & 0\\ 1 & 0\end{pmatrix}$; rows give $\log(1 + e^{-1}) = 0.3133$ and $\log(1 + e) = 1.3133$, columns give $\log 2$ each; $\mathcal L = \frac12(0.8133 + 0.6931) = 0.7532$. since the two images are identical, the floor is $\log 2 = 0.6931$: each column's two logits are always equal, and the two rows share one softmax, so their losses average at least $\log 2$. an encoder that makes the two captions equally similar to the image reaches it; this configuration pays 0.06 more.

</details>

3. **masking and EMA.** $N = 5$ patches, mask ratio 0.6, noise $(0.7, 0.1, 0.9, 0.4, 0.3)$: find $N_\text{keep}$, the kept patches, `ids_restore` and `mask`. then one BYOL target update with momentum 0.9 from target weight 1.0 and online weight 3.0.

<details><summary>answer</summary>

$N_\text{keep} = \lfloor 5 \cdot 0.4 + 0.5\rfloor = 2$; `ids_shuffle` $= (1, 4, 3, 0, 2)$, so patches 1 and 4 are kept; `ids_restore` $= (3, 0, 4, 2, 1)$; `mask` $= (1, 0, 1, 1, 0)$. the target becomes $0.9 \cdot 1.0 + 0.1 \cdot 3.0 = 1.2$.

</details>

## 3. blind rebuild

in 60 minutes, with no notes, rebuild in fresh copies of the exercise notebooks:

- topic 02: `info_nce_loss` (stage 1) and `train_simclr` (stage 4), using your own `SimCLRAugment` and `ProjectionHead`;
- topic 03: `simsiam_loss`, `byol_loss` (stage 1) and `collapse_metrics` (stage 2).

check with:

```bash
python -m pytest 02-contrastive-learning -k "stage1 or stage4"
python -m pytest 03-non-contrastive-methods -k "stage1 or stage2"
```

## 4. readiness

move on when you score at least 80% on sections 1 and 2 and the rebuild passes. the same checkpoint works as a placement test: if you pass it cold, skim the lessons' spec boxes (they fix conventions later chapters rely on, such as the MAE patch order) and go straight to the capstone.
