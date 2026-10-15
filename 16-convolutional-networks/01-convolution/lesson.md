# convolution

## why this matters

feed a $32 \times 32$ colour image to chapter 15's MLP and its first layer, with 1,000 hidden units, has $3{,}072 \times 1{,}000 \approx 3$ million weights. that's a lot, but the size isn't the real problem. the real problem is what those weights *don't know*. a cat in the top-left corner and the same cat moved three pixels to the right are, to an MLP, unrelated inputs: different pixels light up, different weights respond, and whatever the network learned about the first cat has to be learned again for the second. nothing in a dense layer says that neighbouring pixels belong together, or that an edge is an edge wherever it appears.

a convolutional layer builds both facts in. each output looks only at a small **local** window of the input (a $3 \times 3$ patch, say), and **the same** small set of weights slides over every position. a layer with 64 filters of size $3 \times 3$ over 3 colour channels has $64 \times 27 + 64 = 1{,}792$ parameters, whatever the image size, and an edge detector learned in one corner works everywhere. in chapter 00's capstone you slid hand-made kernels (blurs, Sobel edges) over photos. this topic turns that operation into a trainable layer: you'll write it as loops you can read, then as one matrix product you can afford (the trick every deep-learning library is built on), and then its backward pass, so the layer can learn its own kernels. in the lab you'll see what it learns, and whether learning the filters matters as much as you'd think.

## warm-up: predict first

1. a $3 \times 3$ convolution maps 64 channels to 128. how many weights does it have, and how does that change if the images grow from $32 \times 32$ to $224 \times 224$?
2. a $3 \times 3$ kernel with stride 2 and padding 1 slides over a $7 \times 7$ image. how big is the output?
3. you stack two $3 \times 3$ convolutions. how many input pixels can one output of the second layer depend on?
4. deep-learning "convolution" doesn't flip the kernel; the mathematical convolution does. does it matter for a layer whose kernel is learned?
5. the gradient of a convolution with respect to its *input* is also a convolution-like operation. with which kernel?

## prerequisites check

- [ ] you can slide a kernel over an image with loops and with array slicing (→ ch 00 · capstone, `correlate2d` and `filter2d`)
- [ ] you can reshape, transpose and broadcast arrays and multiply batched matrices with `@` (→ ch 00 · topic 02)
- [ ] you know what a vector–Jacobian product is and that the backward pass of $y = Wx$ is $\bar x = W^\top \bar y$, $\bar W = \bar y x^\top$ (→ ch 14 · topic 03)
- [ ] you know PyTorch's tensor conventions: batch first, float64 for comparisons (→ ch 15 · topic 07)

## core ideas

### idea 1: cross-correlation, padding, stride, dilation

start with one channel. a kernel $w$ of size $k_h \times k_w$ slides over an image $x$; at each position, multiply the overlapping numbers and add them up:

$$y[r, c] = \sum_{i=0}^{k_h - 1}\sum_{j=0}^{k_w - 1} w[i, j]\; x[r + i,\; c + j].$$

that's **cross-correlation**. the textbook **convolution** flips the kernel first ($x[r - i, c - j]$), which makes it commutative and associative, properties signal processing relies on. a learned kernel doesn't care: if flipping helped, the network would simply learn the flipped kernel. so every deep-learning library, PyTorch included, computes cross-correlation and calls it convolution. you'll do the same.

real images have **channels** (three colours in, many feature maps further in). a layer with $C_\text{in}$ input and $C_\text{out}$ output channels has a weight of shape $(C_\text{out}, C_\text{in}, k_h, k_w)$ and a bias of shape $(C_\text{out},)$: each output channel has its own 3-D kernel that spans *all* input channels, and

$$y[n, o, r, c] = b[o] + \sum_{m=0}^{C_\text{in}-1}\sum_{i, j} w[o, m, i, j]\; x[n, m, r s + i d,\; c s + j d].$$

two numbers have appeared. the **stride** $s$ is how far the window jumps between outputs ($s = 2$ halves the resolution). the **dilation** $d$ is the spacing between the kernel's taps: a $3 \times 3$ kernel with $d = 2$ reads a $5 \times 5$ area with holes, seeing farther without more weights. and before any of this, the input may be **padded** with $p$ rows and columns of zeros on every side, so windows can be centred on the border pixels.

how many outputs fit along one axis of length $n$? a dilated kernel spans $d(k - 1) + 1$ pixels. the padded axis has $n + 2p$ pixels, so the first window starts at 0 and the last possible start is $n + 2p - d(k - 1) - 1$; starts are multiples of $s$, so

$$n_\text{out} = \left\lfloor \frac{n + 2p - d(k - 1) - 1}{s} \right\rfloor + 1.$$

with $k = 3$, $p = 1$, $s = 1$ the size is unchanged ("same" padding); with $s = 2$ it halves (rounding up for odd $n$). pixels past the last window start are silently ignored.

the definition translates directly into loops: pad, then for every output position $(r, c)$, cut out the window `xp[:, :, r*s : r*s + d*(kh-1)+1 : d, c*s : ... : d]`, of shape $(N, C_\text{in}, k_h, k_w)$, and contract it with the weight over the last three axes (`np.tensordot(patch, w, axes=([1, 2, 3], [1, 2, 3]))` gives the $(N, C_\text{out})$ outputs for that position at once).

> **▶ build: stage 1.** implement `conv_output_size` and `conv2d_naive` (loop over output pixels; vectorize over the batch and channels). a helper `as_pair` turns an int or a pair into `(a, b)`, so `kernel_size`, `stride`, `padding` and `dilation` can each be either.

> **🔮 predict:** your naive version loops over $64 \times 64 = 4{,}096$ output positions for a $64 \times 64$ image and does one `tensordot` per position. a version that does exactly the same multiplications as one big matrix product: how many times faster will it be?

### idea 2: im2col, every window as a column

the naive loop does a small, identical computation at thousands of positions. the classic fix is to rearrange the input so that all of them become **one** matrix product. lay every window out as a column: for each image, build a matrix whose column $\ell$ holds the $C_\text{in} k_h k_w$ numbers under the kernel when it sits at output position $\ell$. that's **im2col** ("image to columns"), with output shape $(N,\; C_\text{in} k_h k_w,\; H_\text{out} W_\text{out})$.

the row order is a convention, and it has to match how you flatten the weight. use $(c, i, j)$: row $c \cdot k_h k_w + i \cdot k_w + j$ holds tap $(i, j)$ of channel $c$, which is exactly the order of `w.reshape(C_out, -1)`'s columns (and of PyTorch's `F.unfold`). output positions run in row-major order.

how to build it without looping over output pixels: notice that **row $(c, i, j)$, viewed as an $H_\text{out} \times W_\text{out}$ image, is just a strided slice of the padded input**: tap $(i, j)$ of every window is pixel $(r s + i d, c s + j d)$, so the whole row is `xp[:, c, i*d : i*d + s*(H_out-1)+1 : s, j*d : ... : s]`. loop over the $k_h k_w$ kernel offsets (9 iterations for a $3 \times 3$ kernel), copy one slice per offset for all images and channels at once into an array of shape $(N, C, k_h, k_w, H_\text{out}, W_\text{out})$, and reshape. a few dozen Python lines run, however large the image.

im2col is a **linear map** from images to column matrices, so it has a **transpose** (an adjoint), which you'll need for the backward pass. the adjoint is defined by

$$\langle \operatorname{im2col}(x), C\rangle = \langle x, \operatorname{col2im}(C)\rangle \quad \text{for all } x, C,$$

and since im2col *copies* each pixel into every column whose window covers it, its adjoint **adds** every column entry back into the pixel it came from. overlapping windows sum. **col2im** is the same loop over kernel offsets with `+=` into a zero padded image, then the padding is cut off (entries that landed in the padding came from zeros and have nowhere to go). `col2im(im2col(ones))` is therefore not `ones` but a count of how many windows cover each pixel: col2im is the transpose, not the inverse.

the price of im2col is memory: every pixel is copied into up to $k_h k_w$ columns, so the matrix is about $k^2$ times bigger than the input (9× for $3 \times 3$ kernels, 49× for $7 \times 7$).

> **▶ build: stage 2.** implement `im2col` and `col2im`, looping over kernel offsets only. the checks count the Python lines your code runs, so a loop over output pixels fails even if its answer is right.

### idea 3: convolution as one matrix product

with the columns in hand, the convolution is a matrix product. flatten the weight to $W \in \mathbb{R}^{C_\text{out} \times C_\text{in} k_h k_w}$; then for every image,

$$Y_n = W\, \operatorname{im2col}(x)_n + b\,\mathbf{1}^\top \;\in\; \mathbb{R}^{C_\text{out} \times H_\text{out} W_\text{out}},$$

and a reshape to $(N, C_\text{out}, H_\text{out}, W_\text{out})$ finishes the job. in NumPy, `W @ cols` with `cols` of shape $(N, K, L)$ broadcasts over the batch.

this is not a trick for toy code. it's how convolution was first put on GPUs (Chellapilla et al., 2006) and how cuDNN's GEMM-based algorithms work: matrix multiplication is the most optimized routine in numerical computing, so recasting convolution as one buys decades of tuning for free. it also makes the cost easy to count: one output needs $C_\text{in} k_h k_w$ multiply–adds, so a layer costs

$$N \cdot C_\text{out} \cdot H_\text{out} W_\text{out} \cdot C_\text{in} k_h k_w \quad \text{multiply–accumulates (MACs)},$$

about $2\times$ that in FLOPs. modern libraries pick among several algorithms per layer (im2col + GEMM, "implicit GEMM" that never materializes the matrix, Winograd, FFT), which is part of why PyTorch can beat you in the lab.

> **▶ build: stage 3.** implement `conv2d` as `im2col` plus one matrix product. it must agree with `conv2d_naive` and with `torch.nn.functional.conv2d`.

### idea 4: the backward pass

a convolutional layer is only useful if gradients flow through it. thanks to idea 3, its backward pass is the backward pass of a matrix product, which you derived in chapter 14. write $X_n = \operatorname{im2col}(x)_n$ and $G_n = \partial L/\partial Y_n$ (the upstream gradient `dout`, reshaped to $C_\text{out} \times H_\text{out} W_\text{out}$). from $Y_n = W X_n + b\mathbf{1}^\top$:

$$\frac{\partial L}{\partial W} = \sum_n G_n X_n^\top, \qquad \frac{\partial L}{\partial b} = \sum_n G_n \mathbf{1}, \qquad \frac{\partial L}{\partial X_n} = W^\top G_n.$$

the weight's gradient sums over the batch and over every position the kernel visited (weight sharing means one weight collects gradient from everywhere). the bias's gradient sums `dout` over the batch and both spatial axes. and the input's gradient goes through the column matrix: $W^\top G_n$ is the gradient with respect to the *columns*, and since the columns were made from $x$ by the linear map im2col, the chain rule applies im2col's transpose:

$$\frac{\partial L}{\partial x} = \operatorname{col2im}\!\left(W^\top G\right).$$

that's the whole backward pass: `im2col` once, two matrix products (one summed over the batch: `np.einsum("nol,nkl->ok", G, X)` does it in a single call), and `col2im`.

what does $\partial L/\partial x$ look like as an operation on images? each output pixel's gradient is spread back over the window it came from, weighted by the kernel: for stride 1 that's a cross-correlation of the (zero padded) `dout` with the kernel **flipped** in both directions and with its in/out channels swapped. libraries call it a **transposed convolution** (on paper 2), and it's the same layer U-Nets use for upsampling in topic 06.

> **🔮 predict:** your `conv2d_backward` returns `dw` for a $3 \times 3$ kernel applied to a $96 \times 96$ image. how many terms are summed into each entry of `dw`, per image?

> **▶ build: stage 4.** implement `conv2d_backward(dout, x, w, ...)`, returning `(dx, dw, db)`. it's checked against PyTorch's autograd and finite differences.

## worked example

take the $4 \times 4$ image $x[r, c] = 4r + c$ (the numbers 0 to 15 row by row) and the $2 \times 2$ kernel $w = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$, which subtracts each pixel's lower-right neighbour from it.

**no padding, stride 1.** $n_\text{out} = \lfloor (4 - 1 - 1)/1 \rfloor + 1 = 3$, and every output is $x[r, c] - x[r + 1, c + 1] = (4r + c) - (4r + 4 + c + 1) = -5$: a $3 \times 3$ block of $-5$s. (a flipped kernel, true convolution, would give $+5$s: the one test that tells the two apart.)

**padding 1, stride 2, bias 2.** $n_\text{out} = \lfloor (4 + 2 - 1 - 1)/2 \rfloor + 1 = 3$. window $(r, c)$ starts at padded position $(2r, 2c)$, i.e. at original pixel $(2r - 1, 2c - 1)$, and outputs $x[2r - 1, 2c - 1] - x[2r, 2c] + 2$ with out-of-range pixels read as 0:

$$y = \begin{pmatrix} 0 - 0 + 2 & 0 - 2 + 2 & 0 + 2 \\ 0 - 8 + 2 & 5 - 10 + 2 & 7 + 2 \\ 0 + 2 & 13 + 2 & 15 + 2 \end{pmatrix} = \begin{pmatrix} 2 & 0 & 2 \\ -6 & -3 & 9 \\ 2 & 15 & 17 \end{pmatrix}.$$

the bottom-right output reads pixel 15 and a padding zero: padding lets windows hang over the edge, and the edge pixels get a different, one-sided treatment, which is why convolutional networks are not exactly translation-equivariant near the border (topic 02).

**im2col.** without padding, column 0 of `im2col(x, 2)` is the top-left window read row by row, $(0, 1, 4, 5)$, and row 0 (the top-left tap of every window) is $(0, 1, 2, 4, 5, 6, 8, 9, 10)$. `w.reshape(1, -1)` is $(1, 0, 0, -1)$, and its product with column 0 is $0 - 5 = -5$, as before.

**parameters and cost.** a $3 \times 3$ convolution from 64 to 128 channels has $128 \cdot 64 \cdot 9 = 73{,}728$ weights and 128 biases, for any image size. on $32 \times 32$ images with padding 1 it does $128 \cdot 32 \cdot 32 \cdot 64 \cdot 9 \approx 75.5$ million MACs per image; on $224 \times 224$ images, 49 times more. a dense layer from $64 \times 32 \times 32$ inputs to $128 \times 32 \times 32$ outputs would have $65{,}536 \times 131{,}072 \approx 8.6$ billion weights.

the tests check both output grids and the im2col layout above.

## on paper

1. **the output size.** derive $n_\text{out}$ for padding $p$, stride $s$, dilation $d$. which input sizes give exactly $n_\text{out} = n/s$ with $k = 3$, $p = 1$, $d = 1$?

<details><summary>solution</summary>

the padded axis has $n + 2p$ pixels, indexed $0, \dots, n + 2p - 1$. a window starting at $t$ reads pixels $t, t + d, \dots, t + d(k - 1)$, so it fits when $t + d(k - 1) \le n + 2p - 1$. starts are $t = 0, s, 2s, \dots$, so the number of starts is $\lfloor (n + 2p - d(k - 1) - 1)/s \rfloor + 1$. with $k = 3$, $p = 1$, $d = 1$: $\lfloor (n - 1)/s \rfloor + 1 = \lceil n/s \rceil$, which equals $n/s$ exactly when $s$ divides $n$. that's why vision architectures use image sizes divisible by large powers of 2.

</details>

2. **the input gradient is a transposed convolution.** in 1-D with one channel, stride 1 and no padding, $y[r] = \sum_{i=0}^{k-1} w[i]\, x[r + i]$ for $r = 0, \dots, n - k$. show that $\partial L/\partial x[t] = \sum_i w[i]\, \bar y[t - i]$, and write it as a cross-correlation of a zero-padded $\bar y$ with the flipped kernel.

<details><summary>solution</summary>

$x[t]$ appears in $y[r]$ exactly when $t = r + i$, i.e. $r = t - i$, with coefficient $w[i]$. so $\partial L/\partial x[t] = \sum_i w[i]\,\bar y[t - i]$, where $\bar y[r] = 0$ outside $0 \le r \le n - k$. substituting $i' = k - 1 - i$ and $\tilde w[i'] = w[k - 1 - i']$ (the flipped kernel): $\sum_{i'} \tilde w[i']\, \bar y[t - k + 1 + i']$, which is the cross-correlation of $\bar y$, zero-padded by $k - 1$ on each side, with $\tilde w$. the output has length $(n - k + 1) + 2(k - 1) - k + 1 = n$, as it must. with several channels, the kernel also swaps its input and output channel axes; with stride $s$, zeros are inserted between the entries of $\bar y$ first.

</details>

3. **what im2col costs.** a ResNet layer applies a $3 \times 3$ convolution, 64 → 64 channels with padding 1, to a batch of 128 images of size $32 \times 32$, in float32. how big is the input, how big is the im2col matrix, and how many MACs does the matrix product do?

<details><summary>solution</summary>

the input has $128 \cdot 64 \cdot 32 \cdot 32 = 8.4$ million numbers, 33.5 MB. the column matrix has $128 \cdot (64 \cdot 9) \cdot 1{,}024 = 75.5$ million, 302 MB, nine times more. the product does $128 \cdot 64 \cdot 1{,}024 \cdot 576 \approx 4.8$ billion MACs. one layer's scratch matrix being ten times the size of its input is why libraries use implicit GEMM (computing the column entries on the fly inside the matrix-multiply kernel) instead of materializing it.

</details>

4. **dilation as a sparse kernel.** show that a $3 \times 3$ kernel with dilation 2 computes the same thing as a $5 \times 5$ kernel with dilation 1 whose entries are zero except at even positions. how many weights does each have, and how many input pixels does each output see?

<details><summary>solution</summary>

the dilated kernel reads $x[r + 2i, c + 2j]$ for $i, j \in \{0, 1, 2\}$. a $5 \times 5$ kernel $v$ with $v[2i, 2j] = w[i, j]$ and zeros elsewhere reads $x[r + i', c + j']$ for $i', j' \in \{0, \dots, 4\}$, and only the even $(i', j')$ contribute, with the same weights: identical outputs. the dilated kernel has 9 weights, the sparse one 25 entries (16 forced to zero); both span a $5 \times 5$ area but use only 9 of its 25 pixels. stacking dilations $1, 2, 4, \dots$ grows the span exponentially with depth (topic 02).

</details>

## experiment

`lab.py` runs your convolution three ways:

1. your loops, your im2col version and PyTorch's `conv2d`, timed across image sizes, with im2col's memory cost.
2. a tiny network trained on MNIST with your `conv2d` and `conv2d_backward`; its learned filters next to the hand-made ones from chapter 00.
3. the same network with its filters learned, frozen at your hand-made ones, or frozen at random noise.

## check your understanding

1. why is a convolutional layer's parameter count independent of the image size, and what is the price of that?

<details><summary>answer</summary>

the same kernel is applied at every position (weight sharing), so the parameters are $C_\text{out} C_\text{in} k_h k_w + C_\text{out}$ regardless of $H$ and $W$. the price is an assumption: that the same feature detector is useful everywhere in the image (translation equivariance), and that each output only needs a local neighbourhood. the computation still grows with the image size.

</details>

2. a learner's `conv2d` matches `conv2d_naive` but not `F.conv2d`, and the error disappears for symmetric kernels. what's wrong?

<details><summary>answer</summary>

both of their functions flip the kernel (true convolution), or both read the window in the wrong orientation. a kernel symmetric under a 180° rotation is unchanged by the flip, so the bug is invisible for it. the worked example's kernel $\begin{pmatrix}1 & 0\\0 & -1\end{pmatrix}$ distinguishes the two: $-5$ for cross-correlation, $+5$ for convolution.

</details>

3. why must im2col's row order be $(c, i, j)$ in this course's code, and what goes wrong if it's $(i, j, c)$?

<details><summary>answer</summary>

`w.reshape(C_out, -1)` flattens the kernel in $(c, i, j)$ order (the last axes vary fastest), and the matrix product pairs column $k$ of $W$ with row $k$ of the column matrix. with rows in $(i, j, c)$ order, each weight multiplies the wrong tap or channel: the result is still a valid-looking linear function of the input, just not this convolution, and only an oracle test catches it.

</details>

4. col2im is called the adjoint of im2col, not its inverse. what does `col2im(im2col(x))` return?

<details><summary>answer</summary>

each pixel times the number of windows covering it: for a $3 \times 3$ kernel with padding 1, interior pixels are multiplied by 9, edges by 6, corners by 4. im2col copies a pixel into several columns; its transpose adds those copies back. an inverse would have to average them instead (and wouldn't exist for strides larger than the kernel, where some pixels are never read).

</details>

5. which pixels of the input get zero gradient from a convolution with kernel 2, stride 3, no padding?

<details><summary>answer</summary>

windows start at $0, 3, 6, \dots$ and cover two pixels each, so pixels $2, 5, 8, \dots$ (positions $\equiv 2 \bmod 3$) are never read and get zero gradient. any stride larger than the dilated kernel span skips pixels entirely; col2im, which only adds to positions the windows touched, gives this for free.

</details>

## common bugs

- **flipping the kernel.** symptom: matches a hand-written convolution, disagrees with PyTorch on asymmetric kernels (and gives $+5$ instead of $-5$ on the worked example). deep learning uses cross-correlation.
- **mismatched flattening orders.** symptom: `conv2d` is a plausible linear map but disagrees with `F.conv2d`. im2col's rows and `w.reshape(C_out, -1)`'s columns must both be $(c, i, j)$.
- **assigning instead of adding in col2im.** symptom: the adjoint test fails as soon as windows overlap (kernel larger than the stride); gradients near overlaps are too small.
- **forgetting the dilation in the window's extent.** symptom: correct for $d = 1$, wrong (or a shape error) for $d > 1$. the window spans $d(k - 1) + 1$ pixels, and its taps are $d$ apart.
- **padding only one side, or cropping wrong in col2im.** symptom: outputs shifted by $p$ pixels or the wrong size; `dx` shifted relative to autograd's.
- **looping over output pixels.** symptom: correct but hundreds of times slower than it should be on large images (and the stage checks fail the line budget). loop over the kernel's offsets, never over the image.

## going deeper

- Dumoulin & Visin (2016), "A guide to convolution arithmetic for deep learning". padding, strides, dilation and transposed convolutions, with pictures. [arXiv:1603.07285](https://arxiv.org/abs/1603.07285)
- LeCun et al. (1989), "Backpropagation applied to handwritten zip code recognition", *Neural Computation* 1(4): 541–551. the first trained convolutional network. [doi:10.1162/neco.1989.1.4.541](https://doi.org/10.1162/neco.1989.1.4.541)
- Chellapilla, Puri & Simard (2006), "High performance convolutional neural networks for document processing", *10th International Workshop on Frontiers in Handwriting Recognition*. convolution as unrolling plus matrix multiplication. [inria.hal.science/inria-00112631](https://inria.hal.science/inria-00112631)
- Chetlur et al. (2014), "cuDNN: efficient primitives for deep learning". GEMM-based and implicit-GEMM convolution on GPUs. [arXiv:1410.0759](https://arxiv.org/abs/1410.0759)
- Lavin & Gray (2016), "Fast algorithms for convolutional neural networks", *CVPR*. Winograd convolution. [arXiv:1509.09308](https://arxiv.org/abs/1509.09308)
- Goodfellow, Bengio & Courville (2016), *Deep Learning*, chapter 9 (convolutional networks). [deeplearningbook.org](https://www.deeplearningbook.org)

## connections

- ← builds on: chapter 00's capstone (`correlate2d`, hand-made filters), vectorization and strided slicing (ch 00 · topic 02), VJPs of matrix products (ch 14 · topic 03), PyTorch conventions and float64 comparisons (ch 15 · topic 07).
- → used in: pooling, receptive fields and equivariance (topic 02); every convolutional architecture in this chapter via `nn.Conv2d`, now unlocked (topics 03–06); transposed convolutions in U-Nets (topic 06); patch embeddings in vision transformers (ch 18); convolutional encoders and decoders in VAEs, GANs and diffusion models (ch 21–22); FLOP counting and kernels (ch 26).
