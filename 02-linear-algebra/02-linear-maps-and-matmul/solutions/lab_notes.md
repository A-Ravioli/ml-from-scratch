# linear maps & matmul · lab notes

## experiment 1: your matmuls against BLAS

**what people usually predict:** BLAS is "a lot" faster than the triple loop, maybe 10–100×, and the four of your versions are about the same, since they do the same arithmetic.

**what happens** (numbers vary by machine; these are typical of a laptop): at n = 128, the triple loop runs at about 0.01 GFLOP/s, **about 10,000–30,000× slower than `np.matmul`**. the column and row versions run around 0.2 GFLOP/s (roughly 20× faster than the triple loop), the outer-product version 2–3 GFLOP/s, the blocked version 10–15 GFLOP/s, and BLAS 100–300+ GFLOP/s. BLAS's GFLOP/s *grows* with n over this range: at small n its fixed per-call overhead and thread start-up dominate, and only bigger products amortize them and keep all cores busy. the triple loop's speed is flat.

**why:** three effects stack.
1. **interpreter overhead.** each Python-level step costs about 50–100 ns, and much more in a triple loop of scalar indexing. the triple loop takes $n^3$ steps that each do one multiply-add. the column and row versions take $n^2$ steps that each do $n$, outer products take $n$ steps that each do $n^2$, and the blocked version $(n/b)^3$ steps that each do $b^3$. the closer you get to "one big operation", the more the overhead is amortized. this is the vectorization lesson of chapter 00, measured.
2. **memory traffic.** `np.outer` and `C += ...` stream the whole $n \times n$ matrix through memory on every step. a tile product touches small blocks that stay in cache, which is why `blocked` beats `outer` even though both are vectorized.
3. **the kernel itself.** BLAS uses CPU vector (SIMD) instructions doing 4–16 multiply-adds per instruction, fused multiply-add units, carefully tuned register and cache blocking, and all your CPU cores. that gets within a small factor of the chip's peak. no amount of NumPy-level cleverness reaches that, which is why you use `@` from now on. the build-before-you-use rule says you've earned it.

## experiment 2: where you put the parentheses

**what people usually predict:** they're the same computation, so they take the same time. or maybe a modest difference.

**what happens:** identical answers (relative difference around $10^{-15}$, pure rounding). but $A(Bx)$ is **tens of times faster** at n = 1500, and the gap grows with n. the flop counts differ by a factor of about $n/2 \approx 750$. the measured speedup is smaller than that, typically 20–100×.

**why:** $(AB)x$ first builds the $n \times n$ product ($2n^3$ flops), while $A(Bx)$ does two matrix-vector products ($4n^2$ flops). the measured ratio falls short of the flop ratio because a matrix-vector product is **memory-bound**: it reads each of the $n^2$ matrix entries once and does one multiply-add with it, so it runs at memory bandwidth, not at arithmetic peak. a matrix-matrix product reuses each entry about $n$ times, so it runs close to peak. the lesson that runs through the rest of the course: **multiply into vectors when you can**. reverse-mode autodiff (chapter 14) computes $v^\top J$ by pushing a vector backwards through each layer, never forming a Jacobian, and that's this experiment at scale.

## experiment 3: composing maps of the plane

**what people usually predict:** "it's the same two operations", so the same picture. and for the projection, that "the circle becomes a smaller circle" or "a point".

**what happens:** for a nonzero shear and a rotation that isn't 0° or 180°, $RS$ and $SR$ produce different parallelograms (at 180°, $R = -I$ commutes with everything), and their matrices differ. both have determinant exactly 1: same area, different shape and orientation. the projection squashes the square and the circle onto a segment along $u$ (the circle becomes a doubly covered segment of length 2), with determinant 0, and every vector along the dashed perpendicular line lands on the origin.

**why:** $RS$ means "shear first, then rotate". the shear acts in the original horizontal direction, while in $SR$ it acts after the rotation, on rotated vectors. composition order matters, and matrix multiplication records it faithfully. determinants multiply, $\det(RS) = \det R\det S = 1 \cdot 1 = \det(SR)$: area scaling doesn't care about order, even though the maps do. for $P = uu^\top$, rank–nullity says $1 + 1 = 2$: the image is the line through $u$ (rank 1), and the kernel is the perpendicular line (nullity 1). every point of the plane slides perpendicularly onto the line, so the unit circle covers the segment from $-u$ to $u$ twice. that's the picture behind least squares (topic 04): the fit is the projection of the data onto the column space, and the residual lives in the perpendicular directions the projection throws away.
