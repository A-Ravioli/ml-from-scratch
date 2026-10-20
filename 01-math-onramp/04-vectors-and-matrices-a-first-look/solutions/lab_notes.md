# vectors & matrices, a first look · lab notes

## experiment 1: what a matrix does to the unit circle

**what people usually predict:** a bigger circle, or an ellipse stretched along the $x$- or $y$-axis, often with radii 2 and 1 or 3 and 2 (read off the matrix entries).

**what happens:** a tilted ellipse, leaning at $45°$, with longest radius **3.000** (along the diagonal $(1, 1)$) and shortest radius **1.000** (along the other diagonal, $(1, -1)$). the area of the image is **3.000** times the circle's area, and so is `det2(A)` and the product of the radii. as you move the sliders, "longest × shortest" always equals the area factor, and the (signed) area factor always equals $\det A$ itself: its size is $|\det A|$, and it's negative exactly when $\det A$ is. when $\det A < 0$ the image is still an ellipse, but the arrows for $Ae_1$ and $Ae_2$ swap their order around the circle (the plane has been flipped over), and the signed area comes out negative. when $\det A = 0$ (for example $a = d = 1$, $b = c = 1$), the ellipse collapses to a line segment.

**why:** $A(1, 1) = (3, 3) = 3 \cdot (1, 1)$ and $A(1, -1) = (1, -1)$: those two directions don't turn at all, they're only stretched, by 3 and by 1. (directions like these are **eigenvectors**, and 3 and 1 are **eigenvalues**: chapter 02, topic 05.) every other unit vector is a mix of these two directions, so its image is the same mix of a length-3 and a length-1 arrow, which traces out an ellipse. in general (non-symmetric $A$) the ellipse's axes aren't eigenvectors, but there's always a longest and a shortest direction, perpendicular to each other: the **singular values** and singular vectors of chapter 02, topic 06. the area factor is the product of the two radii because an ellipse with radii $r_1, r_2$ has area $\pi r_1r_2$. and it equals $|\det A|$ because a matrix scales *every* area by the same factor: the circle is no different from the unit square.

## experiment 2: does the order matter?

**what people usually predict:** "a rotation and a shear are both simple, so the result is the same either way", or a vague sense that the pictures will differ without knowing how.

**what happens:** the two products are different matrices: $SR = \begin{pmatrix}1&-1\\1&0\end{pmatrix}$ sends $e_1$ to $(1, 1)$, and $RS = \begin{pmatrix}0&-1\\1&1\end{pmatrix}$ sends $e_1$ to $(0, 1)$. the two ellipses have exactly the same shape (both have radii $1.618$ and $0.618$, and area factor 1) but they're tilted differently.

**why:** follow $e_1$. for $SR$, the rotation acts first and turns $(1, 0)$ into $(0, 1)$; the shear then slides it sideways by its height, to $(1, 1)$. for $RS$, the shear acts first and leaves $(1, 0)$ alone (it's on the $x$-axis); the rotation then turns it to $(0, 1)$. the matrix on the right acts first because $(AB)v = A(Bv)$: $B$ touches $v$ first. the shapes match because rotating before or after only changes *which* directions get stretched, not *how much* (a rotation doesn't change lengths). in general, matrix multiplication is not commutative, and in a neural network the order of layers is part of what the network computes.

## experiment 3: stretching without growing

**what people usually predict:** the shear changes area the most (it "looks" the most distorted) or the scaling does (it stretches by 2). the parallel-columns matrix gets ranked somewhere in the middle. guesses for the shear's radii are usually 1 and $\sqrt2 \approx 1.41$.

**what happens:**

| matrix | longest radius | shortest radius | area factor | det |
|---|---|---|---|---|
| shear $\begin{pmatrix}1&1\\0&1\end{pmatrix}$ | 1.618 | 0.618 | **1.000** | 1 |
| scaling $\begin{pmatrix}2&0\\0&0.5\end{pmatrix}$ | 2.000 | 0.500 | **1.000** | 1 |
| parallel columns $\begin{pmatrix}1&2\\0.5&1\end{pmatrix}$ | 2.500 | ≈ 0 | **0.000** | 0 |

the shear and the scaling both keep every area exactly the same, even though both visibly stretch the circle. the shear's radii are $\varphi = 1.618\ldots$ and $1/\varphi = 0.618\ldots$, the **golden ratio** and its reciprocal, whose product is exactly 1. the third matrix squashes the whole plane onto a line: its image is a segment of half-length 2.5 (the measured shortest radius is a few thousandths rather than exactly 0 only because we sampled the circle at 2000 points).

**why:** a shear slides each horizontal slice of the plane sideways, by an amount proportional to its height, like pushing a deck of cards. no slice changes length or thickness, so no area changes, and $\det = 1 \cdot 1 - 1 \cdot 0 = 1$. stretching one direction while squeezing another can leave area unchanged; the determinant only measures the *net* effect, the product of the two radii. the golden ratio comes from the stretch factors of the shear: they're the square roots of the eigenvalues of $S^\top S = \begin{pmatrix}1&1\\1&2\end{pmatrix}$, which are $\frac{3 \pm \sqrt5}{2} = \varphi^2$ and $\varphi^{-2}$. for the third matrix, the second column is twice the first, so every output $v_1(1, 0.5) + v_2(2, 1) = (v_1 + 2v_2)(1, 0.5)$ lies on the line through $(1, 0.5)$. infinitely many inputs share each output, so no matrix can undo it: $\det = 0$ is exactly the condition for "can't be inverted", and in chapter 02 it's the condition for a linear system to have no unique solution.
