# vectors & matrices, a first look

## why this matters

machine learning runs on lists of numbers. a house is described by its size, age and number of rooms: $(120, 35, 4)$. a $28 \times 28$ grayscale image is a list of 784 brightness values. a word, inside a language model, is a list of a few thousand numbers. those lists are **vectors**, and once your data is vectors you can ask geometric questions about it: how far apart are two houses? how *similar* are two words? (a search engine or a chatbot answers that one with the dot product you'll build in stage 1.)

a model, in turn, is mostly made of **matrices**: tables of numbers that turn one vector into another. a layer of a neural network takes a vector of 784 pixel values and produces a vector of 100 features by multiplying by a $100 \times 784$ matrix. to understand what a model does, it helps enormously to see what a matrix *does*, and the cleanest place to see it is the flat plane, with $2 \times 2$ matrices. every $2 \times 2$ matrix moves the plane around: rotating it, stretching it, shearing it, flipping it. in this topic you build the operations from scratch and a visualizer that shows any matrix's effect at a glance. chapter 02 then builds the full theory on top of these pictures.

## warm-up: predict first

1. the vectors $(3, 4)$ and $(-4, 3)$: what angle is between them?
2. what's the length of the vector $(1, 1, 1, \ldots, 1)$ with 100 entries?
3. a matrix rotates by $90°$ and another stretches the $x$-direction by 2. does it matter which you do first?
4. the matrix $\begin{pmatrix}2&1\\1&2\end{pmatrix}$ is applied to every point of the unit circle. what shape comes out?
5. a matrix doubles the length of every horizontal arrow and leaves vertical arrows alone. by how much does it change the area of a circle?

## prerequisites check

- [ ] you can plot points $(x, y)$ and know Pythagoras' theorem (high school geometry)
- [ ] you know $\sin$ and $\cos$ on the unit circle: the point at angle $\theta$ is $(\cos\theta, \sin\theta)$ (high school trigonometry)
- [ ] you can index NumPy arrays: `A[:, j]` is column $j$, `A[i, :]` is row $i$ (→ ch 00 · topic 02)
- [ ] you can make a matplotlib plot (→ ch 00 · topic 04)

## core ideas

### idea 1: vectors, and the dot product

a **vector** is an ordered list of numbers, $v = (v_1, v_2, \ldots, v_n)$. in two dimensions you can draw it as an arrow from the origin to the point $(v_1, v_2)$. vectors add entry by entry (put the arrows head to tail) and scale entry by entry ($2v$ is the same arrow, twice as long; $-v$ points the other way).

the **dot product** of two vectors of the same length multiplies matching entries and adds:

$$u \cdot v = u_1v_1 + u_2v_2 + \dots + u_nv_n.$$

for example $(1, 2, 3) \cdot (4, -5, 6) = 4 - 10 + 18 = 12$. it takes two vectors and gives one number. it's the workhorse of ML: a linear model's prediction is a dot product between a weight vector and a feature vector, $w \cdot x$ (topic 06 and chapter 08).

the dot product of a vector with *itself* is the sum of its squared entries, which by Pythagoras is its squared length. so the **length** (or **norm**) of a vector is

$$\|v\| = \sqrt{v \cdot v} = \sqrt{v_1^2 + v_2^2 + \dots + v_n^2}.$$

$\|(3, 4)\| = \sqrt{9 + 16} = 5$. in more dimensions, Pythagoras just keeps going: the vector of 100 ones has length $\sqrt{100} = 10$ (warm-up 2). the distance between two points is the length of their difference, $\|u - v\|$.

> **▶ build: stage 1.** implement `dot(u, v)` (raising `ValueError` for different lengths) and `norm(v)`. no `@` or `np.dot`: multiply entry by entry and sum.

### idea 2: angles and shadows

the dot product also measures **angles**. for any two vectors,

$$u \cdot v = \|u\|\,\|v\|\cos\theta,$$

where $\theta$ is the angle between them (you'll prove it on paper from the law of cosines). so

$$\theta = \arccos\frac{u \cdot v}{\|u\|\,\|v\|}.$$

the ratio $\frac{u \cdot v}{\|u\|\|v\|}$ is the **cosine similarity**: $1$ when the vectors point the same way, $0$ when they're perpendicular, $-1$ when they point in opposite directions. it ignores length and keeps only direction, and it's exactly how ML systems compare word or document vectors. the most useful special case: **$u \cdot v = 0$ means $u$ and $v$ are perpendicular**. $(3, 4) \cdot (-4, 3) = -12 + 12 = 0$, so that angle is $90°$ (warm-up 1). angles here are in **radians**: a full turn is $2\pi$, so $90° = \pi/2$.

one numerical catch: rounding can make the ratio come out as $1.0000000000000002$, and $\arccos$ of anything above 1 is undefined (NumPy gives `nan`). clip it into $[-1, 1]$ first.

**projection.** shine a light straight down onto the line through $v$. the shadow of $u$ is its **projection** onto $v$: the multiple of $v$ closest to $u$,

$$\operatorname{proj}_v u = \frac{u \cdot v}{v \cdot v}\,v.$$

dividing by $v \cdot v$ (not by $\|v\|$) makes the answer independent of how long $v$ is. what's left over, $u - \operatorname{proj}_vu$, is perpendicular to $v$. that "closest point, with a perpendicular leftover" picture is least squares in miniature, and it comes back in chapter 02 (topic 04) and chapter 08.

> **▶ build: stage 2.** implement `angle(u, v)` (in radians, clipped) and `project(u, v)`.

### idea 3: a matrix moves vectors

a **matrix** is a rectangular table of numbers, $m$ rows by $n$ columns. what makes matrices interesting is what they *do*: a matrix $A$ turns a vector $v$ with $n$ entries into a new vector $Av$ with $m$ entries. the best way to see it is through the columns of $A$:

$$Av = v_1\,(\text{column } 1) + v_2\,(\text{column } 2) + \dots + v_n\,(\text{column } n).$$

$Av$ is a **combination of the columns**, with weights taken from $v$. for example

$$\begin{pmatrix}1&2\\3&4\end{pmatrix}\begin{pmatrix}10\\1\end{pmatrix} = 10\begin{pmatrix}1\\3\end{pmatrix} + 1\begin{pmatrix}2\\4\end{pmatrix} = \begin{pmatrix}12\\34\end{pmatrix}.$$

(you may have learned it as "row times column": entry $i$ of $Av$ is the dot product of row $i$ with $v$. it's the same arithmetic, organized differently.) the column view has a wonderful consequence. the standard basis vectors $e_1 = (1, 0)$ and $e_2 = (0, 1)$ pick out single columns: $Ae_1$ is column 1 and $Ae_2$ is column 2. **the columns of a matrix are where the basis vectors land.** and since every vector is a combination $v = v_1e_1 + v_2e_2$, its image is the same combination of the landed columns. so a $2 \times 2$ matrix is completely described by where it sends two arrows.

**doing one matrix after another.** apply $B$ to $v$, then $A$ to the result: $A(Bv)$. that combined transformation is itself a matrix, the product $AB$. its columns are where the basis vectors land after both steps, so

$$\text{column } j \text{ of } AB = A\,(\text{column } j \text{ of } B).$$

that's how you'll compute it. $AB$ means "first $B$, then $A$" (the same right-to-left order as composing functions, topic 01). and order matters: in general $AB \ne BA$ (warm-up 3).

> **🔮 predict:** $A = \begin{pmatrix}1&1\\0&1\end{pmatrix}$ and $B = \begin{pmatrix}0&-1\\1&0\end{pmatrix}$. without multiplying, which of $AB$ and $BA$ sends $e_1$ to $(1, 1)$?

> **▶ build: stage 3.** implement `matvec(A, v)` as a combination of columns, and `matmul(A, B)` one column at a time with your `matvec`.

### idea 4: a gallery of transformations

since a $2 \times 2$ matrix is decided by where $e_1$ and $e_2$ land, you can *design* matrices by asking where those two arrows should go.

- **rotation by $\theta$ (counterclockwise).** $e_1 = (1, 0)$ turns to $(\cos\theta, \sin\theta)$. $e_2 = (0, 1)$ is a quarter turn ahead of $e_1$, so it lands at $(\cos(\theta + 90°), \sin(\theta + 90°)) = (-\sin\theta, \cos\theta)$. those are the columns:
$$R_\theta = \begin{pmatrix}\cos\theta & -\sin\theta\\ \sin\theta & \cos\theta\end{pmatrix}.$$
- **scaling.** $e_1 \to (s_x, 0)$, $e_2 \to (0, s_y)$: a diagonal matrix $\begin{pmatrix}s_x&0\\0&s_y\end{pmatrix}$. a negative factor flips that axis, so $\begin{pmatrix}-1&0\\0&1\end{pmatrix}$ is a mirror.
- **shear.** $e_1$ stays put and $e_2 \to (k, 1)$: $\begin{pmatrix}1&k\\0&1\end{pmatrix}$. every point slides sideways by $k$ times its height, like pushing the top of a deck of cards.

composing them gives everything else. rotating by $\alpha$ and then by $\beta$ is rotating by $\alpha + \beta$, so $R_\beta R_\alpha = R_{\alpha + \beta}$ (a neat way to remember the angle-addition formulas from trigonometry). a remarkable fact you'll prove in chapter 02 (the singular value decomposition): *every* $2 \times 2$ matrix is a rotation (or a reflection), then a scaling along the axes, then another rotation (or reflection). a reflection is needed exactly when $\det A < 0$ (idea 6).

> **▶ build: stage 4.** implement `rotation(theta)`, `scaling(sx, sy)` and `shear(k)`.

### idea 5: seeing a transformation

to see what a matrix does, apply it to many points at once and look at the picture. store $n$ points as the **rows** of an $n \times 2$ array $P$. to transform them all, put them in columns (`P.T`, shape $2 \times n$), multiply by $A$, and transpose back: `matmul(A, P.T).T`.

the most revealing input is the **unit circle**, all the points $(\cos t, \sin t)$ for $t$ from $0$ to $2\pi$. every point on it is a unit vector, so its image shows how $A$ stretches *every direction at once*. for any matrix that doesn't collapse the plane, the image is an **ellipse** (warm-up 4). its longest radius is the most $A$ stretches any vector, its shortest radius the least, and those two numbers, the **singular values** of $A$, are among the most important numbers in all of numerical linear algebra (chapter 02, topic 06).

for $\begin{pmatrix}2&1\\1&2\end{pmatrix}$: $A(1, 1) = (3, 3)$, so the diagonal direction is stretched by 3, and $A(1, -1) = (1, -1)$, so the other diagonal isn't stretched at all. the circle becomes an ellipse with radii 3 and 1, tilted $45°$.

> **▶ build: stage 5.** implement `transform_points(A, P)` and the visualizer `plot_transform(A, *, ax=None, n=200)`: the unit circle (dashed), its image, and arrows for $Ae_1$ and $Ae_2$.

### idea 6: how a matrix changes area

every $2 \times 2$ matrix multiplies **every** area by the same factor. to find it, look at the unit square, with sides $e_1$ and $e_2$. it lands on the parallelogram with sides $(a, c)$ and $(b, d)$, the columns of $A = \begin{pmatrix}a&b\\c&d\end{pmatrix}$. that parallelogram's area is

$$\det A = ad - bc,$$

the **determinant**. (on paper, problem 3, you'll show this.) because the matrix acts the same way everywhere, any shape you can tile with tiny squares gets its area multiplied by $\det A$ too. some sizes: a rotation has $\det = \cos^2\theta + \sin^2\theta = 1$, a scaling has $\det = s_xs_y$ (warm-up 5: doubling one direction doubles every area), and a shear has $\det = 1$.

the determinant has a sign. **a negative determinant means the matrix flips the plane over**, like a mirror: counterclockwise turns into clockwise. and **$\det A = 0$ means $A$ squashes the plane flat** onto a line (or a point). then different inputs land on the same output, and the transformation can't be undone. that's the 2D picture of a matrix with no inverse, and of a linear system with no unique solution (chapter 02).

to check all this numerically you need the area of a polygon. the **shoelace formula** gives the signed area of a polygon with corners $(x_1, y_1), \ldots, (x_n, y_n)$ in order:

$$\text{area} = \frac12\sum_{i=1}^{n}(x_iy_{i+1} - x_{i+1}y_i),$$

where corner $n + 1$ means corner 1 again. it's positive when the corners go counterclockwise. a circle drawn with many points is a polygon, so you can measure how much a matrix stretches the circle's area and compare with $\det A$.

> **🔮 predict:** the shear $\begin{pmatrix}1&1\\0&1\end{pmatrix}$ turns the unit circle into a tilted ellipse. its longest radius is about 1.62. what's its shortest radius?

> **▶ build: stage 6.** implement `det2(A)` and `polygon_area(P)`.

## worked example

$A = \begin{pmatrix}2&1\\1&2\end{pmatrix}$.

- **columns:** $Ae_1 = (2, 1)$ and $Ae_2 = (1, 2)$, the arrows the visualizer draws.
- **matrix times vector:** $A\,(1, 1) = 1 \cdot (2, 1) + 1 \cdot (1, 2) = (3, 3)$, and $A\,(1, -1) = (2, 1) - (1, 2) = (1, -1)$.
- **angles:** $(3, 3)$ points the same way as $(1, 1)$ (angle 0), and $(1, -1)$ is unchanged. the angle between $(1, 1)$ and $(1, -1)$ is $\arccos\frac{1 - 1}{\sqrt2\sqrt2} = \frac\pi2$, so the two special directions are perpendicular.
- **the circle's image:** an ellipse with radii 3 (along $(1, 1)$) and 1 (along $(1, -1)$).
- **area:** $\det A = 2 \cdot 2 - 1 \cdot 1 = 3$. the unit circle (area $\pi$) becomes an ellipse with area $3\pi$, which is also $\pi \times 3 \times 1$, the area of an ellipse with radii 3 and 1.
- **order matters:** with $B = \begin{pmatrix}0&1\\1&0\end{pmatrix}$ (swap $x$ and $y$), $AB = \begin{pmatrix}1&2\\2&1\end{pmatrix}$ (columns of $A$ swapped) but $BA = \begin{pmatrix}1&2\\2&1\end{pmatrix}$ too, only because this $A$ is symmetric. with $\begin{pmatrix}1&2\\3&4\end{pmatrix}$ instead, $AB = \begin{pmatrix}2&1\\4&3\end{pmatrix}$ and $BA = \begin{pmatrix}3&4\\1&2\end{pmatrix}$: different.

the tests check the matrix-vector product $\begin{pmatrix}1&2\\3&4\end{pmatrix}(10, 1) = (12, 34)$, the column swap, $\det = 3$, and that a transformed 399-sided "circle" has its area multiplied by exactly $\det A$.

## on paper

1. **the dot product measures angles.** the law of cosines says that a triangle with sides $a$, $b$ and angle $\theta$ between them has third side $c$ with $c^2 = a^2 + b^2 - 2ab\cos\theta$. apply it to the triangle with sides $u$, $v$ and $u - v$, and expand $\|u - v\|^2 = (u - v)\cdot(u - v)$, to show $u \cdot v = \|u\|\|v\|\cos\theta$.

<details><summary>solution</summary>

$\|u - v\|^2 = (u - v)\cdot(u - v) = u\cdot u - 2u\cdot v + v\cdot v = \|u\|^2 + \|v\|^2 - 2u\cdot v$. the law of cosines says the same quantity is $\|u\|^2 + \|v\|^2 - 2\|u\|\|v\|\cos\theta$. comparing, $u \cdot v = \|u\|\|v\|\cos\theta$.

</details>

2. **rotations compose by adding angles.** multiply $R_\beta R_\alpha$ and use the angle-addition formulas to show it equals $R_{\alpha + \beta}$.

<details><summary>solution</summary>

the top-left entry is $\cos\beta\cos\alpha - \sin\beta\sin\alpha = \cos(\alpha + \beta)$, and the bottom-left is $\sin\beta\cos\alpha + \cos\beta\sin\alpha = \sin(\alpha + \beta)$. the second column works the same way, giving $(-\sin(\alpha + \beta), \cos(\alpha + \beta))$. so $R_\beta R_\alpha = R_{\alpha + \beta}$, and rotations in 2D commute (unlike most matrices).

</details>

3. **the determinant is an area.** the unit square lands on the parallelogram with corners $(0, 0)$, $(a, c)$, $(a + b, c + d)$, $(b, d)$. use the shoelace formula to show its signed area is $ad - bc$.

<details><summary>solution</summary>

the terms $x_iy_{i+1} - x_{i+1}y_i$ around the corners are: $0\cdot c - a \cdot 0 = 0$; $a(c + d) - (a + b)c = ad - bc$; $(a + b)d - b(c + d) = ad - bc$; $b \cdot 0 - 0 \cdot d = 0$. half their sum is $ad - bc$.

</details>

4. **a matrix you can't undo.** show that $\begin{pmatrix}1&2\\2&4\end{pmatrix}$ sends every vector onto the line through $(1, 2)$. what's its determinant? find two different vectors with the same image.

<details><summary>solution</summary>

the second column is twice the first, so $Av = v_1(1, 2) + v_2(2, 4) = (v_1 + 2v_2)(1, 2)$, always a multiple of $(1, 2)$. $\det = 4 - 4 = 0$. for example $(2, 0)$ and $(0, 1)$ both map to $(2, 4)$, so from the output you can't tell which input you started with.

</details>

## experiment

`lab.py` runs three experiments on your code:

1. what a matrix does to the unit circle, with four sliders for its entries, and the image's longest and shortest radius against its area.
2. a rotation and a shear, applied in both orders.
3. two matrices that look very different on screen: a shear, and a matrix whose columns are parallel. what happens to area?

## check your understanding

1. two vectors have a dot product of $-5$. what can you say about the angle between them?

<details><summary>answer</summary>

it's more than $90°$ (obtuse): $\cos\theta = \frac{u\cdot v}{\|u\|\|v\|} < 0$. the dot product's sign tells you whether two vectors point roughly the same way (positive), at right angles (zero), or roughly opposite ways (negative).

</details>

2. why is it enough to know where $e_1$ and $e_2$ go to know what a $2 \times 2$ matrix does to every vector?

<details><summary>answer</summary>

every vector is $v = v_1e_1 + v_2e_2$, and matrices respect combinations: $Av = v_1Ae_1 + v_2Ae_2$. so the images of $e_1$ and $e_2$ (the columns) decide the image of every $v$.

</details>

3. what are the columns of the matrix that reflects the plane across the line $y = x$? its determinant?

<details><summary>answer</summary>

the reflection swaps $x$ and $y$: $e_1 \to (0, 1)$, $e_2 \to (1, 0)$, so the matrix is $\begin{pmatrix}0&1\\1&0\end{pmatrix}$, with $\det = -1$ (it flips orientation and keeps areas).

</details>

4. you rotate by $90°$ and then scale $x$ by 2, or scale first and then rotate. where does $e_1$ end up in each case?

<details><summary>answer</summary>

rotate then scale: $e_1 \to (0, 1) \to (0, 1)$ (the scaling doesn't touch the $y$-direction). scale then rotate: $e_1 \to (2, 0) \to (0, 2)$. different results, so the order matters: $SR \ne RS$.

</details>

5. a matrix sends the unit circle to an ellipse with radii 4 and 0.5. what's the absolute value of its determinant?

<details><summary>answer</summary>

the circle's area $\pi$ becomes the ellipse's area $\pi \cdot 4 \cdot 0.5 = 2\pi$, so $|\det A| = 2$. (in general $|\det A|$ is the product of the two radii, the singular values.)

</details>

6. why does `project` divide by $v \cdot v$ and not by $\|v\|$?

<details><summary>answer</summary>

$\frac{u \cdot v}{\|v\|}$ is the shadow's *length*; you then need a unit vector $\frac{v}{\|v\|}$ to point it. together that's $\frac{u\cdot v}{\|v\|}\frac{v}{\|v\|} = \frac{u \cdot v}{v \cdot v}v$. dividing by $\|v\|$ only once would make the answer grow with the length of $v$.

</details>

7. (the predict callout in idea 3.) which of $AB$ and $BA$ sends $e_1$ to $(1, 1)$, for the shear $A$ and the quarter turn $B$?

<details><summary>answer</summary>

$BA$ applies the shear first: $Ae_1 = (1, 0)$, then the quarter turn gives $(0, 1)$. $AB$ applies the quarter turn first: $Be_1 = (0, 1)$, then the shear gives $(1, 1)$. so it's $AB$.

</details>

## common bugs

- **combining rows instead of columns.** in `matvec`, `v[j] * A[j, :]` uses rows. symptom: you compute $A^\top v$. right for symmetric matrices, wrong otherwise. (that's why the tests use non-symmetric, non-square matrices.)
- **degrees instead of radians.** `np.cos` and `np.arccos` work in radians. symptom: `rotation(90)` doesn't rotate by a quarter turn.
- **a clockwise rotation.** swapping the signs of the sines rotates the wrong way. symptom: $e_1$ goes to $(0, -1)$ instead of $(0, 1)$ for a quarter turn.
- **`nan` from `arccos`.** rounding pushes the cosine to $1.0000000000000002$. symptom: the angle between a vector and itself is `nan`. clip first.
- **points as columns or rows, mixed up.** `transform_points` takes points as rows. symptom: shape errors, or the image of the circle is transformed by $A^\top$.
- **forgetting the wrap-around in the shoelace formula.** symptom: polygon areas come out wrong unless one corner happens to be at the origin.

## going deeper

- 3Blue1Brown (Grant Sanderson), *Essence of Linear Algebra* (video series, 2016). chapters 1–6 animate exactly this topic: vectors, matrices as transformations, composition and determinants. [3blue1brown.com/topics/linear-algebra](https://www.3blue1brown.com/topics/linear-algebra)
- Strang (2016), *Introduction to Linear Algebra*, 5th ed., Wellesley-Cambridge Press, chapters 1–2 (vectors, dot products, and matrices acting on vectors).
- Deisenroth, Faisal & Ong (2020), *Mathematics for Machine Learning*, Cambridge University Press, chapters 2–3 (linear algebra and analytic geometry, with ML in mind). free online. [doi:10.1017/9781108679930](https://doi.org/10.1017/9781108679930)

## connections

- ← builds on: functions and composition (topic 01), NumPy indexing (ch 00 · topic 02), plotting (ch 00 · topic 04).
- → used in: everything in chapter 02 (linear algebra), in particular matrix multiplication four ways (ch 02 · topic 02), determinants and solving systems (ch 02 · topic 03), projections and least squares (ch 02 · topic 04), and the SVD as the "circle to ellipse" picture made precise (ch 02 · topic 06). dot products are the prediction of every linear model (topic 06, ch 08), and cosine similarity compares embeddings (ch 17, ch 25).
