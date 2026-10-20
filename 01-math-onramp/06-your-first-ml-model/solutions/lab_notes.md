# your first ML model · lab notes

## experiment 1: learning rates

**what people usually predict:** bigger is faster, up to some vague point where things "become unstable". often: 0.9 is the fastest of the four, 1.05 is "a bit worse" or "a bit unstable".

**what happens:** after 50 steps (measured as loss minus the best possible loss, $0.0916$):

| learning rate | behavior | excess loss after 50 steps |
|---|---|---|
| 0.01 | smooth, slow | $1.9$ |
| 0.1 | smooth | $3 \times 10^{-9}$ |
| 0.3 | smooth, fast | $0$ (to machine precision) |
| 0.5 | **exactly right after one step** | $0$ |
| 0.9 | zigzags, converges | $3 \times 10^{-9}$ (same as 0.1!) |
| 1.0 | bounces forever | $14.0$ (no progress at all) |
| 1.05 | zigzags outward, explodes | $2 \times 10^{5}$ |

the right-hand panel shows why: for 0.9, $w$ jumps from one side of the best value to the other on every step, by a shrinking amount, while for 0.1 it creeps up from one side; for 1.05 the jumps grow. the best learning rate is 0.5, not the largest one that converges. 0.9 converges at exactly the same speed as 0.1, and 0.99 at the same speed as 0.01. at exactly 1.0 the loss never changes.

**why:** for standardized inputs, the loss is a perfectly round bowl with curvature $a = 2$ in every direction, so every step multiplies the distance to the minimum by $1 - 2\eta$. at $\eta = 0.5$ that factor is 0: one step lands on the minimum. at $\eta = 0.1$ it's $0.8$, and at $\eta = 0.9$ it's $-0.8$: the same size, so the same speed, with the sign making the second one jump from side to side. at $\eta = 1$ the factor is $-1$: the parameters bounce between two points forever, at the same loss. beyond 1 the factor is bigger than 1 in size, and the distance grows geometrically. what matters is never the learning rate alone but $\eta$ times the curvature, which is why a learning rate that works for one problem can diverge on another (or on the same problem in different units: experiment 2). real loss landscapes aren't perfect bowls, so in practice you search learning rates on a log scale and pick something comfortably below the point of divergence (chapter 15 builds a learning-rate finder that automates this).

## experiment 2: the same data, in square meters

**what people usually predict:** that with the largest safe learning rate and 1,000 steps, gradient descent finds the line, maybe a little less precisely than the standardized version.

**what happens:** the largest stable learning rate on raw sizes is about **$3.8 \times 10^{-5}$**. at 90% of it, after 1,000 steps, gradient descent reaches $w = 0.0230$, $b = 0.005$ (loss $0.130$), against the best line $w = 0.0199$, $b = 0.541$ (loss $0.0916$). the intercept has barely moved from its starting value of 0. at 110% of the limit, the parameters reach $10^{77}$ and the loss $10^{159}$. the standardized run, converted back, gets $w = 0.019901$, $b = 0.540538$, the exact answer to six digits, in 30 steps.

**why:** the curvature of the loss in the slope direction is about $2\,\overline{x^2} \approx 2 \times 151^2 \approx 5 \times 10^4$, because sizes are numbers around 150. the steepest curvature limits the learning rate for *every* direction, so $\eta$ must stay below about $4 \times 10^{-5}$. but the intercept direction has curvature about 2 (more precisely, the flattest direction of the valley has curvature 0.27). at $\eta \approx 3.4 \times 10^{-5}$, that direction shrinks by a factor of about $1 - 3.4 \times 10^{-5} \times 0.27 \approx 0.99999$ per step: it would need around half a million steps to converge. meanwhile the slope has quickly done the best it can *given* the wrong intercept, which is why $w$ is too big (it compensates for $b \approx 0$ by tilting the line up). standardizing makes both curvatures 2 at once, and the model is the same line, written differently. rule: always standardize features before gradient descent. the capstone has eight features with wildly different scales (incomes around 4, populations in the thousands), where it matters even more.

## experiment 3: the shape of the valley

**what people usually predict:** ellipses or circles on both sides; a straight path to the minimum, maybe a bit curved.

**what happens:** on the left (raw sizes), the contours are extremely long, thin ellipses: a narrow valley running almost straight along the $b$ axis. gradient descent's steps bounce back and forth *across* the valley (the slope $w$ overshoots, because the learning rate is close to the limit set by the steep direction), settle onto its floor, and then the path stops, nowhere near the minimum: after 300 steps $b$ has hardly moved from 0. on the right (standardized), the contours are circles and the path goes straight to the center, landing there within a few steps.

**why:** the gradient is always perpendicular to the contour lines, pointing straight uphill. in a narrow valley, "straight downhill" points almost directly at the valley floor, not along it. so gradient descent quickly fixes the steep direction (the slope $w$), and then the remaining gradient, which points along the valley, is tiny, and the learning rate (capped by the steep direction) is far too small to make progress on it. in a round bowl, downhill points straight at the center from everywhere. this is the most important picture in optimization: an **ill-conditioned** problem (very different curvatures in different directions, a ratio of about 200,000 here) makes gradient descent slow, and the fixes are to change the coordinates (standardizing, and in general "preconditioning") or to use a smarter optimizer (momentum and Newton's method, chapter 06). you'll meet the same ratio as a matrix's **condition number** in chapter 02, topic 08.
