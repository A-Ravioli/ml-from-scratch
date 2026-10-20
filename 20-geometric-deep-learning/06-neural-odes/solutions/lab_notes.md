# neural ODEs · lab notes

numbers measured with the reference solution on a laptop CPU (about two minutes for the whole lab). everything is synthetic, so `verify.py`'s offline run gives the same accuracies and errors; the times depend on the machine.

## experiment 1: what does the adjoint save, and what does it cost?

**what people usually predict:** that the adjoint saves "a lot" of memory, and costs a little time, perhaps 10–20%.

**what happens:**

| steps | 10 | 20 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|---|
| backprop, MB saved | 11.3 | 22.6 | 56.6 | 113 | 226 | 566 |
| adjoint, MB saved | 0.022 | 0.022 | 0.022 | 0.022 | 0.022 | 0.022 |
| backprop, ms | 18.6 | 34 | 89 | 167 | 350 | 886 |
| adjoint, ms | 32.8 | 63 | 168 | 351 | 702 | 1,830 |

backprop through the solver saves 1.13 MB per RK4 step, exactly linearly; at 500 steps that's 566 MB, **25,000 times** what the adjoint saves (22 kB, independent of the number of steps). the adjoint is consistently about **twice as slow** (1.8–2.1×).

**why:** each RK4 step evaluates the MLP four times, and backprop must keep the inputs to every linear layer and every tanh for every evaluation: for a batch of 256 and widths 64, that's a few hundred kB per evaluation. the adjoint's forward solve runs under `no_grad`, so nothing is saved but the outputs ($y(0)$ and $y(1)$, plus the parameters' references), which is the 22 kB. its backward pass re-solves an ODE of three times the size, and every stage evaluates $f$ *and* a vector-Jacobian product through $f$ (itself a forward and backward pass of the MLP), against backprop's single backward pass through the stored graph: twice the time is what the operation count predicts. the trade is worth it when memory, not time, is the limit: big states (images, long sequences), many steps, or a GPU that can't hold the graph.

## experiment 2: is the adjoint gradient the right gradient?

**what people usually predict:** that both methods agree to rounding error everywhere, since the adjoint is "exact in the limit"; perhaps that float32 is a little worse.

**what happens** (relative error of $\partial L/\partial A$ against a float64, step-0.005 reference; RK4 step 0.05, $T = 2$):

| k | 0.5 | 2 | 5 | 10 | 15 | 20 |
|---|---|---|---|---|---|---|
| backprop float32 | 2.5e-07 | 1.8e-07 | 2.0e-07 | 7.0e-07 | 1.9e-07 | 2.4e-06 |
| adjoint float32 | 2.4e-07 | 2.3e-07 | 1.2e-05 | **4.1e-02** | **2.2e+03** | **3.3e+07** |
| backprop float64 | 2.3e-09 | 1.6e-09 | 3.8e-11 | 2.4e-11 | 2.4e-11 | 2.4e-11 |
| adjoint float64 | 3.3e-10 | 3.4e-09 | 1.3e-07 | 4.2e-06 | 3.3e-05 | 1.0e-01 |

backprop's gradient is accurate to float32 rounding for every $k$. the float32 adjoint is fine for slow dynamics, 4% wrong at $k = 10$, and at $k = 20$ wrong by a factor of 33 million: the gradient is garbage. float64 only delays the collapse: 10% wrong at $k = 20$.

**why:** the adjoint recomputes $y(t)$ by integrating backwards from $y(2)$. the fast mode decays like $e^{-kt}$ forwards, so backwards it *grows* like $e^{kt}$: by $t = 0$ it has been amplified by $e^{2k}$, which is $5 \times 10^8$ at $k = 10$ and $2 \times 10^{17}$ at $k = 20$. the true $y(2)$ has essentially nothing left in the fast direction, but its computed value carries a rounding error of relative size $10^{-7}$ (float32) or $10^{-16}$ (float64) in that direction, because the rotated coordinates mix it with the slow mode. that error is what gets amplified. measured directly (integrating the 64 starting points forward to $t = 2$ and back in float32 with the same solver), the reconstructed $y(0)$ is off by up to 0.002 at $k = 5$, 42 at $k = 10$ (the points themselves are of size 4), $1.2 \times 10^6$ at $k = 15$ and $1.9 \times 10^{10}$ at $k = 20$. so why is the $k = 10$ gradient only 4% wrong when the end of the reconstructed trajectory is garbage? because the gradient is an integral, $\partial L/\partial A = \int_0^2 a(t)\,y(t)^\top dt$, and the reconstruction error grows like $e^{k(2-t)}$ as the backward solve approaches $t = 0$: it's large only in the last stretch of width about $1/k$ before $t = 0$, a small part of the integral, weighted there by $a(t)$, which has no reason to be large. the error in the gradient is therefore a few orders of magnitude smaller than the error in $y(0)$, until the amplification $e^{2k}$ outruns that discount, as it does between $k = 10$ and $k = 15$. the gradient with respect to $A$ needs $y(t)$ (its vector-Jacobian product is $a\,y^\top$), so it inherits the error; the gradient with respect to $y_0$ wouldn't, for this linear system, since $a^\top A$ doesn't involve $y$. backprop never runs anything backwards in time: it reverses the *computation*, using the stored forward values, and its backward pass for a decaying system is itself decaying, hence stable. remedies: checkpoint the forward trajectory (store $y$ every few steps and reset the backward reconstruction there, as this implementation already does at output times), use float64, or use backprop when the system is stiff. Gholami et al. (2019) analyze exactly this failure.

## experiment 3: what a flow can't do

**what people usually predict:** either 100% for both ("a neural network can learn anything") or, from those who know the topology argument, about 50% for the 2-D model.

**what happens** (test accuracy, seeds 0, 1, 2):

| model | mean | per seed |
|---|---|---|
| 2-D (plain) | 0.992 | 0.992, 0.992, 0.992 |
| 3-D (augmented) | 1.000 | 1.000, 1.000, 1.000 |

the 2-D model gets 99.2%: high, but every seed misclassifies the same fraction of the test points, and the middle plot shows how. the flow tears the picture as far as a continuous map can: it pulls the inner disk down into a compact cluster and stretches the outer ring into a long curve that arcs over the top and comes down on both sides of the cluster. the ring is still a closed curve around the red points, so a few stretches of it must pass between the red cluster and the line the classifier draws; the blue points that land on those stretches (and one red point dragged along with them) are the errors, circled in the plot. the 3-D model simply lifts the inner circle up the third axis and separates the classes perfectly.

**why:** the exact flow of a smooth ODE is a homeomorphism of the plane: continuous, invertible, trajectories never crossing (on paper 4). the outer ring is a closed curve with the inner disk inside it, and a homeomorphism maps it to a closed curve with the disk's image inside, which no straight line can separate from the curve. so 100% is impossible in the continuum, and the best a 2-D flow can do is to make the parts of the ring that must stay "in the way" as long and thin as possible, so that few samples land on them. with finitely many samples that gets close to 100%, but the errors can't be trained away, and the vector field that stretches the ring this far has to be steep. one extra dimension removes the obstruction: in 3-D the disk can move out of the plane of the ring without crossing it, which is the whole point of augmented neural ODEs.
