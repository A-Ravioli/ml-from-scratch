# the perceptron · lab notes

## experiment 1: the margin and the number of mistakes

**what people usually predict:** the number of updates tracks the bound, perhaps a constant factor below it, and quadruples when the margin halves.

**what happens** (600 points in the unit disc, the points within the gap removed; 10 shuffling orders each):

| gap | points | margin of the true line | bound $(R/\gamma)^2$ | updates (10 orders) | control (line through the origin): median |
|---|---|---|---|---|---|
| 0.2 | 442 | 0.200 | 50 | 2 – 6 (median 5) | 4 |
| 0.1 | 531 | 0.100 | 200 | 2 – 16 (median 9) | 8 |
| 0.05 | 567 | 0.053 | 710 | 67 – 83 (median 73) | 8 |
| 0.02 | 588 | 0.025 | 3,342 | 117 – 125 (median 122) | 16 |
| 0.01 | 591 | 0.013 | 12,254 | 149 – 191 (median 168) | 29 |
| 0.005 | 595 | 0.006 | 58,538 | 229 – 283 (median 248) | 28 |

the bound always holds, by a factor of 10 to 240. the actual counts grow far more slowly than $1/\gamma^2$: from gap 0.05 to 0.005 the bound grows 82-fold and the median count 3.4-fold. and there's a jump between gaps 0.1 and 0.05 that the bound doesn't predict.

**why:** the bound is a worst case over every possible order of the examples and every dataset with that $R$ and $\gamma$; the proof only uses that each mistake makes *some* progress along $w^\star$. random orders on random data make much more progress per mistake. the bound also only sees the margin of one particular separator: the perceptron is free to find any other. that explains the jump. the true line is $x_1 + x_2 = 0.1$, at distance $0.07$ from the origin. with a gap of 0.1, the line through the origin, $x_1 + x_2 = 0$, lies inside the empty band, so a separator with $b = 0$ exists and the perceptron finds one in a handful of updates. with a gap of 0.05 it doesn't: the boundary must sit off the origin, with $|b|/\|w\| \approx 0.07$. but $b$ moves in steps of $\pm 1$ while each update changes $w$ by a point of norm at most 1, so the perceptron needs $\|w\|$ somewhere between about 8 and 14 before $b = -1$ puts the boundary in the right place ($1/0.07 \approx 14$ is the limit as the gap shrinks; with a gap of 0.05, about 8.5 suffices, since $b/\|w\|$ only has to land inside the empty band), and that takes dozens of updates. the control confirms it: with the line moved to pass through the origin, no intercept is needed and the counts stay small and smooth (median 4, 8, 8, 16, 29, 28), with no jump. the augmented "1" is large compared with data in a unit disc: the intercept is the slow part. the bound doesn't depend on $n$ or $d$ except through $R$ and $\gamma$, which is why it remains meaningful in very high dimensions (and why the kernel perceptron of chapter 09 works).

## experiment 2: when the data aren't separable

**what people usually predict:** the final weights wobble by a point or two; averaging smooths that a little.

**what happens:** from epoch 10 to 30, the test accuracy of the final weights ranges from **0.44 to 0.75** (standard deviation 0.085), sometimes worse than guessing. the averaged weights stay between **0.819 and 0.824** (standard deviation 0.002), close to the true line's accuracy of about 0.85 (the ceiling, since 15% of test labels are flipped).

**why:** with 15% flipped labels, about 75 training points are on the wrong side of any good line, and each visit to one of them is a mistake that moves $w$ by a whole data point. so the weights never settle: they jump around a good region, and the last jump decides where the final weights land. the boundary can rotate a long way from one epoch to the next, which is why the final weights' accuracy swings by 30 points, and stopping at "the last epoch" is a lottery. the average of thousands of iterates sits near the middle of the region they bounce around, so the random jumps cancel. this is the same reason Polyak averaging and decaying step sizes help SGD (chapter 06): with a constant step, SGD doesn't converge, it hovers. with the plain perceptron, you'd have to pick the epoch on a validation set ("pocket" the best weights so far), which costs data; averaging needs nothing.

## experiment 3: XOR, and a feature that fixes it

**what people usually predict:** the perceptron takes a while on XOR but eventually gets most points right.

**what happens:** with $(x_1, x_2)$, it runs all **200** epochs, still makes 95 mistakes (of 200 points) in the last one, and ends at **50%** training accuracy: no better than a constant. with $(x_1, x_2, x_1x_2)$ it stops after **2** epochs with 100% training accuracy, and its boundary in the original plane is a hyperbola that bends between the clusters.

**why:** a line would need $b \le 0$ (for $(0, 0)$), $w_1 + w_2 + b \le 0$ (for $(1, 1)$), $w_1 + b > 0$ and $w_2 + b > 0$ (for the other two). adding the last two gives $w_1 + w_2 + 2b > 0$, and subtracting the first two gives the opposite: no line exists, so every epoch has mistakes and the perceptron cycles forever, ending wherever `max_epochs` stops it. the feature $x_1x_2$ is 1 only at $(1, 1)$, and in three dimensions the classes are separable (e.g. $x_1 + x_2 - 2x_1x_2 > 0.5$ is exactly XOR on the corners). a linear boundary in feature space is a curved one in input space. this is the whole idea of feature maps: hand-picked ones (polynomials, this topic and topic 01), infinitely many fixed ones through kernels (chapter 09), or ones learned from data by a hidden layer (chapter 14). Minsky and Papert's XOR argument was about single-layer perceptrons on raw inputs, and it's correct; the field took a while to absorb that the fix is more layers.
