# from linear models to neurons · lab notes

## experiment 1: depth without nonlinearity

**what people usually predict:** that a deep linear network is "approximately" a single layer, with differences that grow with depth; and that on the ring data a network without an activation still learns something curved, just badly.

**what happens:** the largest difference between a deep linear network and its collapsed layer, over 500 random points, is **0** for 1 layer and between **$1.3 \times 10^{-15}$ and $5.3 \times 10^{-15}$** for 2 to 10 layers: rounding error, with no growth in depth. on the ring data (300 points, 25% of them inside the circle), the tanh network reaches **100%** training accuracy (final loss 0.029) with a round decision boundary. the identity network reaches **75.3%** (final loss 0.558): exactly what you get by predicting "outside" everywhere, and its predicted probability is nowhere above 0.5, so it has no decision boundary inside the plot at all.

**why:** the stack *is* one linear map, $X(W_L\cdots W_1)^\top + b$, so computing it layer by layer or in one product differs only by the order of floating-point operations. without a nonlinearity, the 16-unit network is a logistic regression in disguise, and a logistic regression's decision boundary is a straight line. no straight line separates a disc from the ring around it better than "everything is outside", so the best linear classifier has the majority-class accuracy of 75%, and its fitted probabilities hover around the class frequency, 0.25. with tanh, the 16 hidden units bend the boundary into a closed curve. depth without nonlinearity buys nothing.

## experiment 2: big enough isn't enough

**what people usually predict:** that a network that *can* represent XOR (2 hidden units) learns it essentially every time, since the problem has only four points.

**what happens:**

| hidden units | solved (of 50) | median final loss | worst final loss |
|---|---|---|---|
| 1 | 0 | 0.479 | 0.693 |
| 2 | **27** | 0.002 | 0.349 |
| 3 | 44 | 0.002 | 0.348 |
| 4 | 49 | 0.002 | 0.348 |
| 8 | 50 | 0.001 | 0.001 |

with exactly enough units, gradient descent solves XOR in only about half the runs. every extra unit raises the success rate, and 8 units never fail. the failed runs aren't still slowly converging: their losses sit at about **0.348**, a plateau, while successful runs reach 0.002.

**why:** one hidden unit can't solve XOR at all: its decision region is a half-plane, which gets at most three corners right (median loss 0.479: the typical width-1 run is certain and right about one corner, and gives the other three the same prediction, probability 2/3 for whichever class two of them share (e.g. $(0.002, 0.666, 0.666, 0.667)$): $(3\log 3 - 2\log 2)/4 = 0.477$). two units can, but the loss surface has other attractors. in every stuck run the network is confident and right about two corners and predicts exactly 0.5 for the other two (one of each class), for example $(0.001, 0.5, 0.999, 0.5)$: the mean loss is then $2\log 2/4 = \tfrac12\log 2 = 0.347$, the plateau in the table. the two hidden units have saturated so that those two corners land on (nearly) the same hidden point, a corner of the square $[-1, 1]^2$ (in one run, $(0, 1)$ and $(1, 1)$ both map to $(-1, 1)$): no output weights can tell them apart, and saturated tanh units pass almost no gradient back to pull them apart again. which basin a run falls into depends on the random init. extra units make it overwhelmingly likely that *some* pair of units starts out in a useful arrangement, and the network can shrink the outgoing weights of the rest. this is a first glimpse of something that holds at every scale: networks larger than strictly necessary are much easier to *train*, even when the extra capacity isn't needed to *represent* the solution (overparameterization; chapter 19 returns to it).

## experiment 3: what the hidden layer does

**what people usually predict:** that the corners stay in a square, maybe stretched, and the output layer somehow draws a curved boundary.

**what happens:** in the hand-wired ReLU network the corners land at $(0, 0)$, $(1, 0)$, $(1, 0)$ and $(2, 1)$: $(0, 1)$ and $(1, 0)$ land on the **same point**, and the output line $h_1 - 2h_2 = 0.5$ separates that point from the other two. in the trained 2-unit tanh network the corners land near the corners of the square $[-1, 1]^2$, but rearranged: the two class-1 corners have merged near one corner, $(0.9, 1.0)$, while $(0, 0)$ and $(1, 1)$ sit at two others, and a single straight line splits them.

**why:** the output layer is linear, so it can only draw a line in hidden space. the hidden layer's job is to move the points so that a line suffices: here, by folding the plane so that the two "exactly one input on" corners coincide. tanh saturates, so trained hidden units tend to push points towards the corners of the hypercube, which acts like a learned binary feature ("is $x_1 + x_2$ more than about 0.5?", "is it more than about 1.5?"). in one sentence: a hidden layer learns a representation in which the task is linear. every deep network is a stack of these, which is why deep learning is sometimes called representation learning.
