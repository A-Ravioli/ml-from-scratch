# decision trees · lab notes

the numbers below are from the reference solution with the lab's seeds. your own trees should reproduce them closely: the fits are deterministic (no `max_features`), but where two splits tie exactly, a different tie-break or a different rounding in the impurity can move a leaf or a digit.

## experiment 1: depth versus overfitting

**what people usually predict:** that the fully grown tree overfits "a bit": maybe 75% test accuracy against the best possible 80%, because the noise only affects 20% of the points. many expect the best depth to be large (10 or more), since the boundary is a curve.

**what happens:** the fully grown tree (268 leaves) is perfect on the training set and gets **0.664** on the test set, worse than a single split (0.700). it agrees with the *clean* labels on only **0.770** of the test points. the best depth is **3** (test 0.767), and depths 1 and 2 are identical (0.700). cross-validation picks `ccp_alpha = 0.005`, which prunes the full tree back to **7** leaves with test accuracy **0.762**: within a percentage point of the best depth, without ever looking at the test set.

**why:** a fully grown tree puts (almost) every training point in its own small leaf, so a test point is predicted with the label of a nearby training point: it's a 1-nearest-neighbour rule with box-shaped neighbourhoods. that neighbour's label was flipped with probability $\varepsilon = 0.2$, and the test point's own label independently with probability $\varepsilon$, so they disagree with probability $2\varepsilon(1 - \varepsilon) = 0.32$ even where the boundary is learned perfectly: accuracy 0.68 at best, the dashed line. the extra 1.6 points come from the boundary itself being drawn imperfectly with 1,000 points. against the clean labels, the full tree is wrong wherever a flipped training point owns a leaf: about 20% of the area, which is the 0.770. (Cover & Hart, 1967, proved the same $2\varepsilon(1 - \varepsilon)$ limit for 1-NN.)

depths 1 and 2 tie because of greediness: the best single split cuts the sine wave horizontally, and the best second-level splits don't change any leaf's majority class, so the predictions are the same. depth 3 is the first level at which leaves on the two humps can disagree with their parents. this is the XOR lesson in miniature: the value of a split can depend on the splits after it. pruning gets around it by growing first and judging whole subtrees afterwards.

## experiment 2: how stable is a tree?

**what people usually predict:** that removing 5% of the rows changes a tree by about 5%: a few leaves move, a few test predictions flip.

**what happens:** two fully grown trees fitted on 95% subsets disagree on **15.3%** of the test points on average (7.3% to 21.8% depending on the pair). two depth-3 trees disagree on **1.8%** (0% to 6.3%). the root splits barely move: every refit splits on $x_2$ first, with thresholds between 0.084 and 0.133.

**why:** the fully grown tree's predictions are dominated by its smallest leaves, which hold one or two training points each. remove a few rows and the boxes around those rows reshape: a noise point that owned a leaf disappears, a neighbouring leaf grows, and a split lower down picks a different feature because a different row now gives the best decrease. the change in a split near the top also cascades: every node below it sees a different set of rows. the disagreement concentrates where the noisy labels are, i.e. everywhere (look at the two pictures: the coarse structure is shared, the small boxes are not). a depth-3 tree only has 8 big leaves whose majorities rarely change. this high variance of deep trees is exactly what averaging attacks: a random forest (topic 02) averages hundreds of such unstable trees, and the disagreement between them averages out while their (low) bias stays.

## experiment 3: boundaries that aren't axis-aligned

**what people usually predict:** that a straight line is easy for any classifier, so a handful of leaves (4–8) and near-perfect accuracy.

**what happens:** at 0° and 90° the tree needs **2** leaves and is perfect. at 45° it needs **50** leaves and still gets only **0.9804** on the test set; the leaf count rises and falls with the angle (14 at 5°, 50 at 45°), and test accuracy mirrors it.

**why:** every split is perpendicular to an axis, so a tree approximates a diagonal line by a **staircase**. each step costs a leaf on each side, and the training points only pin the staircase down where they happen to lie, so the errors are the small triangles between the steps and the line. with $n$ uniformly spread points, the staircase needs on the order of $\sqrt n$ steps to follow the line to within the spacing of the points (doubling the data roughly multiplies the leaves by 1.4: about 23 leaves at 500 points, 47 at 2,000 and 99 at 8,000 when we tried it), and the test error falls only slowly. a single extra feature, $x_2 - x_1$ (or the two features rotated by 45°), makes the problem one split again: trees are excellent at thresholds and interactions, and bad at linear combinations, so feature engineering for trees means *adding the combinations they can't build*. random rotations (rotation forests) and oblique trees exploit this; gradient boosting with many shallow trees can approximate a diagonal well but still pays for it in trees.
