# visualizing data · lab notes

## experiment 1: how many peaks?

**what people usually predict:** all three show the same basic shape, with more bins just giving more detail. the 150-bin histogram is "the most accurate" because it's the most detailed.

**what happens:** with **4 bins** the counts are 61, 66, 72, 101: a lopsided ramp, one lump leaning right, no hint of two groups. with **12 bins** two clear peaks appear, near −1 and +1, with a dip in between: the true story. with **150 bins** each bin holds 0 to 7 values and the picture is a ragged comb with a dozen "peaks", none of which mean anything. on the slider, anything from about 8 to 30 bins shows the two groups; below about 6 they merge, and above about 50 the noise takes over.

**why:** the data is a 50/50 mix of two bell curves centred at −1 and +1 with spread 0.55. with 4 bins over a range of about 4, each bin is 1 unit wide, as wide as the gap between the groups, so the dip between them is averaged into its neighbours; and because the edges happen to fall where they do, the right group lands mostly in one bin and looks bigger. with 150 bins, a bin expects about 2 values, and counts of that size fluctuate by about $\sqrt{2} \approx 1.4$, as big as the signal. this is the bias–variance trade-off you'll meet formally in chapter 07: wide bins are biased (they blur real structure), narrow bins are noisy. in practice, look at several bin counts and believe only features that survive across a range of them, or use a smooth density estimate (chapter 09's kernel density estimation).

## experiment 2: which model is better?

**what people usually predict:** model A. on the linear plot it drops much faster, sits below B for most of the run (at step 300, A is about 0.05 and B about 0.09), and both look like they've flattened out at "zero" by the end.

**what happens:** on the linear axis the last three quarters of the plot look identical, both hugging zero. switch to the log axis and the story reverses. **A flattens out at 0.03 by step 500 and never improves again. B keeps falling in a straight line** (it's a power law, so on log–log axes it would be perfectly straight; here it's a gentle curve) and ends at 0.015, half of A's loss, still dropping. B crosses back below A for good at about step 980, and from there on the gap only widens, but on the linear plot that crossing is invisible.

**why:** a linear axis spends its whole height on the range from 0 to 1. differences between 0.03 and 0.015 are about 1.5% of that height, invisible. a log axis gives every factor of 10 the same height, so the late phase of training, where models really differ, gets as much space as the early phase. A's curve is an exponential approach to a floor (often a sign of a model that's too small, or a learning rate that's too large to settle further), while B's slow power-law decay is typical of a model that would keep improving with more training. the decision you'd make from the linear plot is the wrong one. **plot losses on a log scale.**

## experiment 3: four datasets, one set of statistics

**what people usually predict:** if all the summary numbers agree to two or three decimals, the scatter plots should look similar: four noisy, upward-sloping clouds.

**what happens:** they're completely different. **I** is the cloud you expected: a noisy linear trend, for which the line is a fine model. **II** is a smooth curve (a parabola) with no noise at all; a line is the wrong model. **III** is a perfect straight line except for one outlier, which tilts the fitted line away from the other ten points. **IV** has ten points stacked at $x = 8$ with no relationship to $y$ whatsoever, plus one point at $x = 19$ that single-handedly creates the whole correlation of 0.82 and the slope of the line.

**why:** means, variances, a correlation and a line are five or six numbers. they can't describe 11 points in 2-D, which take 22 numbers. Anscombe built the quartet to make the point that summary statistics are compressions, and very different data can compress to the same summary. Matejka & Fitzmaurice (2017) showed you can make almost any shape (including a dinosaur) with the same statistics. the same applies to an ML model's accuracy: one number that can hide a model that's great on most classes and useless on one, or one that's right for the wrong reasons. look at the data, and look at the errors.

## experiment 4: a perfect classifier?

**what people usually predict:** with 100% training accuracy, a clean boundary between the two classes. maybe a little wiggly.

**what happens:** with $k = 1$ the boundary is a mess: the plane is broken into many little islands, one around each training point that sits on the "wrong" side of the overlap. training accuracy is **1.000**, but accuracy on 1,000 new points from the same distribution is only about **0.62**. as you raise $k$, the islands disappear and the boundary smooths toward a single gentle curve between the classes; at $k = 25$ the training accuracy has *dropped* to about 0.74, and the accuracy on new points has *risen* to about 0.73.

**why:** 1-NN memorizes the training set. every training point, including the noisy ones that wandered into the other class's region, claims a small neighbourhood for its label. those islands are pure noise, and new points that land in them get misclassified. larger $k$ averages over more neighbours, so a single stray point can no longer flip the prediction. the training accuracy goes down because the model stops fitting the noise, and the test accuracy goes up for the same reason. this is **overfitting**, seen in one picture, and the lesson that runs through the whole course: **training accuracy is not the goal; performance on new data is** (chapter 07). the best possible accuracy here is about 0.73 (the classes genuinely overlap), and $k \approx 25$ gets there.
