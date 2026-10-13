# __TITLE__ · lab notes

<!-- TEMPLATE: for each experiment: what learners usually predict, what actually happens, why. -->

## experiment 1: how fast does the sample mean settle down?

**what people usually predict:** the error falls 4× when n grows 4×.

**what happens:** it falls 2×. the log-log slope is −1/2.

**why:** the standard deviation of the mean is $\sigma/\sqrt{n}$, so error ∝ $n^{-1/2}$. halving the error takes four times the data. this rate comes back over and over (monte carlo, SGD noise, generalization bounds).
