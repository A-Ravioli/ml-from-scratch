# flow matching & rectified flow · lab notes

numbers measured with the reference solution on an Apple M2 laptop CPU (one thread, as in CI) on a shared machine; the whole lab took about 90 s. the data are synthetic, so offline runs are the same experiments. one training seed per configuration; the sampling noise on a fraction from 4,000 samples is about ±0.008.

## experiment 1: straighter paths, fewer steps

**what people usually predict:** with straight training paths, a few Euler steps should already be fine; reflow helps a little.

**what happens** (fraction within 3 std of a mode, modes covered in brackets):

| flow | straightness | 1 step | 2 steps | 4 steps | 8 steps | 16 steps | 64 steps |
|---|---|---|---|---|---|---|---|
| independent (1-rectified) | 1.661 | 0.000 (0) | 0.003 (0) | 0.635 (8) | 0.873 (8) | 0.935 (8) | 0.941 (8) |
| reflow (2-rectified) | 0.0025 | **0.789** (8) | 0.827 (8) | 0.853 (8) | 0.872 (8) | 0.889 (8) | 0.902 (8) |

final training losses: 1.746 for the independent coupling, 0.0017 after reflow. training took 27 s, the 10,000 reflow pairs 6 s, retraining 21 s.

the original flow with one Euler step puts *every* sample in a small blob at the origin (no mode covered); with two steps, still nothing. after one round of reflow, a single step covers all 8 modes with 79% of the samples near a mode, about as good as the original flow with 6–8 steps; the rest sit on the straight segments between neighbouring modes (the plot shows an octagon). but the reflowed flow tops out lower: 0.90 at 64 steps against 0.94.

**why:** with independent pairs, the marginal velocity at $t = 0$ is $\mathbb{E}[x_1] - x_0$ (on paper 4): every starting point heads for the data mean, the origin, and one Euler step of size 1 lands there exactly. the paths only bend towards their modes later, so a solver needs several steps to follow them. reflow trains on pairs $(x_0, \Phi(x_0))$ produced by the first flow's ODE, a coupling without crossings, so the conditional targets through each point agree, the loss drops to almost zero (0.0017, against the irreducible 1.75 of the independent coupling), and the paths are nearly the straight lines between the pairs (straightness 660 times smaller): one Euler step follows a straight line exactly. the price shows at 64 steps: the second flow learned the first flow's *samples* (generated with 100 Euler steps), so its distribution inherits the first flow's errors plus its own; reflow buys speed, not quality.

## experiment 2: what the paths look like

**what people usually predict:** trained on straight lines, the flow moves along straight lines.

**what happens:** the independent-coupling trajectories start out heading towards the centre, then hook sharply towards their modes near the end; their endpoints are far from the dotted straight lines through start and end in the middle of the path. after reflow, the trajectories lie on top of their straight lines.

**why:** at time $t$ many training lines pass through the same point in different directions, and the network learns their average (idea 1). near $t = 0$ that average points at the data mean; only once $x_t$ is close enough to the data to "know" which mode it belongs to does the average direction point at that mode, hence the hook. two trajectories of an ODE can never cross (uniqueness of solutions), while independent training lines cross everywhere: that mismatch is the curvature. reflow replaces the crossing lines by the ODE's own non-crossing pairing, so the training lines and the trajectories coincide.

## experiment 3: noise to noise

**what people usually predict:** when data and noise have the same distribution, the flow is the identity, the loss is about zero, and one step is perfect.

**what happens:** the loss settles at **1.577**, against the floor $\pi/2 = 1.571$ (on paper 3); 1-step samples have a standard deviation of about 0.06 (everything collapses near the origin); 64-step samples have the right spread (std 0.98) and end on average only 0.075 away from their own starting noise; straightness 0.86.

**why:** with independent pairs the network sees, at a point $x_t$, lines from many $x_0$ to many $x_1$, and can't know which $x_1$ a given line is heading to: the conditional variance $s^2/((1 - t)^2 + t^2 s^2)$ integrates to $\pi/2$ per coordinate, so the loss can't go lower, however good the network. the learned field is not zero: it's the marginal field, which first pulls everything towards 0 and then pushes it back out, $x_t = x_0\sqrt{(1 - t)^2 + t^2}$, so the end-to-end map is (nearly) the identity, but the path isn't: halfway, each point is at 71% of its starting radius. one Euler step follows only the initial pull and lands at the origin. one round of reflow would see pairs $(x_0, \approx x_0)$ and learn a field near zero: straight, trivial paths and a loss near 0.

## references

- Liu, Gong & Liu (2023), "Flow straight and fast: learning to generate and transfer data with rectified flow", *ICLR*. [arXiv:2209.03003](https://arxiv.org/abs/2209.03003)
- Lipman, Chen, Ben-Hamu, Nickel & Le (2023), "Flow matching for generative modeling", *ICLR*. [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
- Tong et al. (2024), "Improving and generalizing flow-based generative models with minibatch optimal transport", *TMLR*. [arXiv:2302.00482](https://arxiv.org/abs/2302.00482)
