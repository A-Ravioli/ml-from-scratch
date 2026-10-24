# 19 · capstone rubric: reproduce a theory figure

grade your write-up on four criteria, each 0–3. a solid capstone scores at least 8 of 12, with no criterion at 0.

## 1. correctness (0–3)

- **0:** the checks don't pass, or the figure comes from a different protocol than the one described (minibatches, early stopping, a different loss) without saying so.
- **1:** the checks pass and the sweep ran, but the reported numbers can't be traced to the run (no table of widths, parameters and errors), or the threshold $nK$ is computed wrongly.
- **2:** the figure, the table and the located interpolation width, peak and first minimum all come from your run and agree with each other; parameter counts are right.
- **3:** as 2, and you checked something the tests don't: that the widest models really fit every training label including the mislabeled ones, that training had converged (the training loss curve for a few widths), or that a second data subset gives the same peak location.

## 2. results against the theory, with uncertainty (0–3)

- **0:** no seed intervals; the peak is asserted from a single run.
- **1:** intervals are shown but not used: "there is a peak" without checking that it rises above its neighbours' intervals.
- **2:** the peak's height and location are reported with t intervals over seeds, compared with the threshold $nK$ and with the random-features curve of topic 04 (location, height, sharpness), and the clean-label curve is used as a control.
- **3:** as 2, and the comparison is quantitative: e.g. the ratio of peak to first-minimum test error for clean and noisy labels, or where the training error crosses zero relative to $nK$ in parameters, with intervals.

## 3. analysis and ablations (0–3)

- **0:** the write-up restates the figure.
- **1:** prompts 3–5 are answered in general terms ("overparameterization helps") without connecting to the measurements.
- **2:** each explanation is tied to a measured quantity (the interpolation width, the effect of noise, the gap to the random-features curve), and the fragility prompt names concrete protocol changes with reasons.
- **3:** you ran at least one ablation (a ridge or weight decay, cross-entropy instead of squared loss, fewer or more steps, or one of the options in the brief) and report what it did to the peak, with intervals.

## 4. clarity (0–3)

- **0:** no figure or unreadable axes.
- **1:** a figure without labels, units or the threshold marked; numbers buried in text.
- **2:** one clear figure (training error, test error and test loss against parameters, both noise levels, the threshold marked), a compact table, and answers that a classmate could check against the table.
- **3:** as 2, and the write-up states its limitations plainly (subset size, number of seeds, fixed step budget, one architecture) and what you'd do with ten times the compute.
