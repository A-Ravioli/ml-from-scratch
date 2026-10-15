# 01 · capstone: predict house prices

## goal

predict the median house price of a California district from eight numbers about it (location, age of the houses, rooms, population, income), with a linear model

$$\hat y = w \cdot x + b = w_1x_1 + w_2x_2 + \dots + w_8x_8 + b.$$

it's topic 06 again, with eight inputs instead of one, and it uses every topic of this chapter. you'll standardize the features (topic 05's means and standard deviations), compute dot products for the predictions (topic 04), derive the gradient of the mean squared error with respect to all nine parameters by hand (topic 02's rules), train with gradient descent and watch the loss curve (topic 06), and check the result against the exact least-squares answer from the **normal equations**. finally you'll evaluate on held-out data against a baseline, with an error bar (topic 05), and explain what you see.

this is the first real model of the course, and it's honest about its limits: a straight-line model of house prices is decent but far from perfect, and finding out *where* it fails is half the point.

## data

the California housing data (Pace & Barry, 1997), derived from the 1990 U.S. census: one row per census block group (a "district", typically 600–3,000 people), 20,433 districts after dropping rows with missing values. the eight features are `longitude`, `latitude`, `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households` and `median_income` (in tens of thousands of dollars), and the target is the district's median house value in units of \$100,000. load it with

```python
from mlfs import datasets
X, y, names = datasets.california_housing()   # float64 (20433, 8), float64 (20433,), list of 8 names
```

the first call downloads a CSV once and caches it under `~/.cache/mlfs/`. with `MLFS_OFFLINE=1`, or with no network, you get a synthetic stand-in with the same format (1,000 districts), so the notebook always runs. the tests never touch either: they generate their own synthetic houses.

the notebook keeps a random 20% of the districts aside as a test set (`SEED = 0`). standardization statistics and all fitting use the training set only.

## configurations

| config | data | sizes | runtime (laptop CPU) |
|---|---|---|---|
| **small** (tests, `verify.py`, offline) | synthetic stand-in, or the tests' own generator | 800 train × 8 features (300–400 × 4 in the tests); 500 steps | about a second |
| **full** (your write-up) | real California housing | 16,346 train × 8 features; 500 steps (`LR = 0.1`, `N_STEPS = 500`) | a few seconds |

the notebook picks the config automatically from `datasets.is_offline()`. both use the same settings in the setup cell. change `LR` and `N_STEPS` there for the write-up experiments.

## deliverables

1. `capstone.py` with the five stages implemented and their checks green:
   - `standardize_features(X_train, X_test)`: rescale with training statistics only
   - `predict(X, w, b)` and `mse(X, y, w, b)`
   - `mse_grad(X, y, w, b)`: the hand-derived gradient, $\nabla_w = \frac2nX^\top r$ and $\frac{\partial}{\partial b} = \frac2n\sum r_i$
   - `fit_gd(X, y, *, lr=0.1, n_steps=500)`: gradient descent from zero, with the loss curve
   - `fit_closed_form(X, y)`: the normal equations $(A^\top A)\theta = A^\top y$ with $A = [X \mid 1]$, solved with the provided `gauss_solve`
2. the write-up section at the end of the notebook, answering its five prompts (results with uncertainty, training behavior, the no-standardization ablation, the weights, and a failure analysis).

## requirements

| check | threshold |
|---|---|
| tests in `test_capstone.py` pass | all (`pytest test_capstone.py`) |
| gradient descent with the defaults reaches the least-squares loss on the tests' synthetic houses | within $10^{-8}$ (`test_stage4_converges_to_least_squares`) |
| held-out $R^2$ of the full pipeline on the tests' synthetic houses | ≥ 0.90 (`test_stage5_heldout_r2_threshold`) |
| full config: test MSE on the real data | ≤ 0.55 (\$100k)², reported with its standard error |
| comparison with the baseline | report "predict the training mean" next to your model |

with the notebook's split and settings, expect a test MSE of about **0.488 ± 0.017** (a typical error, the RMSE, of about \$70,000) and $R^2 \approx 0.63$, against **1.33 ± 0.03** for the baseline. gradient descent's 500 steps end within about $10^{-4}$ of the closed form's training loss, and the two give the same test MSE to three decimals.

## stretch goals

- **the learning-rate limit.** compute the largest stable learning rate, $2/\lambda_{\max}$, where $\lambda_{\max}$ is the largest eigenvalue of the matrix of second derivatives $\frac2nA^\top A$ (use `np.linalg.eigvalsh` as a black box for now: you'll build it in chapter 02). compare with what you find by trial and error.
- **better features.** add $\log(\text{median\_income})$, rooms per household, or bedrooms per room as extra columns. which help on the test set, and by more than the standard error?
- **location.** the residuals have a strong geographic pattern. plot them on a longitude/latitude scatter. what does the model miss about the coast and the big cities?
- **a log target.** fit $\log y$ instead of $y$ and transform predictions back. does the error get better or worse, and for which districts?

## self-assessment

grade your write-up with [`rubric.md`](rubric.md).

## references

- Pace & Barry (1997), "Sparse spatial autoregressions", *Statistics & Probability Letters* 33(3): 291–297. the paper that introduced this dataset. [doi:10.1016/S0167-7152(96)00140-X](https://doi.org/10.1016/S0167-7152%2896%2900140-X)
- Géron (2019), *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*, 2nd ed., O'Reilly, chapter 2. the version of the data used here, with a full worked project. [github.com/ageron/handson-ml2](https://github.com/ageron/handson-ml2)
- James, Witten, Hastie & Tibshirani (2021), *An Introduction to Statistical Learning*, 2nd ed., Springer, §3.2 (multiple linear regression). [doi:10.1007/978-1-0716-1418-1](https://doi.org/10.1007/978-1-0716-1418-1)
