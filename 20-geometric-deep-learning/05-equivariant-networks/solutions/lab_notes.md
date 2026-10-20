# equivariant networks · lab notes

numbers measured with the reference solution (seed 0) on a laptop CPU. the molecules are generated in the notebook, so `verify.py`'s offline run gives the same numbers. the six-atom energies have a mean absolute deviation of 0.50 around their mean (the "predict the mean" baseline).

## experiment 1: turn the test molecules

**what people usually predict:** that the MLP is good on standard-orientation molecules and bad on rotated ones; that augmentation fixes the rotated case at little cost; that the EGNN is about as good as the MLP, just more robust.

**what happens** (test MAE, 1,000 molecules):

| model | standard orientation | randomly rotated |
|---|---|---|
| MLP | 0.353 | 0.460 |
| MLP + rotation augmentation | 0.400 | 0.399 |
| EGNN | **0.0187** | **0.0187** |

predicting the mean gives 0.504. the plain MLP, trained on 2,000 molecules, is only 30% better than the mean on molecules in their standard orientation, and almost no better than the mean on rotated ones. augmentation makes it orientation-independent but worse on standard-orientation molecules. the EGNN's error is **19 times smaller** than the best MLP's on upright molecules and 21 times smaller on rotated ones, and its two columns agree to every digit. retraining with seeds 1 and 2 gives the same picture: EGNN 0.024 and 0.019, best MLP 0.367 and 0.368 upright, augmented MLP 0.400 and 0.397 rotated, so the factor is 15–19× upright and 17–21× rotated across seeds 0–2.

**why:** the energy is a sum over 15 atom pairs of a Gaussian of their distance, which is a complicated function of 18 raw coordinates: to compute it, the MLP has to rediscover subtraction, squaring and the distance formula from 2,000 examples. the canonical orientation gives it a narrow slice of input space to fit; rotated molecules fall outside that slice, where its fit means nothing. augmentation spreads the training data over all orientations, so the MLP must fit the same function on a 3-dimensionally larger input space with the same data and capacity: it becomes consistent and uniformly mediocre. the EGNN sees distances directly and can't depend on orientation, so the symmetry costs it nothing; its two columns are equal because the output is mathematically invariant (stage 3's check), not because it learned anything about rotations. the big lesson isn't robustness, it's accuracy: building the symmetry in removes most of what the MLP has to learn.

## experiment 2: how many molecules do you need?

**what people usually predict:** a constant factor between the curves that closes as data grows, "the MLP catches up eventually".

**what happens** (test MAE on rotated molecules, 600 training steps):

| model | 50 | 200 | 800 | 3,200 |
|---|---|---|---|---|
| MLP + augmentation | 0.497 | 0.423 | 0.395 | 0.397 |
| EGNN | 0.074 | 0.020 | 0.019 | 0.017 |

with 50 molecules the EGNN is already 5 times better than the augmented MLP with 3,200. the MLP's curve is almost flat from 800 on (0.40, about 20% better than the mean); the EGNN's drops fast and then flattens around 0.017.

**why:** for the MLP, each training molecule seen in random orientations is a cloud of different inputs that must all map to the same output; learning that the function depends on distances only, from scratch, takes far more than 3,200 molecules. its curve is flat from 800 on, which suggests that at this budget (600 steps, two hidden layers of 128) it's limited by optimization and capacity rather than data; we haven't verified that (try more steps or a wider MLP). the EGNN starts from the right function class, so 50 molecules already pin down a good deal; its flattening around 0.017 is likely the same kind of budget limit. the gap between the curves is the value of the inductive bias, and it's largest exactly where data is scarce, which is the usual situation in chemistry, where every label is a quantum-chemistry calculation.

## experiment 3: left hand, right hand

**what people usually predict:** that the EGNN, the better model in experiments 1 and 2, learns this simple rotation-invariant target too, probably better than the MLP.

**what happens:**

| model | test R² | max \|f(mol) − f(mirror)\| | std of predictions |
|---|---|---|---|
| MLP + rotation augmentation | 0.838 | 32.8 | 4.34 |
| EGNN | **0.000** | **0.0** | 0.040 |

(the true signed volume has standard deviation 4.94.) the augmented MLP learns the signed volume reasonably ($R^2 = 0.84$). the EGNN learns *nothing*: $R^2 = 0.000$, its predictions are a near-constant close to 0 (the flat line in the plot), and it gives every molecule and its mirror image exactly the same output.

**why:** the EGNN is invariant to all of $E(3)$, including reflections, so $f(X) = f(\text{mirror}(X))$ exactly (the middle column's 0.0 is a theorem, like topic 04's). the signed volume flips sign under reflection, so the best an invariant model can do is predict the average of $V$ and $-V$, which is 0 (on paper 3); training finds that, and $R^2 = 0$ is the best possible result, not a training failure. the MLP is only made approximately invariant to *proper* rotations by its augmentation, which preserve the signed volume, so nothing stops it from learning handedness. the fix for chiral tasks is to build in the right group, $SE(3)$, with a reflection-odd feature such as a triple product (check your understanding 6). this is the third time the chapter has shown that a symmetry is only a free lunch when the task really has it.
