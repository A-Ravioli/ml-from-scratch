# 20 · capstone rubric

score each criterion 0–3. a solid capstone scores at least 9 out of 12, with no zeros.

| criterion | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **correctness** | fewer than half the checks pass | most checks pass | all checks pass, and the full config's EGNN reaches test MAE $\le 0.25$ at 4,000 training molecules | all of that, plus your own checks for something the provided ones don't test (e.g. that forces from `autograd` of your EGNN energy are rotation-equivariant, or that `gin_pos` becomes invariant if you canonicalize the frame) |
| **results** | no test numbers | MAEs without intervals or baseline | all five rows (per-atom mean, GCN, GIN, `gin_pos`, EGNN) at 4,000 molecules with 95% bootstrap intervals and the 3-seed spread, the learning curves, and the invariance audit of every model | all of that, plus paired intervals for the comparisons you make claims about, and a statement of which differences are larger than both the test-set uncertainty and the seed spread |
| **analysis** | numbers without explanation | describes which model wins | explains each model's result by what it can see (topology, raw coordinates, distances) and connects the audit to the errors | all of that plus at least one ablation that tests an explanation (e.g. GIN with distances as features, a smaller graph cutoff, rotation augmentation for `gin_pos`, a second task beyond 1-WL) and an honest account of what didn't work |
| **clarity** | hard to follow | readable | one table, one learning-curve figure, one audit table, each with config and units | a reader could rerun your full config from the write-up alone (sizes, epochs, learning rate, cutoff, seeds, runtime, machine) |

## what graders look for

- **baselines first.** every MAE sits next to the per-atom-mean baseline, which knows the number of atoms and nothing else. "GIN beats it by a third" is a result; "GIN has MAE 0.6" isn't.
- **uncertainty that matches the claim.** a bootstrap interval over test molecules says how much the number would move with a different test set; the seed spread says how much it would move with a different training run. a claim that one model beats another needs the paired interval of the difference to exclude zero, and a difference smaller than the seed spread deserves a caveat.
- **the audit read correctly.** $10^{-12}$ and $10^{-1}$ mean different things: exact invariance versus none. the strongest write-ups connect the audit to the error: a model whose prediction moves by 0.3 when you rotate the molecule can't have a test MAE much below that on randomly oriented molecules.
- **mechanisms.** "EGNN is better" is a ranking; "GCN and GIN only see which atoms are within 2σ of each other, so they can count contacts but not tell a compressed contact (strongly repulsive) from a relaxed one; EGNN's messages see the squared distance" is an explanation, and it predicts what an ablation will show.
- **honesty about the substitution.** the benchmark is synthetic. say what that buys (exact labels, known symmetries, unlimited data) and what it doesn't (real chemistry, bond topology, long-range effects), and what you'd expect to change on QM9.
