# 02 · capstone rubric

score each criterion 0–3. a solid capstone scores at least 9 out of 12, with no zeros.

| criterion | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **correctness** | tests fail | most tests pass | all tests pass, PCA fitted on training data only | all pass, plus your own extra checks (e.g. your `svd` against `np.linalg.svd` on the real data, orthonormality of your eigenfaces) |
| **results** | no baseline | accuracy reported against the majority-class baseline | meets the ≥ 0.85 threshold on real faces at a k chosen on the training set, with a standard error, and compared with raw-pixel 1-NN | accuracy across the whole k grid with error bars, and a clear statement of whether differences exceed the uncertainty |
| **analysis** | none | describes the plots | explains the spectrum (variance vs k) and one ablation (dropping leading components, skipping centering, …) with a reason | several ablations, a failure analysis of misclassified faces, and a link back to the theory (Eckart–Young, $\sigma_i^2/(n-1)$ as variance) |
| **clarity** | hard to follow | readable | clear figures with labeled axes, and a narrative that answers the five prompts | a peer could reproduce every number from the write-up alone (split, k, config, runtime) |

## what graders look for

- **leakage.** the mean face and the eigenfaces must come from the training images only. fitting PCA on all 400 images and then "testing" is the classic mistake, and it scores at most 1 on results.
- **choosing k on the test set.** picking the $k$ with the best *test* accuracy is leakage too. choose it on the training set (the notebook's leave-one-out) and report the test accuracy once.
- **honest uncertainty.** with 120 test images and accuracy near 0.89, the standard error is about $\sqrt{0.89 \cdot 0.11/120} \approx 0.028$, so differences of a few points between settings are noise. say so.
- **the right comparison.** "eigenfaces get 0.89" means little without "raw-pixel 1-NN also gets 0.89, using 4096 numbers per face; eigenfaces get it with 20".
- **connecting to the chapter.** the best write-ups explain *why* the reduced SVD of the $n \times d$ data matrix is the efficient route when $n \ll d$, and what the singular values say about how compressible faces are.
