# 02 · capstone: eigenfaces

## goal

build a face recognizer using nothing but this chapter's linear algebra. a 64 × 64 face image is a point in $\mathbb{R}^{4096}$, but faces don't fill that space: they lie near a low-dimensional subspace spanned by a few dozen "eigenfaces", the principal components of the training images. you'll compute that subspace with **your own** `mlfs.linalg.svd`, describe every face by its handful of coordinates in it, and recognize a new face by finding the nearest training face in those coordinates. this is the method of Sirovich & Kirby (1987) and Turk & Pentland (1991), one of the first face-recognition systems that worked, and still the cleanest example of PCA doing something useful.

along the way you'll see Eckart–Young (topic 06) as a picture, connect singular values to explained variance, and run the evaluation of a classifier with honest baselines and uncertainty.

## data

the [Olivetti faces](https://cam-orl.co.uk/facedatabase.html) (AT&T Laboratories Cambridge): 400 grayscale images of 40 people, 10 each, with variation in lighting, expression and pose. `mlfs` loads Sam Roweis's widely used 64 × 64 version of the set, not the original 92 × 112 images, so your numbers may differ from papers that used the full-size originals. load them with

```python
from mlfs import datasets
images, labels = datasets.olivetti_faces()   # float64 (400, 64, 64) in [0, 1]; int64 (400,)
```

the first call downloads the file once and caches it under `~/.cache/mlfs/`. with `MLFS_OFFLINE=1`, or with no network, you get a small synthetic stand-in with the same format (200 images, 20 "people"), so the notebook always runs. the tests never touch either: they generate their own synthetic faces.

the notebook splits each person's images 70/30 (7 train and 3 test per person for the real data). all fitting (the mean face and the eigenfaces) uses the training images only.

## configurations

| config | data | sizes | runtime (laptop CPU) |
|---|---|---|---|
| **small** (tests, `verify.py`, offline) | synthetic stand-in, or the tests' own generator | 131 train × 4096 pixels (or 105 × 64 in the tests); k up to 60 | a few seconds |
| **full** (your write-up) | real Olivetti faces | 280 train × 4096 pixels; k up to 200 | under a minute with the reference `svd`. your own may be slower |

the notebook picks the config automatically from `datasets.is_offline()`.

## deliverables

1. `capstone.py` with the four required functions implemented and their checks green:
   - `fit_pca(X, k)` → `(mean, components, explained_variance)` via your `svd` on centered data
   - `project(X, mean, components)` and `reconstruct(Z, mean, components)`
   - `nearest_neighbor(Z_train, y_train, Z_test)`: vectorized 1-NN
   - `eigenface_classifier(X_train, y_train, X_test, k)`: the full pipeline
2. the write-up section at the end of the notebook, answering its five prompts (results with uncertainty, the spectrum, one ablation, failure analysis, and where the linear algebra entered).

choose $k$ with the training data only. the notebook does it by leave-one-out 1-NN accuracy on the training images, and reports the test accuracy once, at that $k$.

## requirements

| check | threshold |
|---|---|
| tests in `test_capstone.py` pass | all (`pytest test_capstone.py`) |
| held-out accuracy on the tests' synthetic faces at k = 20 | ≥ 0.90 (enforced by `test_stage4_accuracy_threshold`) |
| held-out accuracy on the real Olivetti faces (full config), at the k chosen on the training set | ≥ 0.85, reported with its standard error |
| comparison with baselines | report 1-NN on raw pixels and the majority class next to your result |

with the notebook's fixed 7/3 split, expect about **0.89** on the real data (0.892 ± 0.028 at the k that leave-one-out picks, k = 80 for the reference; any k from about 20 to 200 gives the same). random 7/3 splits give anywhere from about 0.85 to 0.93, which shows how much a 120-image test set can wobble. raw-pixel 1-NN also scores about 0.89 on this split: PCA's win here is a representation 100× smaller with no loss of accuracy, not higher accuracy. explaining that honestly is part of the write-up.

## stretch goals

- **fisherfaces.** project onto the top eigenfaces, then onto the directions that best separate *people* (linear discriminant analysis, a generalized symmetric eigenproblem you can solve with your `cholesky` and `eigh`). compare with plain eigenfaces (Belhumeur, Hespanha & Kriegman, 1997).
- **drop the first components.** the top eigenfaces often encode lighting, not identity. measure accuracy when you discard the first 1, 2 or 3.
- **reconstruction as anomaly detection.** a non-face (a shuffled image, a photo of something else) has a large reconstruction error in face space. plot the distribution of errors for faces versus non-faces.
- **speed.** time your `svd` on the $280 \times 4096$ centered data against an eigendecomposition (your `eigh`) of the $280 \times 280$ Gram matrix $X_cX_c^\top$, and against the $4096 \times 4096$ covariance. explain the differences in cost and accuracy.

## self-assessment

grade your write-up with [`rubric.md`](rubric.md).

## references

- Sirovich & Kirby (1987), "Low-dimensional procedure for the characterization of human faces", *JOSA A* 4(3): 519–524. [doi:10.1364/JOSAA.4.000519](https://doi.org/10.1364/JOSAA.4.000519)
- Turk & Pentland (1991), "Eigenfaces for recognition", *Journal of Cognitive Neuroscience* 3(1): 71–86. [doi:10.1162/jocn.1991.3.1.71](https://doi.org/10.1162/jocn.1991.3.1.71)
- Belhumeur, Hespanha & Kriegman (1997), "Eigenfaces vs. Fisherfaces: recognition using class specific linear projection", *IEEE TPAMI* 19(7): 711–720. [doi:10.1109/34.598228](https://doi.org/10.1109/34.598228)
- Samaria & Harter (1994), "Parameterisation of a stochastic model for human face identification", *Proceedings of the 2nd IEEE Workshop on Applications of Computer Vision*: 138–142. the paper that introduced the Olivetti/ORL face set. [doi:10.1109/ACV.1994.341300](https://doi.org/10.1109/ACV.1994.341300)
