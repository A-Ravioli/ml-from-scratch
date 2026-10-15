# evaluating generative models · hints

open a hint only after you've been stuck for a while. this chapter is spec-driven: one hint per stage.

## stage 1

<details><summary>hint</summary>

`bits_per_dim` is one line, `nll / (n_dims * math.log(2))`, and it works for floats, arrays and tensors alike because it only uses arithmetic. for `dequantize`, draw `u = torch.rand(x.shape, generator=generator)` (float32), then return `(x.float() + u) / n_levels`. `discrete_nll` is `-log_density + n_dims * math.log(n_levels)`: the second term is the log of the box volume you divided away by scaling to $[0, 1)$.

</details>

## stage 2

<details><summary>hint</summary>

`w, V = np.linalg.eigh(A)`, then `V @ np.diag(np.sqrt(np.clip(w, 0, None))) @ V.T` (or the cheaper `(V * np.sqrt(...)) @ V.T`, which scales the columns). for the distance: `s1 = sqrtm_psd(cov1)`, `M = s1 @ cov2 @ s1`, and the trace of $M^{1/2}$ is `np.sqrt(np.clip(np.linalg.eigvalsh(M), 0, None)).sum()`. symmetrize `M` with `(M + M.T) / 2` before `eigvalsh` to remove round-off asymmetry, and return `max(float(d), 0.0)`.

</details>

## stage 3

<details><summary>hint</summary>

`fid_from_features` is `frechet_distance(f1.mean(0), np.cov(f1, rowvar=False), f2.mean(0), np.cov(f2, rowvar=False))`. for `fid_infinity`, compute the real mean and covariance once; for each `n`, `idx = rng.choice(len(fake), size=n, replace=False)`; then solve the least-squares problem with design matrix `np.c_[np.ones(len(ns)), 1 / np.asarray(ns)]` and target `fids` (`np.linalg.lstsq`); the coefficients are `(a, b)`.

</details>

## stage 4

<details><summary>hint</summary>

all pairwise squared distances at once: `(a**2).sum(1)[:, None] + (b**2).sum(1)[None, :] - 2 * a @ b.T`, clipped at 0 before the square root. for the radii, set the diagonal of the within-set distance matrix to `np.inf` (so a point isn't its own neighbour) and take the $k$-th smallest of each row with `np.partition(d, k - 1, axis=1)[:, k - 1]`. then "inside some ball" is `np.any(dist(points, centres) <= radii[None, :], axis=1)`, and precision and recall are its means with the roles of the two sets swapped.

</details>

## stage 5

<details><summary>hint</summary>

a body like `Conv(C→32) ReLU Conv(32→32) ReLU MaxPool · Conv(32→64) ReLU MaxPool · Conv(64→128) ReLU · AdaptiveAvgPool2d(1) Flatten Linear(128→feat_dim) ReLU` and `self.fc = nn.Linear(feat_dim, n_classes)`. in `embed`, remember `was = self.training`, call `self.eval()`, loop over slices of `x` under `torch.no_grad()`, concatenate `.double().cpu()` results, `self.train(was)`, and return `.numpy()`. in `cached_feature_net`, feed `X.numpy().tobytes()`, `y.numpy().tobytes()`, the settings and `inspect.getsource(FeatureNet)` (and of `train_feature_net`) into one `hashlib.sha1()`, build the path from its hex digest, and either `load_state_dict(torch.load(path))` or train and `torch.save(net.state_dict(), path)`.

</details>
