# pretraining data & compute · hints

one hint per stage. open it only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

`norm = " ".join(text.lower().split())`, then `{norm[i:i + k] for i in range(len(norm) - k + 1)}`; handle `len(norm) <= k` first (`{norm}` if non-empty, else `set()`). in `jaccard`, branch on `isinstance(a, np.ndarray)`: signatures give `float(np.mean(a == b))` after a shape check; sets give `len(a & b) / len(a | b)` with the empty-union case returning 1.0.

</details>

## stage 2

<details><summary>hint</summary>

base hash: `int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")`, reduced mod $p = 2^{31} - 1$, collected with `np.fromiter(..., dtype=np.uint64)`. draw `a` (in $[1, p)$) and `b` (in $[0, p)$) as `(n_perm, 1)` uint64 arrays from `np.random.default_rng(seed)`, then `((a * x[None, :] + b) % p).min(axis=1)` gives all `n_perm` minima at once. return `np.full(n_perm, np.iinfo(np.uint64).max, dtype=np.uint64)` for the empty set.

</details>

## stage 3

<details><summary>hint</summary>

stack the signatures into `(n_docs, n_perm)`. for each band, a dict from `sigs[i, band * r:(band + 1) * r].tobytes()` to the list of documents in that bucket; every pair inside a bucket is a candidate. union-find: `parent = list(range(n))`, `find` follows parents to the root (with path halving `parent[i] = parent[parent[i]]`), and to union two roots point the larger index at the smaller, so each root is its component's smallest index. verify with `np.mean(sigs[i] == sigs[j]) >= threshold` before the union (skip pairs already in one component). keep `[i for i in range(n) if find(i) == i]`.

</details>

## stage 4

<details><summary>hint</summary>

compute each statistic separately and append its reason in the spec's order: `len(words) < min_words`; `mean(len(w))` against `word_len`; `(text.count("#") + text.count("...") + text.count("…")) / len(words)`; `sum(any(c.isalpha() for c in w) for w in words) / len(words)`; for lines, walk the stripped non-empty lines with a `seen` set and count the ones already in it. return `(not reasons, reasons)`.

</details>

## stage 5

<details><summary>hint</summary>

`logw = np.log(n) / T` for the non-empty sources (`-inf` for empty ones), then `w = np.exp(logw - logw.max())` and `w / w.sum()`: subtracting the maximum is the log-sum-exp trick from chapter 00. for `effective_tokens`, return `float(total)` when `total <= U`; otherwise `R = total / U - 1` and `U + U * r_star * (1 - math.exp(-R / r_star))`.

</details>
