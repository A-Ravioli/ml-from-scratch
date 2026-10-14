# information theory · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

normalize with the provided `normalize`, then sum only over the entries where `p > 0` (a boolean mask). for cross-entropy and KL, check first whether `q` is 0 anywhere `p` is positive. convert to another base by dividing by `math.log(base)`.

</details>

<details><summary>hint 2: the approach</summary>

entropy: `nz = p[p > 0]`, `-np.sum(nz * np.log(nz))`. cross-entropy: with `m = p > 0`, return `math.inf` if `np.any(q[m] == 0)`, else `-np.sum(p[m] * np.log(q[m]))`. KL: same check, then `np.sum(p[m] * (np.log(p[m]) - np.log(q[m])))`, clipped below at 0.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
def kl_divergence(p, q, *, base=None):
    p, q = normalize(p), normalize(q)
    nz = p > 0
    if np.any(q[nz] == 0):
        return math.inf
    d = max(float(np.sum(p[nz] * (np.log(p[nz]) - np.log(q[nz])))), 0.0)
    return d / math.log(base) if base is not None else d
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

mutual information is the KL divergence from the joint table to the product of its marginals. get the marginals by summing the table along each axis.

</details>

<details><summary>hint 2: the approach</summary>

`px = pxy.sum(axis=1, keepdims=True)` (shape `(kx, 1)`) and `py = pxy.sum(axis=0, keepdims=True)` (shape `(1, ky)`), so `px * py` broadcasts to the full product table. sum `pxy * (log pxy - log(px * py))` over the cells where `pxy > 0`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
pxy = normalize(joint)
px = pxy.sum(axis=1, keepdims=True)
py = pxy.sum(axis=0, keepdims=True)
nz = pxy > 0
mi = float(np.sum(pxy[nz] * (np.log(pxy[nz]) - np.log((px * py)[nz]))))
mi = max(mi, 0.0)
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

turn scalars into `(1, 1)` matrices and `(1,)` vectors with `np.atleast_2d` / `np.atleast_1d`. everything you need from a covariance comes from its Cholesky factor: the log-determinant, and solves with `cho_solve`.

</details>

<details><summary>hint 2: the approach</summary>

`logdet = 2 * np.sum(np.log(np.diag(L)))`. for the KL: `trace = np.trace(cho_solve(L1, cov0))` is $\operatorname{tr}(\Sigma_1^{-1}\Sigma_0)$, and `diff @ cho_solve(L1, diff)` is the Mahalanobis term. combine with $-d$ and $\log\det\Sigma_1 - \log\det\Sigma_0$, all times 1/2.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
L0, L1 = cholesky(cov0), cholesky(cov1)
trace = float(np.trace(cho_solve(L1, cov0)))
diff = mu1 - mu0
maha = float(diff @ cho_solve(L1, diff))
logdet0 = 2.0 * float(np.sum(np.log(np.diag(L0))))
logdet1 = 2.0 * float(np.sum(np.log(np.diag(L1))))
return 0.5 * (trace + maha - d + logdet1 - logdet0)
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

keep a min-heap of nodes. a node is a probability plus the list of symbols it contains. each merge prepends one bit to the codeword of every symbol in each of the two merged nodes. handle `k == 1` separately.

</details>

<details><summary>hint 2: the approach</summary>

start with `heap = [(p[i], i, [i]) for i in range(k)]` and `heapq.heapify(heap)`; the middle element is a unique counter, so ties never compare the lists. loop while more than one node remains: pop two, prepend `'0'` to the codes of the first's symbols and `'1'` to the second's, push `(p0 + p1, counter, s0 + s1)` with a fresh counter.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
codes = [""] * k
heap = [(float(p[i]), i, [i]) for i in range(k)]
heapq.heapify(heap)
counter = k
while len(heap) > 1:
    p0, _, s0 = heapq.heappop(heap)
    p1, _, s1 = heapq.heappop(heap)
    for i in s0:
        codes[i] = "0" + codes[i]
    for i in s1:
        codes[i] = "1" + codes[i]
    heapq.heappush(heap, (p0 + p1, counter, s0 + s1))
    counter += 1
return codes
```

</details>

## stage 5

<details><summary>hint 1: a nudge</summary>

encoding is a `"".join`. for decoding, build a dictionary from codeword to symbol, and grow a buffer one bit at a time.

</details>

<details><summary>hint 2: the approach</summary>

`lookup = {c: i for i, c in enumerate(code)}`. for each bit, `cur += b`; if `cur in lookup`, append `lookup[cur]` and reset `cur = ""`. after the loop, a non-empty `cur` means the bits ended mid-codeword: raise `ValueError`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
lookup = {c: i for i, c in enumerate(code)}
out, cur = [], ""
for b in bits:
    cur += b
    if cur in lookup:
        out.append(lookup[cur])
        cur = ""
if cur:
    raise ValueError("bits end in the middle of a codeword")
return out
```

</details>
