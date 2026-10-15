import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import heapq
    import math

    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.linalg import cho_solve, cholesky  # yours, from ch 02 · topic 03


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # information theory · build (reference solution)

    work through the stages in order. you'll build the core quantities of information theory
    for discrete distributions and for Gaussians, then a Huffman coder that turns entropy from
    a formula into an actual number of bits. the information measures become part of
    `mlfs.stats`: cross-entropy is the loss you'll train classifiers with, KL divergence shows
    up in variational inference and RL, and mutual information in representation learning.

    | stage | you build |
    |---|---|
    | 1 | `entropy`, `cross_entropy`, `kl_divergence`: discrete distributions, in nats or bits |
    | 2 | `mutual_information`: how much knowing $X$ tells you about $Y$ |
    | 3 | `gaussian_entropy`, `gaussian_kl`: the same ideas for multivariate Gaussians |
    | 4 | `huffman_code`, `expected_length`: an optimal prefix code |
    | 5 | `encode`, `decode`: compress and decompress a message |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def normalize(p) -> np.ndarray:
    """turn nonnegative weights p into a float probability array that sums to 1 (ValueError otherwise)."""
    p = np.asarray(p, dtype=float)
    if np.any(p < 0) or not np.isfinite(p).all() or p.sum() <= 0:
        raise ValueError("probabilities must be finite, nonnegative, and not all zero")
    return p / p.sum()


@app.function
def entropy(p, *, base=None) -> float:
    """stage 1: H(p) = -sum p log p of a discrete distribution p (k,), normalized first (use normalize).

    terms with p = 0 contribute 0. natural log (nats) unless base is given (base=2 -> bits).
    """
    p = normalize(p)
    nz = p[p > 0]
    h = float(-np.sum(nz * np.log(nz)))
    h = max(h, 0.0)
    return h / math.log(base) if base is not None else h


@app.function
def cross_entropy(p, q, *, base=None) -> float:
    """stage 1: H(p, q) = -sum p log q, with p and q (k,) normalized first.

    terms with p = 0 contribute 0; returns inf if q = 0 anywhere p > 0.
    """
    p, q = normalize(p), normalize(q)
    nz = p > 0
    if np.any(q[nz] == 0):
        return math.inf
    h = float(-np.sum(p[nz] * np.log(q[nz])))
    return h / math.log(base) if base is not None else h


@app.function
def kl_divergence(p, q, *, base=None) -> float:
    """stage 1: KL(p || q) = sum p log(p / q) >= 0, with p and q (k,) normalized first.

    terms with p = 0 contribute 0; inf if q = 0 anywhere p > 0. clip tiny negative rounding to 0.
    """
    p, q = normalize(p), normalize(q)
    nz = p > 0
    if np.any(q[nz] == 0):
        return math.inf
    d = max(float(np.sum(p[nz] * (np.log(p[nz]) - np.log(q[nz])))), 0.0)
    return d / math.log(base) if base is not None else d


@app.cell(hide_code=True)
def _():
    checks.run(1, entropy=entropy, cross_entropy=cross_entropy, kl_divergence=kl_divergence)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def mutual_information(joint, *, base=None) -> float:
    """stage 2: I(X; Y) of a joint probability table joint (kx, ky), normalized first.

    I = sum over cells with joint > 0 of p(x, y) log(p(x, y) / (p(x) p(y))) = KL(joint || product of marginals).
    """
    pxy = normalize(joint)
    if pxy.ndim != 2:
        raise ValueError(f"joint must be a 2-D table, got shape {pxy.shape}")
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    nz = pxy > 0
    mi = float(np.sum(pxy[nz] * (np.log(pxy[nz]) - np.log((px * py)[nz]))))
    mi = max(mi, 0.0)
    return mi / math.log(base) if base is not None else mi


@app.cell(hide_code=True)
def _():
    checks.run(2, mutual_information=mutual_information)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def gaussian_entropy(cov) -> float:
    """stage 3: the differential entropy (nats) of N(mu, cov): (d/2) log(2 pi e) + (1/2) log det cov.

    cov is (d, d) symmetric positive definite (a scalar is a 1-D variance). use your cholesky:
    log det cov = 2 * sum(log(diag(L))).
    """
    cov = np.atleast_2d(np.asarray(cov, dtype=float))
    d = cov.shape[0]
    L = cholesky(cov)
    logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
    return 0.5 * d * math.log(2 * math.pi * math.e) + 0.5 * logdet


@app.function
def gaussian_kl(mu0, cov0, mu1, cov1) -> float:
    """stage 3: KL(N(mu0, cov0) || N(mu1, cov1)) in nats.

    (1/2) [ tr(cov1^-1 cov0) + (mu1 - mu0)^T cov1^-1 (mu1 - mu0) - d + log det cov1 - log det cov0 ].
    mu (d,), cov (d, d) SPD (scalars allowed for d = 1). use cholesky / cho_solve, not an explicit inverse.
    """
    mu0 = np.atleast_1d(np.asarray(mu0, dtype=float))
    mu1 = np.atleast_1d(np.asarray(mu1, dtype=float))
    cov0 = np.atleast_2d(np.asarray(cov0, dtype=float))
    cov1 = np.atleast_2d(np.asarray(cov1, dtype=float))
    d = len(mu0)
    L0, L1 = cholesky(cov0), cholesky(cov1)
    trace = float(np.trace(cho_solve(L1, cov0)))
    diff = mu1 - mu0
    maha = float(diff @ cho_solve(L1, diff))
    logdet0 = 2.0 * float(np.sum(np.log(np.diag(L0))))
    logdet1 = 2.0 * float(np.sum(np.log(np.diag(L1))))
    return max(0.5 * (trace + maha - d + logdet1 - logdet0), 0.0)


@app.cell(hide_code=True)
def _():
    checks.run(3, gaussian_entropy=gaussian_entropy, gaussian_kl=gaussian_kl)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def huffman_code(probs) -> list:
    """stage 4: a Huffman code for symbols 0..k-1 with probabilities probs (k,). returns k bit strings.

    repeatedly merge the two least probable nodes (heapq); prefix '0' to every codeword in one and '1' in
    the other. break ties between equal probabilities with an insertion counter, so heapq never compares
    lists. a single symbol gets the code ['0'].
    """
    p = normalize(probs)
    k = len(p)
    if k == 1:
        return ["0"]
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


@app.function
def expected_length(probs, code) -> float:
    """stage 4: the expected codeword length sum p_i len(code[i]) in bits per symbol (probs normalized)."""
    p = normalize(probs)
    return float(sum(pi * len(c) for pi, c in zip(p, code)))


@app.cell(hide_code=True)
def _():
    checks.run(4, huffman_code=huffman_code, expected_length=expected_length)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def encode(symbols, code) -> str:
    """stage 5: concatenate the codewords of a sequence of symbol indices. returns a string of '0'/'1'."""
    return "".join(code[int(s)] for s in symbols)


@app.function
def decode(bits: str, code) -> list:
    """stage 5: invert encode for a prefix-free code: read bits left to right, emitting a symbol index
    each time the bits read so far form a codeword. raises ValueError if bits end mid-codeword."""
    lookup = {c: i for i, c in enumerate(code)}
    out, cur = [], ""
    for b in bits:
        cur += b
        if cur in lookup:
            out.append(lookup[cur])
            cur = ""
    if cur:
        raise ValueError(f"bits end in the middle of a codeword: leftover {cur!r}")
    return out


@app.cell(hide_code=True)
def _():
    checks.run(5, encode=encode, decode=decode)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab (`marimo edit lab.py`) and watch a real
    compressor close in on the entropy.
    """)
    return


if __name__ == "__main__":
    app.run()
