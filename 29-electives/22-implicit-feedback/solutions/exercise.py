import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    from types import SimpleNamespace

    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.utils import as_rng, log_sigmoid, sigmoid  # yours, from ch 00


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # implicit feedback · build (reference solution)

    this block is spec-driven (L4): each stage's exact specification is in the lesson's
    **spec** box next to it, the checks below enforce it, and how you structure the code is up
    to you. there are no stars here, only who watched what. you'll build the metrics that
    score a top-10 list, two models made for this kind of data (implicit ALS and BPR), and the
    evaluation protocol the lab uses to find out how misleading those metrics can be.

    | stage | you build |
    |---|---|
    | 1 | `top_k`, `recall_at_k`, `ndcg_at_k`, `user_auc`: ranking metrics |
    | 2 | `ImplicitALS`: confidence-weighted factorization of the whole matrix (Hu, Koren & Volinsky) |
    | 3 | `BPR`: pairwise ranking by SGD on (user, positive, negative) triples (Rendle et al.) |
    | 4 | `popularity_scores`, `evaluate_topk`: the baseline and the protocol |

    the data come from `make_interactions` (given, below): the simulator of topic 21 with
    play counts instead of stars. you can also run the checks from a terminal:
    `pytest -k stage1`.
    """)
    return


@app.function
def make_interactions(n_users=800, n_items=600, rank=6, *, mean_interactions=30, popularity_exponent=0.8,
                      selection_strength=2.0, rng=None):
    """given: a synthetic implicit-feedback world (see the lesson's "the data" box).

    the same latent world as topic 21: pref[u, i] = 3 + user_bias[u] + item_bias[i] + U[u] . V[i] on a
    1-5 scale, exposure weight pop[i] = (1 + popularity rank of i) ** -popularity_exponent. user u consumes
    n_u items (lognormal, mean about `mean_interactions`), drawn without replacement with probability
    proportional to pop[i] * exp(selection_strength * pref[u, i]), and plays each 1 + Poisson(exp(pref - 3))
    times. nothing else is recorded: no ratings, no dislikes.

    returns a SimpleNamespace with users, items, counts (int64 (n,)), n_users, n_items, preference
    (n_users, n_items) the true taste for every pair, and popularity (n_items,) the exposure weights.
    """
    rng = as_rng(rng)
    scale = (0.8 / rank) ** 0.25
    U = rng.normal(0.0, scale, (n_users, rank))
    V = rng.normal(0.0, scale, (n_items, rank))
    log_pop = -popularity_exponent * np.log1p(rng.permutation(n_items))
    item_bias = 0.25 * (log_pop - log_pop.mean()) / log_pop.std() + rng.normal(0.0, 0.3, n_items)
    user_bias = rng.normal(0.0, 0.4, n_users)
    pref = 3.0 + user_bias[:, None] + item_bias[None, :] + U @ V.T
    n_per = np.clip(np.round(rng.lognormal(np.log(mean_interactions) - 0.5, 1.0, n_users)), 5, n_items // 2).astype(int)
    logits = log_pop[None, :] + selection_strength * (pref - pref.mean(axis=1, keepdims=True))
    order = np.argsort(-(logits + rng.gumbel(size=logits.shape)), axis=1)
    users = np.concatenate([np.full(n, u) for u, n in enumerate(n_per)]).astype(np.int64)
    items = np.concatenate([order[u, :n] for u, n in enumerate(n_per)]).astype(np.int64)
    counts = 1 + rng.poisson(np.exp(pref[users, items] - 3.0))
    return SimpleNamespace(users=users, items=items, counts=counts.astype(np.int64), n_users=n_users,
                           n_items=n_items, preference=pref, popularity=np.exp(log_pop))


@app.function
def split_interactions(users, items, counts, n_users, n_items, *, test_frac=0.2, rng=None):
    """given: hide a random test_frac of the interactions; returns (train, test) count matrices (n_users, n_items)."""
    test = as_rng(rng).random(len(counts)) < test_frac
    train_m = np.zeros((n_users, n_items))
    test_m = np.zeros((n_users, n_items))
    train_m[users[~test], items[~test]] = counts[~test]
    test_m[users[test], items[test]] = counts[test]
    return train_m, test_m


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def top_k(scores, k, *, exclude=None):
    """stage 1: the k highest-scoring items per user, int64 (n_users, k), best first; ties to the lower item id."""
    s = np.asarray(scores, dtype=float)
    if exclude is not None:
        s = np.where(exclude, -np.inf, s)
    return np.argsort(-s, axis=1, kind="stable")[:, :k].astype(np.int64)


@app.function
def recall_at_k(top, relevant):
    """stage 1: per-user hits in `top` / min(k, #relevant), float (n_users,); nan for users with nothing relevant."""
    relevant = np.asarray(relevant, dtype=bool)
    hits = np.take_along_axis(relevant, top, axis=1).sum(axis=1)
    n_rel = relevant.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n_rel > 0, hits / np.minimum(n_rel, top.shape[1]), np.nan)


@app.function
def ndcg_at_k(top, relevant):
    """stage 1: per-user binary NDCG@k (gain 1 / log2(position + 1), position from 1), float (n_users,); nan if nothing relevant."""
    relevant = np.asarray(relevant, dtype=bool)
    k = top.shape[1]
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (np.take_along_axis(relevant, top, axis=1) * discount).sum(axis=1)
    ideal = np.concatenate([[0.0], np.cumsum(discount)])[np.minimum(relevant.sum(axis=1), k)]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ideal > 0, dcg / ideal, np.nan)


@app.function
def user_auc(scores, relevant, *, exclude=None):
    """stage 1: per-user AUC over non-excluded items (ties count 1/2), float (n_users,); nan if no relevant or no irrelevant."""
    scores = np.asarray(scores, dtype=float)
    relevant = np.asarray(relevant, dtype=bool)
    keep = np.ones_like(relevant) if exclude is None else ~np.asarray(exclude, dtype=bool)
    out = np.full(len(scores), np.nan)
    for u in range(len(scores)):
        s, rel = scores[u, keep[u]], relevant[u, keep[u]]
        n_pos, n_neg = rel.sum(), (~rel).sum()
        if n_pos == 0 or n_neg == 0:
            continue
        order = np.argsort(s, kind="stable")
        ranks = np.empty(len(s))
        ranks[order] = np.arange(1, len(s) + 1)
        # average the ranks of tied scores
        uniq, inv = np.unique(s, return_inverse=True)
        ranks = (np.bincount(inv, weights=ranks) / np.bincount(inv))[inv]
        out[u] = (ranks[rel].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return out


@app.cell(hide_code=True)
def _():
    checks.run(1, top_k=top_k, recall_at_k=recall_at_k, ndcg_at_k=ndcg_at_k, user_auc=user_auc)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class ImplicitALS:
    """Hu, Koren & Volinsky (2008): factorize preference p = [r > 0] with confidence c = 1 + alpha r over ALL cells."""

    def __init__(self, rank=16, *, alpha=10.0, lam=1.0, n_iters=10, rng=None):
        """stage 2: store rank, alpha, lam, n_iters, rng."""
        self.rank = rank
        self.alpha = alpha
        self.lam = lam
        self.n_iters = n_iters
        self.rng = rng

    def fit(self, R):
        """stage 2: R (n_users, n_items) counts >= 0. init N(0, 0.1^2) users then items; n_iters sweeps (users, then items); returns self."""
        R = np.asarray(R, dtype=float)
        rng = as_rng(self.rng)
        n_users, n_items = R.shape
        self.user_factors_ = rng.normal(0.0, 0.1, (n_users, self.rank))
        self.item_factors_ = rng.normal(0.0, 0.1, (n_items, self.rank))
        self.history_ = []
        for _ in range(self.n_iters):
            self._sweep(R, self.item_factors_, self.user_factors_)
            self._sweep(R.T, self.user_factors_, self.item_factors_)
            self.history_.append(self.loss(R))
        return self

    def _sweep(self, R, Y, X):
        """solve every row of X given Y, for the counts R (rows of R index rows of X)."""
        YtY = Y.T @ Y
        reg = self.lam * np.eye(self.rank)
        for a in range(R.shape[0]):
            idx = np.flatnonzero(R[a])
            Yi = Y[idx]
            c = 1.0 + self.alpha * R[a, idx]
            A = YtY + (Yi.T * (c - 1.0)) @ Yi + reg
            X[a] = np.linalg.solve(A, Yi.T @ c)

    def loss(self, R):
        """stage 2: sum over ALL cells of c (p - x_u . y_i)^2, plus lam (||X||^2 + ||Y||^2), a float."""
        R = np.asarray(R, dtype=float)
        C = 1.0 + self.alpha * R
        P = (R > 0).astype(float)
        err = P - self.user_factors_ @ self.item_factors_.T
        return float(np.sum(C * err ** 2) + self.lam * (np.sum(self.user_factors_ ** 2) + np.sum(self.item_factors_ ** 2)))

    def scores(self):
        """stage 2: X Y^T, float (n_users, n_items)."""
        return self.user_factors_ @ self.item_factors_.T


@app.cell(hide_code=True)
def _():
    checks.run(2, ImplicitALS=ImplicitALS)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class BPR:
    """Rendle et al. (2009): maximize sum log sigmoid(x_ui - x_uj) over (user, positive i, sampled j) with minibatch SGD."""

    def __init__(self, rank=16, *, lr=0.05, lam=0.01, n_epochs=20, batch_size=256, rng=None):
        """stage 3: store rank, lr, lam, n_epochs, batch_size, rng."""
        self.rank = rank
        self.lr = lr
        self.lam = lam
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.rng = rng

    def fit(self, R):
        """stage 3: init factors N(0, 0.1^2) (users, items), item_bias_ 0; per epoch permute the positives, batch, sample j uniformly, step."""
        R = np.asarray(R)
        rng = as_rng(self.rng)
        n_users, n_items = R.shape
        self.user_factors_ = rng.normal(0.0, 0.1, (n_users, self.rank))
        self.item_factors_ = rng.normal(0.0, 0.1, (n_items, self.rank))
        self.item_bias_ = np.zeros(n_items)
        pos_u, pos_i = np.nonzero(R > 0)
        self.history_ = []
        for _ in range(self.n_epochs):
            perm = rng.permutation(len(pos_u))
            total = 0.0
            for start in range(0, len(perm), self.batch_size):
                b = perm[start:start + self.batch_size]
                j = rng.integers(n_items, size=len(b))
                total += self.step(pos_u[b], pos_i[b], j) * len(b)
            self.history_.append(total / len(perm))
        return self

    def step(self, u, i, j):
        """stage 3: one ascent step on sum log sigmoid(x_uij) - lam ||theta||^2 for a batch (old values; repeats add up); returns mean log sigmoid(x_uij)."""
        u, i, j = np.asarray(u), np.asarray(i), np.asarray(j)
        P, Q, b = self.user_factors_, self.item_factors_, self.item_bias_
        pu, qi, qj = P[u].copy(), Q[i].copy(), Q[j].copy()
        bi, bj = b[i].copy(), b[j].copy()
        x = np.einsum("nk,nk->n", pu, qi - qj) + bi - bj
        g = sigmoid(-x)[:, None]  # d log sigmoid(x) / dx
        lr, lam = self.lr, self.lam
        np.add.at(P, u, lr * (g * (qi - qj) - lam * pu))
        np.add.at(Q, i, lr * (g * pu - lam * qi))
        np.add.at(Q, j, lr * (-g * pu - lam * qj))
        np.add.at(b, i, lr * (g[:, 0] - lam * bi))
        np.add.at(b, j, lr * (-g[:, 0] - lam * bj))
        return float(np.mean(log_sigmoid(x)))

    def scores(self):
        """stage 3: P Q^T + item_bias_, float (n_users, n_items)."""
        return self.user_factors_ @ self.item_factors_.T + self.item_bias_[None, :]


@app.cell(hide_code=True)
def _():
    checks.run(3, BPR=BPR, user_auc=user_auc)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def popularity_scores(R):
    """stage 4: every user's score for item i = number of users who interacted with i, float (n_users, n_items)."""
    R = np.asarray(R)
    counts = (R > 0).sum(axis=0).astype(float)
    return np.broadcast_to(counts, R.shape).copy()


@app.function
def evaluate_topk(scores, train, test, *, k=10):
    """stage 4: top-k excluding train items, scored against test; dict(recall, ndcg, coverage, n_users)."""
    train, test = np.asarray(train), np.asarray(test)
    top = top_k(scores, k, exclude=train > 0)
    relevant = test > 0
    rec, nd = recall_at_k(top, relevant), ndcg_at_k(top, relevant)
    ok = relevant.any(axis=1)
    return {"recall": float(np.mean(rec[ok])), "ndcg": float(np.mean(nd[ok])),
            "coverage": len(np.unique(top[ok])) / train.shape[1], "n_users": int(ok.sum())}


@app.cell(hide_code=True)
def _():
    checks.run(4, popularity_scores=popularity_scores, evaluate_topk=evaluate_topk)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. it asks which of these
    models is best, and finds that the answer depends on which question you ask.
    """)
    return


if __name__ == "__main__":
    app.run()
