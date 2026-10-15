import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.base import Estimator  # yours, from chapter 07
    from mlfs.utils import as_rng  # yours, from chapter 00


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # hidden markov models · build

    work through the stages in order. an HMM has $K$ hidden states and $M$ observable
    symbols: a start distribution `startprob_` $\pi$ (K,), a transition matrix `transmat_`
    $A$ (K, K) with $A_{ij} = P(z_{t+1} = j \mid z_t = i)$, and an emission matrix
    `emissionprob_` $B$ (K, M) with $B_{jm} = P(x_t = m \mid z_t = j)$. `HMM` becomes part of
    `mlfs.bayes`.

    | stage | you build |
    |---|---|
    | 1 | `HMM.__init__`, `HMM.sample`: the generative story |
    | 2 | `HMM.forward_backward`: the likelihood and the posterior over each hidden state (scaled) |
    | 3 | `HMM.viterbi`: the single most probable hidden path (in log space) |
    | 4 | `HMM.fit` (Baum–Welch) and `HMM.score` → `mlfs.bayes`: learning the parameters with EM |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def as_sequences(X):
    """one int sequence -> [array]; a list of sequences -> list of arrays."""
    if isinstance(X, np.ndarray) and X.ndim == 1:
        return [X.astype(np.int64)]
    if isinstance(X, (list, tuple)) and len(X) and np.ndim(X[0]) == 0:
        return [np.asarray(X, dtype=np.int64)]
    return [np.asarray(s, dtype=np.int64) for s in X]


@app.class_definition
class HMM(Estimator):
    """a discrete-emission hidden Markov model with K states and M symbols.

    parameters (set by fit, or directly): startprob_ (K,), transmat_ (K, K), emissionprob_ (K, M).
    observations are int sequences with values in 0..M-1.
    """

    def __init__(self, n_states, n_symbols, *, max_iters=100, tol=1e-06, rng=None):
        """stage 1: store the hyperparameters unchanged."""
        raise NotImplementedError("stage1: implement HMM.__init__")

    def sample(self, n_steps, *, rng=None):
        """stage 1: draw (obs (n_steps,), states (n_steps,)) int arrays from the model."""
        raise NotImplementedError("stage1: implement HMM.sample")

    def forward_backward(self, obs):
        """stage 2: (log p(obs), posteriors (T, K) with row t = P(z_t | obs)), with scaled messages."""
        raise NotImplementedError("stage2: implement HMM.forward_backward")

    def scaled_messages(self, obs):
        """stage 2: scaled messages (alpha (T, K), beta (T, K), c (T,)): alpha[t] = P(z_t | x_1..t), c[t] = P(x_t | x_1..t-1)."""
        raise NotImplementedError("stage2: implement HMM.scaled_messages")

    def viterbi(self, obs):
        """stage 3: the most probable state path (T,) and its log joint probability log p(obs, path)."""
        raise NotImplementedError("stage3: implement HMM.viterbi")

    def score(self, X, y=None):
        """stage 4: the total log-likelihood of one int sequence or a list of them."""
        raise NotImplementedError("stage4: implement HMM.score")

    def fit(self, X, y=None):
        """stage 4: Baum-Welch (EM) on one sequence or a list; starts from the current parameters if all three are set."""
        raise NotImplementedError("stage4: implement HMM.fit")


@app.cell(hide_code=True)
def _():
    checks.run(1, HMM=HMM)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    *`forward_backward` is a method of `HMM` above.*
    """)
    return


@app.cell(hide_code=True)
def _():
    checks.run(2, HMM=HMM)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    *`viterbi` is a method of `HMM` above.*
    """)
    return


@app.cell(hide_code=True)
def _():
    checks.run(3, HMM=HMM)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    *`fit` and `score` are methods of `HMM` above.*
    """)
    return


@app.cell(hide_code=True)
def _():
    checks.run(4, HMM=HMM)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `HMM` is part of `mlfs.bayes` (try
    `from mlfs.bayes import HMM` in a scratch cell). then open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
