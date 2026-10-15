import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # probability & random variables · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson, gives you
    something to write, and checks your code live as you edit. you're building a tiny
    probability library: a distribution object for anything with finitely many outcomes, and
    the three operations all of probability is made of (combining, conditioning,
    marginalizing). bayes' rule falls out at the end as a two-liner.

    | stage | you build |
    |---|---|
    | 1 | `Discrete`: a validated pmf with `pmf(x)` and `prob(event)` |
    | 2 | `pushforward`, `product`, `add`: functions of random variables, independence, sums |
    | 3 | `condition`: restrict to an event and renormalize |
    | 4 | `marginal`, `is_independent`: joint distributions over tuples |
    | 5 | `chain`, `bayes`: the chain rule and bayes' rule |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.class_definition
class Discrete:
    """A probability distribution over finitely many outcomes (any hashable values)."""

    def __init__(self, pmf: dict):
        """stage 1: store a validated copy of `pmf` (a dict outcome -> probability).

        every probability must be a finite number >= 0 and they must sum to 1 within 1e-9;
        otherwise raise ValueError. outcomes with probability exactly 0 are dropped.
        the result is stored as `self.probs`, a NEW dict (outcome -> float).
        """
        probs = {}
        for x, p in dict(pmf).items():
            p = float(p)
            if not np.isfinite(p) or p < 0:
                raise ValueError(f"P({x!r}) = {p} is not a valid probability")
            if p > 0:
                probs[x] = p
        total = sum(probs.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"probabilities sum to {total}, not 1")
        self.probs = probs

    @property
    def support(self) -> list:
        """the outcomes with positive probability, as a list."""
        return list(self.probs)

    def pmf(self, x) -> float:
        """stage 1: P(X = x) as a float; 0.0 for an outcome outside the support."""
        return self.probs.get(x, 0.0)

    def prob(self, event) -> float:
        """stage 1: P(X in event) as a float.

        `event` is either a predicate (a callable outcome -> bool) or a collection of outcomes
        (set, list, ...). outcomes in a collection that aren't in the support count 0.
        """
        if callable(event):
            return float(sum(p for x, p in self.probs.items() if event(x)))
        return float(sum(self.probs.get(x, 0.0) for x in set(event)))

    def __repr__(self) -> str:
        items = ", ".join(f"{x!r}: {p:.4g}" for x, p in self.probs.items())
        return f"Discrete({{{items}}})"


@app.cell(hide_code=True)
def _():
    checks.run(1, Discrete=Discrete)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def pushforward(X: Discrete, f) -> Discrete:
    """stage 2: the distribution of f(X), for a function f: outcome -> hashable value.

    P(f(X) = y) is the total probability of every outcome x with f(x) == y.
    """
    out = {}
    for x, p in X.probs.items():
        y = f(x)
        out[y] = out.get(y, 0.0) + p
    return Discrete(out)


@app.function
def product(X: Discrete, Y: Discrete) -> Discrete:
    """stage 2: the joint distribution of (X, Y) when X and Y are INDEPENDENT.

    outcomes are pairs (x, y) with probability P(X = x) * P(Y = y).
    """
    return Discrete({(x, y): px * py for x, px in X.probs.items() for y, py in Y.probs.items()})


@app.function
def add(X: Discrete, Y: Discrete) -> Discrete:
    """stage 2: the distribution of X + Y for independent X and Y (numeric outcomes).

    build it from product and pushforward.
    """
    return pushforward(product(X, Y), lambda xy: xy[0] + xy[1])


@app.cell(hide_code=True)
def _():
    checks.run(2, Discrete=Discrete, pushforward=pushforward, product=product, add=add)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def condition(X: Discrete, event) -> Discrete:
    """stage 3: the distribution of X given that `event` happened.

    `event` is a predicate or a collection of outcomes, as in Discrete.prob. outcomes outside
    the event get probability 0; the rest are rescaled by 1 / P(event).
    raises ValueError if P(event) == 0 (you can't condition on the impossible).
    """
    inside = event if callable(event) else (lambda x, _s=set(event): x in _s)
    p_event = X.prob(inside)
    if p_event == 0:
        raise ValueError("cannot condition on an event of probability 0")
    return Discrete({x: p / p_event for x, p in X.probs.items() if inside(x)})


@app.cell(hide_code=True)
def _():
    checks.run(3, Discrete=Discrete, condition=condition, add=add, product=product)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def marginal(J: Discrete, idx) -> Discrete:
    """stage 4: a marginal of a joint distribution J whose outcomes are tuples.

    idx an int: the distribution of coordinate idx (plain values, not 1-tuples).
    idx a tuple of ints: the joint distribution of those coordinates, as tuples in that order.
    """
    if isinstance(idx, (tuple, list)):
        keep = tuple(idx)
        return pushforward(J, lambda t: tuple(t[i] for i in keep))
    return pushforward(J, lambda t: t[idx])


@app.function
def is_independent(J: Discrete, tol: float = 1e-9) -> bool:
    """stage 4: True if the two coordinates of a joint J over PAIRS (x, y) are independent.

    independent means P(x, y) == P(x) P(y) (within tol) for EVERY x in the support of the
    first marginal and EVERY y in the support of the second, including pairs J never lists.
    """
    PX, PY = marginal(J, 0), marginal(J, 1)
    return all(abs(J.pmf((x, y)) - px * py) <= tol
               for x, px in PX.probs.items() for y, py in PY.probs.items())


@app.cell(hide_code=True)
def _():
    checks.run(4, Discrete=Discrete, marginal=marginal, is_independent=is_independent, product=product)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def chain(prior: Discrete, likelihood) -> Discrete:
    """stage 5: the joint distribution of (H, D) from P(H) and P(D | H), by the chain rule.

    `likelihood(h)` returns a Discrete: the distribution of the data D when the hypothesis
    is h. outcomes of the result are pairs (h, d) with probability P(h) * P(d | h).
    """
    return Discrete({(h, d): ph * pd for h, ph in prior.probs.items()
                     for d, pd in likelihood(h).probs.items()})


@app.function
def bayes(prior: Discrete, likelihood, observed) -> Discrete:
    """stage 5: the posterior P(H | D = observed), a Discrete over hypotheses.

    build the joint with `chain`, condition on the data, then marginalize out the data.
    raises ValueError if the observation has probability 0 under every hypothesis.
    """
    joint = chain(prior, likelihood)
    return marginal(condition(joint, lambda hd: hd[1] == observed), 0)


@app.cell(hide_code=True)
def _():
    checks.run(5, Discrete=Discrete, chain=chain, bayes=bayes)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, try a puzzle of your own in a scratch cell (the two-child
    problem is a good one), then open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
