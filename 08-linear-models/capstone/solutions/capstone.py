import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import re
    from collections import Counter

    import marimo as mo
    import numpy as np

    from mlfs import checks, datasets
    from mlfs.base import DummyClassifier, Transformer  # yours, from ch 07
    from mlfs.linear import LogisticRegression  # yours, from topic 04
    from mlfs.metrics import (  # yours, from ch 07
        accuracy_score,
        f1_score,
        make_scorer,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from mlfs.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split  # ch 07
    from mlfs.preprocessing import make_pipeline  # yours, from ch 07
    from mlfs.stats import bootstrap_ci, permutation_test  # yours, from ch 05

    # small CPU config (offline: tests, verify) vs full config (your write-up)
    SMALL = datasets.is_offline()
    N_BOOT = 500 if SMALL else 2_000
    ALPHAS = [1e-5, 1e-4, 1e-3, 1e-2] if SMALL else [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    MAX_FEATURES = None if SMALL else 1_000  # the IRLS Hessian is (V+1) x (V+1)
    F1 = make_scorer(f1_score)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 08 · capstone: a text classifier from raw text (reference solution)

    read [`brief.md`](brief.md) first. you'll turn raw SMS messages into numbers (a tokenizer,
    a bag of words, TF-IDF weights), classify them with your own logistic regression, report
    the result with baselines and confidence intervals, and find out what the model learned
    and where it fails.

    | stage | you build |
    |---|---|
    | 1 | `tokenize`: raw text to a list of tokens |
    | 2 | `CountVectorizer`: a vocabulary and a bag-of-words matrix (with n-grams) |
    | 3 | `TfidfTransformer`: TF-IDF weighting and row normalization |
    | 4 | `text_classifier`: the whole pipeline, tokenizer to `LogisticRegression` |
    | 5 | `metric_with_ci`, `paired_comparison`: numbers with uncertainty |
    | 6 | `top_features`, `worst_errors`: what the model learned, and where it's wrong |

    the data loading is given. the experiments after the stages use your functions; the
    write-up prompts are at the end.
    """)
    return


@app.function
def tokenize(text):
    """stage 1: lowercase `text` and return its tokens, in order: every maximal run of the
    characters a-z and 0-9, and every single currency sign £, $ or €, as separate tokens
    (regex [a-z0-9]+|[£$€]). everything else (spaces, punctuation, apostrophes) separates
    tokens. returns a list of str."""
    return re.findall(r"[a-z0-9]+|[£$€]", str(text).lower())


@app.cell(hide_code=True)
def _():
    checks.run(1, _tests="test_capstone.py", tokenize=tokenize)
    return


@app.function
def ngrams(tokens, ngram_range):
    """given: all n-grams of a token list for n in ngram_range = (lo, hi), joined by single
    spaces, unigrams first: ngrams(["a", "b", "c"], (1, 2)) -> ["a", "b", "c", "a b", "b c"]."""
    lo, hi = ngram_range
    out = []
    for n in range(lo, hi + 1):
        out += [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return out


@app.class_definition
class CountVectorizer(Transformer):
    """bag of words: each document becomes a vector of term counts over a learned vocabulary."""

    def __init__(self, *, min_df=1, max_features=None, ngram_range=(1, 1), binary=False):
        self.min_df = min_df
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.binary = binary

    def fit(self, X, y=None):
        """stage 2: learn the vocabulary from an iterable of strings; returns self.

        terms are ngrams(tokenize(doc), ngram_range). keep the terms that appear in at least
        min_df documents; then, if max_features is set, only the max_features terms with the
        largest total count over all documents (ties: alphabetical order). sets terms_ (an
        array of the kept terms in ALPHABETICAL order: column j is terms_[j]) and vocabulary_
        (a dict term -> column).
        """
        df, total = Counter(), Counter()
        for doc in X:
            terms = ngrams(tokenize(doc), self.ngram_range)
            total.update(terms)
            df.update(set(terms))
        kept = [t for t in df if df[t] >= self.min_df]
        if self.max_features is not None:
            kept = sorted(kept, key=lambda t: (-total[t], t))[: self.max_features]
        self.terms_ = np.array(sorted(kept), dtype=object)
        self.vocabulary_ = {t: j for j, t in enumerate(self.terms_)}
        return self

    def transform(self, X):
        """stage 2: a dense float array (n_docs, n_terms) of term counts (1/0 if binary);
        terms that aren't in the vocabulary are ignored."""
        docs = list(X)
        out = np.zeros((len(docs), len(self.terms_)))
        for i, doc in enumerate(docs):
            for t, c in Counter(ngrams(tokenize(doc), self.ngram_range)).items():
                j = self.vocabulary_.get(t)
                if j is not None:
                    out[i, j] = 1.0 if self.binary else c
        return out


@app.cell(hide_code=True)
def _():
    checks.run(2, _tests="test_capstone.py", CountVectorizer=CountVectorizer, tokenize=tokenize)
    return


@app.class_definition
class TfidfTransformer(Transformer):
    """reweights a count matrix: rare terms count more, and every row gets unit length."""

    def __init__(self, *, smooth_idf=True, sublinear_tf=False, norm="l2"):
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf
        self.norm = norm

    def fit(self, X, y=None):
        """stage 3: idf_ (n_terms,) from the document frequencies df_j (rows with X_ij > 0) of
        n rows: log((1 + n) / (1 + df)) + 1 if smooth_idf else log(n / df) + 1. returns self."""
        X = np.asarray(X, dtype=np.float64)
        n, df = X.shape[0], np.count_nonzero(X > 0, axis=0)
        if self.smooth_idf:
            self.idf_ = np.log((1.0 + n) / (1.0 + df)) + 1.0
        else:
            with np.errstate(divide="ignore"):
                self.idf_ = np.log(n / df) + 1.0
        return self

    def transform(self, X):
        """stage 3: tf * idf_, with tf = X, or 1 + log(X) where X > 0 (0 elsewhere) if
        sublinear_tf; then, if norm == "l2", divide each row by its Euclidean norm (rows of
        zeros stay zeros). norm=None skips that. a new array."""
        X = np.asarray(X, dtype=np.float64)
        if self.sublinear_tf:
            tf = np.zeros_like(X)
            pos = X > 0
            tf[pos] = 1.0 + np.log(X[pos])
        else:
            tf = X.copy()
        out = tf * self.idf_
        if self.norm == "l2":
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            out = np.divide(out, norms, out=np.zeros_like(out), where=norms > 0)
        elif self.norm is not None:
            raise ValueError(f"norm must be 'l2' or None, got {self.norm!r}")
        return out


@app.cell(hide_code=True)
def _():
    checks.run(3, _tests="test_capstone.py", TfidfTransformer=TfidfTransformer)
    return


@app.function
def text_classifier(alpha=1e-3, *, min_df=2, max_features=None, ngram_range=(1, 1), tfidf=True):
    """stage 4: an UNFITTED pipeline from raw strings to labels, built with make_pipeline:
    CountVectorizer(min_df=..., max_features=..., ngram_range=...), then (only if tfidf)
    TfidfTransformer(), then LogisticRegression(alpha=alpha) (yours, from topic 04). so the
    steps are named "countvectorizer", "tfidftransformer" and "logisticregression"."""
    steps = [CountVectorizer(min_df=min_df, max_features=max_features, ngram_range=ngram_range)]
    if tfidf:
        steps.append(TfidfTransformer())
    steps.append(LogisticRegression(alpha=alpha))
    return make_pipeline(*steps)


@app.cell(hide_code=True)
def _():
    checks.run(4, _tests="test_capstone.py", text_classifier=text_classifier)
    return


@app.function
def metric_with_ci(y_true, y_pred, metric, *, n_boot=2000, alpha=0.05, rng=None):
    """stage 5: (value, lo, hi): metric(y_true, y_pred) and a (1 - alpha) percentile bootstrap
    CI from mlfs.stats.bootstrap_ci, resampling the (y_true, y_pred) PAIRS jointly. y_pred
    may be labels or scores (for roc_auc_score). three Python floats."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    lo, hi = bootstrap_ci((y_true, y_pred), lambda t, p: float(metric(t, p)), n_boot=n_boot, alpha=alpha, rng=rng)
    return float(metric(y_true, y_pred)), float(lo), float(hi)


@app.function
def paired_comparison(y_true, pred_a, pred_b, *, n_permutations=10000, rng=None):
    """stage 5: does model b classify more rows correctly than model a, on the SAME rows?
    returns (diff, p): diff = accuracy(b) - accuracy(a), and p the two-sided paired
    permutation p-value from mlfs.stats.permutation_test on the per-row correctness
    (1.0 / 0.0) of b and of a."""
    y_true = np.asarray(y_true)
    ca = (np.asarray(pred_a) == y_true).astype(float)
    cb = (np.asarray(pred_b) == y_true).astype(float)
    diff, p = permutation_test(cb, ca, n_permutations=n_permutations, paired=True, rng=rng)
    return float(diff), float(p)


@app.cell(hide_code=True)
def _():
    checks.run(5, _tests="test_capstone.py", metric_with_ci=metric_with_ci, paired_comparison=paired_comparison)
    return


@app.function
def top_features(model, k=10):
    """stage 6: what the fitted pipeline learned: (positive, negative), two lists of k
    (term, weight) pairs: the k terms with the LARGEST logistic coefficients (towards
    classes_[1], largest first) and the k with the SMALLEST (most negative first). terms come
    from the countvectorizer step's terms_, weights from the logisticregression step's coef_."""
    terms = model.named_steps["countvectorizer"].terms_
    coef = np.asarray(model.named_steps["logisticregression"].coef_)
    order = np.argsort(coef, kind="stable")
    positive = [(str(terms[j]), float(coef[j])) for j in order[::-1][:k]]
    negative = [(str(terms[j]), float(coef[j])) for j in order[:k]]
    return positive, negative


@app.function
def worst_errors(y_true, proba, k=5):
    """stage 6: the indices (an int array, at most k) of the misclassified rows on which the
    model was most confident. y_true is 0/1, proba = P(class 1) (n,); a row is misclassified
    when (proba > 0.5) != y_true. order: largest |proba - 0.5| first (ties: smaller index)."""
    y_true, proba = np.asarray(y_true), np.asarray(proba, dtype=np.float64)
    wrong = np.flatnonzero((proba > 0.5).astype(int) != y_true)
    conf = np.abs(proba[wrong] - 0.5)
    return wrong[np.argsort(-conf, kind="stable")][:k].astype(int)


@app.cell(hide_code=True)
def _():
    checks.run(6, _tests="test_capstone.py", top_features=top_features, worst_errors=worst_errors,
               text_classifier=text_classifier)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## the data, and the locked test set (given)

    5,574 SMS messages, about 13% spam (small config: a 600-message synthetic stand-in, with
    15% spam drawn from a separate word list). 20% of the messages are split off now,
    stratified by label, and not touched again until experiment C.
    """)
    return


@app.cell
def _():
    texts, labels = datasets.sms_spam()
    texts = np.asarray(texts, dtype=object)
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, stratify=labels, rng=0)
    CV = StratifiedKFold(5, shuffle=True, rng=1)
    mo.md(f"{len(texts):,} messages ({labels.mean():.1%} spam) · train {len(X_train):,} · test {len(X_test):,} "
          f"(locked){' · small config: synthetic stand-in' if SMALL else ''}\n\n"
          f"an example of each class: *{X_train[np.flatnonzero(y_train == 0)[0]]}* (ham) · "
          f"*{X_train[np.flatnonzero(y_train == 1)[0]]}* (spam)")
    return CV, X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment A (given): baselines, and choosing the penalty

    a majority-class baseline, and a grid search over the penalty $\alpha$ of the TF-IDF +
    logistic-regression pipeline, by 5-fold stratified CV on the training set, scored by F1
    on the spam class. (with a vocabulary of $V$ terms, many messages and few spam examples,
    the training data are often separable: without a penalty the weights would diverge, as
    in topic 04.)
    """)
    return


@app.cell
def _(CV, X_train, y_train):
    _dummy = cross_val_score(DummyClassifier(), X_train, y_train, cv=CV, scoring=F1)
    search = GridSearchCV(text_classifier(max_features=MAX_FEATURES), {"logisticregression__alpha": ALPHAS},
                          cv=CV, scoring=F1).fit(X_train, y_train)
    _rows = [f"| {p['logisticregression__alpha']:g} | {m:.3f} | {s:.3f} |" for p, m, s in
             zip(search.cv_results_["params"], search.cv_results_["mean_test_score"],
                 search.cv_results_["std_test_score"])]
    mo.md("| $\\alpha$ | CV F1 (mean) | std over folds |\n|---|---|---|\n" + "\n".join(_rows)
          + f"\n\nmajority-class baseline: CV F1 {np.mean(_dummy):.3f} (it never predicts spam). chosen: "
          f"$\\alpha$ = **{search.best_params_['logisticregression__alpha']:g}**, vocabulary of "
          f"{len(search.best_estimator_.named_steps['countvectorizer'].terms_):,} terms.")
    return (search,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment B (given): what matters? ablations by cross-validation

    starting from the chosen pipeline, change one thing at a time and compare cross-validated
    F1 on the training set: raw counts instead of TF-IDF, unigrams and bigrams instead of
    unigrams alone, and keeping terms seen in a single document (`min_df=1`). the test set
    stays locked.
    """)
    return


@app.cell
def _(CV, X_train, search, y_train):
    _a = search.best_params_["logisticregression__alpha"]
    _variants = {
        "chosen: TF-IDF, unigrams, min_df = 2": text_classifier(_a, max_features=MAX_FEATURES),
        "raw counts, no TF-IDF": text_classifier(_a, max_features=MAX_FEATURES, tfidf=False),
        "unigrams + bigrams": text_classifier(_a, max_features=MAX_FEATURES, ngram_range=(1, 2)),
        "min_df = 1": text_classifier(_a, max_features=MAX_FEATURES, min_df=1),
    }
    _rows = []
    for _name, _model in _variants.items():
        _s = cross_val_score(_model, X_train, y_train, cv=CV, scoring=F1)
        _rows.append(f"| {_name} | {np.mean(_s):.3f} | {np.std(_s):.3f} |")
    mo.md("| variant | CV F1 | std over folds |\n|---|---|---|\n" + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment C (given): the one look at the test set

    the chosen pipeline, refitted on all training messages, against the majority-class
    baseline on the locked test set, once. every number with a 95% bootstrap CI, and a paired
    permutation test of the accuracy difference.
    """)
    return


@app.cell
def _(X_test, X_train, search, y_test, y_train):
    final = search.best_estimator_
    test_proba = final.predict_proba(X_test)[:, 1]
    _pred = final.predict(X_test)
    _base = DummyClassifier().fit(X_train, y_train).predict(X_test)
    _rows = []
    for _name, _metric, _p in (("accuracy", accuracy_score, _pred), ("precision", precision_score, _pred),
                               ("recall", recall_score, _pred), ("F1", f1_score, _pred),
                               ("ROC AUC", roc_auc_score, test_proba)):
        _v, _lo, _hi = metric_with_ci(y_test, _p, _metric, n_boot=N_BOOT, rng=2)
        _b = _metric(y_test, _base) if _name != "ROC AUC" else 0.5
        _rows.append(f"| {_name} | **{_v:.3f}** | [{_lo:.3f}, {_hi:.3f}] | {_b:.3f} |")
    _diff, _pval = paired_comparison(y_test, _base, _pred, n_permutations=N_BOOT, rng=3)
    mo.md("| metric | test | 95% CI | majority baseline |\n|---|---|---|---|\n" + "\n".join(_rows)
          + f"\n\naccuracy gain over the baseline: {_diff:+.3f} (paired permutation p = {_pval:.4f}).")
    return final, test_proba


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment D (given): error analysis

    which terms push a message towards spam and towards ham, and the test messages the model
    got most confidently wrong.
    """)
    return


@app.cell
def _(X_test, final, test_proba, y_test):
    _pos, _neg = top_features(final, k=12)
    _table = "| towards spam | weight | towards ham | weight |\n|---|---|---|---|\n" + "\n".join(
        f"| `{a}` | {wa:+.2f} | `{b}` | {wb:+.2f} |" for (a, wa), (b, wb) in zip(_pos, _neg))
    _idx = worst_errors(y_test, test_proba, k=6)
    _errs = "\n".join(f"- ({'spam' if y_test[i] else 'ham'}, P(spam) = {test_proba[i]:.2f}) *{X_test[i]}*" for i in _idx)
    mo.md(_table + "\n\n**the most confident mistakes** (true label, predicted probability, text):\n\n"
          + (_errs or "- none"))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## write-up

    replace this cell with your report (grade it with [`rubric.md`](rubric.md)). answer at
    least:

    1. **the result.** how well does your classifier detect spam on the test set? give
       accuracy, precision, recall, F1 and ROC AUC with their CIs, next to the baseline. which
       of these numbers would you show a user deciding whether to trust the spam filter, and
       why is accuracy the least useful one here?
    2. **what mattered.** from experiments A and B: how much did the penalty matter? did
       TF-IDF, bigrams or `min_df` change CV F1 by more than the fold-to-fold noise? (if two
       variants differ by less than the std over folds, say so.)
    3. **what the model learned.** do the top features make sense? are any of them
       artifacts of how this dataset was collected rather than of spam in general (think
       about tokens like numbers, `£`, or `txt`)?
    4. **where it fails.** read the most confident mistakes. group them: mislabeled
       messages? spam that avoids the usual words? ham that looks like an advert? what would
       fix each group (features, data, a different threshold)?
    5. **limits.** your CIs come from bootstrapping test messages. what do they *not* account
       for (the split, the choice of $\alpha$, messages from a different year or country)?
    """)
    return


if __name__ == "__main__":
    app.run()
