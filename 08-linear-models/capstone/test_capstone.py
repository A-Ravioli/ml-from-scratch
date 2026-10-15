"""Checks for capstone.py (a text classifier from raw text).

All text here is small, synthetic and deterministic. The library pieces (pipelines,
searches, LogisticRegression, the bootstrap) come from mlfs (the reference solutions under
pytest); these checks test your tokenizer, vectorizers, pipeline, statistics and error
analysis, with scikit-learn's CountVectorizer and TfidfTransformer as oracles and minimum
metrics on held-out synthetic messages.
"""
import numpy as np
import pytest

from mlfs.testing import bounded

HAM = ("ok see you at home later then call me when you get there sorry i will be late tonight "
       "thanks for the lunch what time is the meeting can you pick up some milk").split()
SPAM = "free win prize cash claim now urgent txt offer winner £ 1000 award reply stop".split()


def _messages(n, seed, spam_rate=0.3):
    """short synthetic messages: ham words only, or ham words with a few spam words mixed in."""
    rng = np.random.default_rng(seed)
    texts, labels = [], []
    for _ in range(n):
        spam = rng.random() < spam_rate
        words = list(rng.choice(HAM, size=int(rng.integers(3, 9))))
        if spam:
            words += list(rng.choice(SPAM, size=int(rng.integers(2, 4))))
            rng.shuffle(words)
        if rng.random() < 0.3:
            words = [w.upper() for w in words]
        texts.append(" ".join(words) + rng.choice(["", "!", "!!", "."]))
        labels.append(int(spam))
    return np.array(texts, dtype=object), np.array(labels)


DOCS = ["Free entry: WIN a £1000 prize!", "ok, see you at 7", "win win win", "see you later, ok?"]


# ---------------------------------------------------------------- stage 1: tokenize


def test_stage1_hand_examples(nb):
    cases = {
        "WIN £1000 now!!": ["win", "£", "1000", "now"],
        "Don't be late :-)": ["don", "t", "be", "late"],
        "call 0800-123, ok?": ["call", "0800", "123", "ok"],
        "   ": [],
        "$5 or €5": ["$", "5", "or", "€", "5"],
    }
    for text, want in cases.items():
        got = nb.tokenize(text)
        assert list(got) == want, (
            f"tokenize({text!r}) should be {want}, got {got}: lowercase, then runs of a-z0-9 and single £ $ € signs "
            "(regex [a-z0-9]+|[£$€]); everything else separates tokens")


# ---------------------------------------------------------------- stage 2: CountVectorizer


def _sk_counts(nb, **kw):
    from sklearn.feature_extraction.text import CountVectorizer

    return CountVectorizer(tokenizer=nb.tokenize, lowercase=False, token_pattern=None, **kw)


def test_stage2_worked_example(nb):
    v = nb.CountVectorizer().fit(DOCS)
    assert list(v.terms_) == sorted(v.terms_) and "£" in v.vocabulary_ and v.vocabulary_["win"] == list(v.terms_).index("win"), (
        "terms_ are sorted alphabetically, and vocabulary_ maps each term to its column")
    X = np.asarray(v.transform(DOCS))
    assert X.shape == (4, len(v.terms_)) and X.dtype == float
    assert X[2, v.vocabulary_["win"]] == 3 and X[0, v.vocabulary_["win"]] == 1, "'win win win' counts 3"
    Xb = np.asarray(nb.CountVectorizer(binary=True).fit(DOCS).transform(DOCS))
    assert Xb.max() == 1.0, "binary=True records presence (1/0), not counts"
    unseen = np.asarray(v.transform(["zebra crossing"]))
    assert unseen.shape == (1, len(v.terms_)) and unseen.sum() == 0, "terms not in the vocabulary are ignored"


@pytest.mark.parametrize("kw", [dict(), dict(min_df=2), dict(ngram_range=(1, 2)), dict(ngram_range=(1, 2), min_df=3)])
def test_stage2_match_sklearn(nb, kw):
    texts, _ = _messages(60, 1)
    ours = nb.CountVectorizer(**kw).fit(texts)
    ref = _sk_counts(nb, **kw).fit(texts)
    assert ours.vocabulary_ == ref.vocabulary_, (
        f"{kw}: the vocabulary should match sklearn's CountVectorizer with your tokenizer: terms are n-grams joined by "
        "single spaces, kept if they appear in at least min_df DOCUMENTS, numbered alphabetically")
    assert np.allclose(ours.transform(texts[:10]), ref.transform(texts[:10]).toarray()), f"{kw}: counts differ"


def test_stage2_max_features(nb):
    tie = nb.CountVectorizer(max_features=2).fit(["zeta beta", "alpha zeta beta alpha", "gamma"])
    assert list(tie.terms_) == ["alpha", "beta"], (
        f"alpha, beta and zeta each occur twice: ties go alphabetically, so max_features=2 keeps alpha and beta; "
        f"got {list(tie.terms_)}")
    docs = ["a a a a b b b c c d", "a b c e", "e e e e e"]  # totals: a 5, e 6, b 4, c 3, d 1
    v2 = nb.CountVectorizer(min_df=2).fit(["spam spam", "eggs", "eggs ham"])
    assert list(v2.terms_) == ["eggs"], (
        f"min_df counts DOCUMENTS: 'spam' occurs twice but in one message, so min_df=2 drops it; got {list(v2.terms_)}")
    v = nb.CountVectorizer(max_features=3).fit(docs)
    assert list(v.terms_) == ["a", "b", "e"], (
        f"max_features keeps the terms with the largest TOTAL count (e 6, a 5, b 4), then numbers them "
        f"alphabetically; got {list(v.terms_)}")
    assert nb.CountVectorizer().get_params() == {"min_df": 1, "max_features": None, "ngram_range": (1, 1),
                                                 "binary": False}


# ---------------------------------------------------------------- stage 3: TfidfTransformer


@pytest.mark.parametrize("kw", [dict(), dict(smooth_idf=False), dict(sublinear_tf=True), dict(norm=None)])
def test_stage3_match_sklearn(nb, kw):
    from sklearn.feature_extraction.text import TfidfTransformer

    texts, _ = _messages(50, 2)
    from sklearn.feature_extraction.text import CountVectorizer

    C = CountVectorizer(min_df=2).fit_transform(texts).toarray().astype(float)
    ours = nb.TfidfTransformer(**kw).fit(C)
    ref = TfidfTransformer(**kw).fit(C)
    assert np.allclose(ours.idf_, ref.idf_), (
        f"{kw}: idf_ should be log((1 + n) / (1 + df)) + 1 (smooth_idf) or log(n / df) + 1, with df the number of "
        "rows where the term occurs")
    new = C[:7].copy()
    new[0] = 0.0  # an empty message
    got = np.asarray(ours.transform(new))
    assert np.allclose(got, ref.transform(new).toarray()), f"{kw}: transform differs from sklearn"
    assert np.all(got[0] == 0), "a row of zeros stays zeros (don't divide by a zero norm)"
    assert np.array_equal(new[0], np.zeros(C.shape[1])), "transform must not modify its input"


# ---------------------------------------------------------------- stage 4: the pipeline


def test_stage4_pipeline_structure(nb):
    m = nb.text_classifier(0.01, min_df=1)
    names = [name for name, _ in m.steps]
    assert names == ["countvectorizer", "tfidftransformer", "logisticregression"], (
        f"build it with make_pipeline(CountVectorizer(...), TfidfTransformer(), LogisticRegression(alpha=...)); "
        f"got steps {names}")
    assert m.named_steps["logisticregression"].alpha == 0.01 and m.named_steps["countvectorizer"].min_df == 1
    assert [n for n, _ in nb.text_classifier(tfidf=False).steps] == ["countvectorizer", "logisticregression"]
    assert nb.text_classifier(ngram_range=(1, 2)).named_steps["countvectorizer"].ngram_range == (1, 2)
    assert not hasattr(m.named_steps["logisticregression"], "coef_"), "return an UNFITTED pipeline"


def test_stage4_classifies_held_out_messages(nb):
    from mlfs.metrics import f1_score

    X, y = _messages(300, 3)
    Xt, yt = _messages(300, 4)
    m = bounded(nb.text_classifier(1e-4).fit, X, y, what="the pipeline's fit", budget=20_000_000)
    f1 = f1_score(yt, m.predict(Xt))
    assert f1 > 0.9, (
        f"on held-out synthetic messages the TF-IDF + logistic pipeline should reach F1 > 0.9 on spam, got {f1:.3f}. "
        "check that every step is fitted on the training texts and only transforms the test texts")
    P = np.asarray(m.predict_proba(Xt))
    assert P.shape == (300, 2) and np.allclose(P.sum(1), 1)


def test_stage4_tunable_by_grid_search(nb):
    from mlfs.metrics import f1_score, make_scorer
    from mlfs.model_selection import GridSearchCV, StratifiedKFold

    X, y = _messages(150, 5)
    search = bounded(GridSearchCV(nb.text_classifier(), {"logisticregression__alpha": [1e-4, 1e-2]},
                                  cv=StratifiedKFold(3, shuffle=True, rng=0), scoring=make_scorer(f1_score)).fit,
                     X, y, what="GridSearchCV over your pipeline", budget=50_000_000)
    assert search.best_params_["logisticregression__alpha"] in (1e-4, 1e-2) and search.best_score_ > 0.7, (
        "the pipeline must survive clone and set_params('logisticregression__alpha') inside a grid search")


# ---------------------------------------------------------------- stage 5: uncertainty


def test_stage5_metric_with_ci(nb):
    from mlfs.metrics import accuracy_score, f1_score

    rng = np.random.default_rng(6)
    y = rng.integers(0, 2, 400)
    pred = np.where(rng.random(400) < 0.85, y, 1 - y)
    v, lo, hi = nb.metric_with_ci(y, pred, accuracy_score, n_boot=1000, rng=0)
    assert isinstance(v, float) and np.isclose(v, accuracy_score(y, pred)) and lo < v < hi, (
        "value is the metric on all rows, and the CI brackets it")
    assert 0.03 < hi - lo < 0.1, f"with 400 rows and accuracy ~0.85, a 95% CI is about 0.07 wide; got {hi - lo:.3f}"
    covered = 0
    for s in range(60):
        r = np.random.default_rng(100 + s)
        yy = r.integers(0, 2, 300)
        pp = np.where(r.random(300) < 0.8, yy, 1 - yy)
        _, a, b = nb.metric_with_ci(yy, pp, accuracy_score, n_boot=400, rng=s)
        covered += a <= 0.8 <= b
    assert covered >= 50, f"95% intervals should contain the true accuracy 0.8 in about 57 of 60 datasets, got {covered}"
    v2, lo2, hi2 = nb.metric_with_ci(y, pred, f1_score, n_boot=300, rng=1)
    assert np.isclose(v2, f1_score(y, pred)) and lo2 < v2 < hi2, "any metric(y_true, y_pred) works, e.g. F1"


def test_stage5_paired_comparison(nb):
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, 500)
    a = np.where(rng.random(500) < 0.75, y, 1 - y)
    b = np.where(rng.random(500) < 0.9, y, 1 - y)
    diff, p = nb.paired_comparison(y, a, b, n_permutations=2000, rng=0)
    assert np.isclose(diff, np.mean(b == y) - np.mean(a == y)) and p < 0.01, (
        f"b is clearly better: diff = accuracy(b) - accuracy(a) > 0 with a small p; got {diff:.3f}, p = {p:.4f}")
    # b fixes 15 of a's mistakes and changes nothing else: a small, perfectly consistent gain. pairing sees it
    # clearly; an unpaired test drowns it in the variation between rows
    wrong = np.flatnonzero(a != y)
    b2 = a.copy()
    b2[wrong[:15]] = y[wrong[:15]]
    d2, p2 = nb.paired_comparison(y, a, b2, n_permutations=4000, rng=1)
    assert np.isclose(d2, 15 / 500) and p2 < 0.001, (
        f"b corrects 15 of a's errors and breaks none: the paired test (sign flips of per-row differences) gives "
        f"p ~ 1/n_permutations, got p = {p2:.4f}. pass paired=True to permutation_test")
    diff0, p0 = nb.paired_comparison(y, a, a.copy(), n_permutations=500, rng=0)
    assert diff0 == 0.0 and p0 > 0.5, f"identical predictions: diff 0 and a large p, got {diff0}, {p0}"


# ---------------------------------------------------------------- stage 6: error analysis


def test_stage6_top_features(nb):
    X, y = _messages(300, 8)
    m = nb.text_classifier(1e-2, min_df=1).fit(X, y)
    pos, neg = nb.top_features(m, k=5)
    assert len(pos) == 5 and len(neg) == 5 and all(isinstance(t, str) for t, _ in pos), (
        "two lists of k (term, weight) pairs")
    assert [w for _, w in pos] == sorted([w for _, w in pos], reverse=True), "positive terms come largest weight first"
    assert [w for _, w in neg] == sorted([w for _, w in neg]), "negative terms come most negative first"
    spam_terms = {t for t in SPAM} | {"1000"}
    assert sum(t in spam_terms for t, _ in pos) >= 4, (
        f"the terms pushing hardest towards spam should be spam words, got {[t for t, _ in pos]}")
    assert all(t not in spam_terms for t, _ in neg), f"the terms pushing towards ham shouldn't be spam words: {neg}"


def test_stage6_worst_errors(nb):
    y = np.array([1, 0, 1, 0, 1, 0])
    proba = np.array([0.9, 0.95, 0.2, 0.4, 0.46, 0.6])
    got = np.asarray(nb.worst_errors(y, proba, k=3))
    assert got.tolist() == [1, 2, 5], (
        "the misclassified rows are 1 (P = 0.95, label 0), 2 (P = 0.2, label 1), 4 (P = 0.46, label 1) and 5 "
        "(P = 0.6, label 0); by confidence |P - 0.5| that's 1 (0.45), 2 (0.30), 5 (0.10), 4 (0.04), so k = 3 gives "
        f"[1, 2, 5]; got {got.tolist()}")
    assert len(nb.worst_errors(y, np.where(y == 1, 0.9, 0.1), k=3)) == 0, "no mistakes: an empty array"
