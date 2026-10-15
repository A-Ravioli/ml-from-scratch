# 08 · capstone: a text classifier from raw text

## goal

build a spam filter for SMS messages, starting from the raw strings. there's no feature table here: you decide what the numbers are. a tokenizer splits messages into tokens, a bag-of-words vectorizer turns each message into a vector of counts over a learned vocabulary, TF-IDF reweights the counts so that rare, informative words weigh more, and your own `LogisticRegression` from topic 04 does the classifying. then you do what separates a model from a result: a baseline, the penalty chosen by cross-validation, ablations that say which choices mattered, one honest look at the test set with confidence intervals, and an error analysis that reads the model's weights and its worst mistakes.

this pulls the chapter together. logistic regression with an $\ell_2$ penalty (topics 02 and 04) is the classifier, and the separability warning of topic 04 is why the penalty is not optional: with a vocabulary of a thousand words and a few hundred spam messages, the training data are almost always separable. the pipeline, grid search, metrics and stratified CV come from chapter 07, and the bootstrap and permutation test from chapter 05.

## data

- **SMS Spam Collection** via `mlfs.datasets.sms_spam()`: 5,574 English text messages, 747 of them (13.4%) spam, collected around 2011 from UK and Singaporean sources (Almeida et al., 2011). downloaded once and cached under `~/.cache/mlfs/`. returns `(texts, labels)` with `labels` an int array, 1 = spam.
- with `MLFS_OFFLINE=1` (tests, `verify.py`): a 600-message synthetic stand-in built from a list of common English words, with spam messages getting 2–4 extra words from a short spam list. it's far easier than the real data (the spam words never appear in ham), so its numbers say nothing about real spam filtering.
- **tests** use only small synthetic messages generated in `test_capstone.py`.

## configurations

| config | where | sizes | runtime (laptop CPU) |
|---|---|---|---|
| **small** (tests, `verify.py`, offline) | `MLFS_OFFLINE=1` | 600 synthetic messages (480 train / 120 test); the full vocabulary (about 70 terms); $\alpha \in \{10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}\}$; 500 bootstrap and permutation draws | a few seconds |
| **full** (your write-up) | default | 5,574 messages (4,459 train / 1,115 test); vocabulary capped at the 1,000 most frequent terms with `min_df=2`; $\alpha \in \{10^{-7}, 10^{-6}, \dots, 10^{-3}\}$; 2,000 draws | about a minute (50–65 s on a laptop), most of it IRLS on a 1,001 × 1,001 Hessian inside the grid search |

the notebook picks the config from `datasets.is_offline()`. the vocabulary cap is there for speed: each IRLS step builds a $(V + 1) \times (V + 1)$ Hessian, so the cost grows like $nV^2$. lifting the cap is a stretch goal.

## deliverables

1. `capstone.py` with the required functions implemented and their checks green:
   - `tokenize(text)`: lowercase, then runs of `a-z0-9` and single `£ $ €` signs
   - `CountVectorizer(*, min_df=1, max_features=None, ngram_range=(1, 1), binary=False)`: a `Transformer` with `terms_` (alphabetical) and `vocabulary_`, and a dense count matrix from `transform`
   - `TfidfTransformer(*, smooth_idf=True, sublinear_tf=False, norm="l2")`: `idf_` from document frequencies, TF-IDF weights, unit-length rows
   - `text_classifier(alpha=1e-3, *, min_df=2, max_features=None, ngram_range=(1, 1), tfidf=True)`: the unfitted pipeline `make_pipeline(CountVectorizer(...), TfidfTransformer(), LogisticRegression(alpha=alpha))`
   - `metric_with_ci(y_true, y_pred, metric, *, n_boot, alpha, rng)` and `paired_comparison(y_true, pred_a, pred_b, *, n_permutations, rng)`: a metric with a paired bootstrap CI, and a paired permutation test of an accuracy difference
   - `top_features(model, k=10)` and `worst_errors(y_true, proba, k=5)`: the most spam- and ham-indicative terms, and the most confident mistakes
2. the write-up section at the end of the notebook, answering its five prompts (the result, what mattered, what the model learned, where it fails, limits). experiments A–D are given and run on your functions.

## requirements

| check | threshold |
|---|---|
| tests in `test_capstone.py` pass | all (`pytest test_capstone.py`) |
| `CountVectorizer`, `TfidfTransformer` | match scikit-learn's (with your tokenizer) on vocabulary, counts, `idf_` and weights, for every option tested |
| the pipeline | F1 > 0.9 on held-out synthetic messages, and it survives `GridSearchCV` over `logisticregression__alpha` |
| `metric_with_ci` | 95% intervals contain the true accuracy in at least 50 of 60 simulated datasets |
| `paired_comparison` | a clear improvement gives $p < 0.01$; identical predictions give a large $p$ |
| full-config report | every number in your write-up is reproducible from the notebook, and reported with its CI and the baseline |

**small config.** the solution chooses $\alpha = 10^{-5}$, the smallest value on the small grid (CV F1 0.959, against 0.000 for the majority baseline, which never predicts spam), and gets every test metric exactly right: 1.000 [1.000, 1.000]. that only shows the code runs: the synthetic spam words never appear in ham.

**full config (the reference run, so you can check your code, not copy a write-up).** CV F1 by $\alpha$: $10^{-7}$ 0.934 ± 0.014, $10^{-6}$ **0.951** ± 0.017, $10^{-5}$ 0.949 ± 0.013, $10^{-4}$ 0.935 ± 0.019, $10^{-3}$ 0.745 ± 0.025 (and $10^{-2}$, outside the grid, predicts no spam at all: F1 0). the optimum is a flat plateau around $10^{-6}$–$10^{-5}$, inside the grid. ablations (CV F1): raw counts 0.931, unigrams + bigrams 0.945, `min_df=1` 0.951, against 0.951 for the chosen pipeline: only TF-IDF versus raw counts is clearly more than the fold-to-fold spread. on the 1,115 test messages (149 spam):

| metric | test | 95% bootstrap CI | majority baseline |
|---|---|---|---|
| accuracy | 0.985 | [0.977, 0.991] | 0.866 |
| precision | 0.971 | [0.939, 0.993] | 0 |
| recall | 0.913 | [0.863, 0.955] | 0 |
| F1 | 0.941 | [0.910, 0.966] | 0 |
| ROC AUC | 0.989 | [0.976, 0.998] | 0.5 |

paired permutation test of the accuracy gain over the baseline: $p = 0.0005$ (the smallest value 2,000 permutations can give). among the terms pushing hardest towards spam are `£`, `txt`, `ringtone`, `150p`, `text`, `uk` and `won` (alongside a few surprises such as `everyone`, `sexy`, `chat` and `ac`: look at your own ranked list); among those pushing towards ham are `me`, `lt`, `gt`, `i`, `my`, `can`, `wife` and `never` (`lt` and `gt` are HTML-escaped `<` and `>`, an artifact of how the ham was collected). your numbers will match these if your functions do; the write-up is about explaining them. for context, published linear models on this dataset reach roughly 97–98% accuracy with spam recall of 80–95% (Almeida et al., 2011).

three things to watch in the full config.

- **the edge of the grid.** if the chosen $\alpha$ sits at the end of the grid (as it does in the small config), extend the grid in that direction before believing it.
- **few spam messages.** with only 149 spam in the test set, the intervals for precision and recall are several times wider than the one for accuracy: they're computed on the spam messages alone (or on the messages flagged as spam).
- **zero-width intervals.** if your classifier makes no false positives at all (say 135 flagged, 135 spam), every bootstrap resample also has precision exactly 1, and the percentile interval is [1.000, 1.000]. that doesn't mean the precision is known exactly: 135 out of 135 is also quite likely if the true precision is 98%. for a proportion near 0 or 1, use an interval built for proportions: the Wilson interval, or the exact Clopper–Pearson interval, whose lower bound for $k = n$ successes is $(\alpha/2)^{1/n}$, here $0.025^{1/135} \approx 0.973$ (Wilson gives $n/(n + z^2) \approx 0.972$). the bootstrap is a fine default, but it can't invent variation that the sample doesn't contain.

## stretch goals

- **lift the vocabulary cap.** the IRLS Hessian is the bottleneck. switch the logistic step to `solver="gd"` (or write an L-BFGS solver with your chapter 06 `lbfgs`) and fit on the full vocabulary. does F1 improve, or does the extra vocabulary mostly add noise?
- **a decision threshold.** a spam filter that deletes a real message is much worse than one that lets a spam through. choose a threshold on `predict_proba` by cross-validation so that at most 1% of ham is flagged, and report the recall you get at that operating point, with CIs. (chapter 07's precision–recall curves help.)
- **better tokens.** replace every number with a `<num>` token, every URL with `<url>`, or add character n-grams. which change helps most, measured by paired comparisons on CV predictions?
- **compare with naive Bayes.** a multinomial naive Bayes classifier (counts, Laplace smoothing) is 15 lines. is it better or worse than logistic regression here, and is the difference significant? (chapter 12 explains why naive Bayes often wins with very little data.)
- **calibration.** draw the reliability diagram (chapter 07) of your classifier on CV predictions. does the penalty make it under- or over-confident?

## self-assessment

grade your write-up with [`rubric.md`](rubric.md).

## references

- Almeida, Gómez Hidalgo & Yamakami (2011), "Contributions to the study of SMS spam filtering: new collection and results", *Proceedings of the 11th ACM Symposium on Document Engineering*: 259–262. the dataset, with baseline results. [doi:10.1145/2034691.2034742](https://doi.org/10.1145/2034691.2034742)
- Manning, Raghavan & Schütze (2008), *Introduction to Information Retrieval*, Cambridge University Press, chapters 2 and 6: tokenization, term weighting and TF-IDF. free online. [nlp.stanford.edu/IR-book](https://nlp.stanford.edu/IR-book/)
- Jurafsky & Martin (2024, draft), *Speech and Language Processing*, 3rd ed., chapters 2, 4 and 5: tokenization, naive Bayes, logistic regression for text, and evaluation. [web.stanford.edu/~jurafsky/slp3](https://web.stanford.edu/~jurafsky/slp3/)
- Sparck Jones (1972), "A statistical interpretation of term specificity and its application in retrieval", *Journal of Documentation* 28(1): 11–21. where inverse document frequency comes from. [doi:10.1108/eb026526](https://doi.org/10.1108/eb026526)
- Brown, Cai & DasGupta (2001), "Interval estimation for a binomial proportion", *Statistical Science* 16(2): 101–133. why Wilson and Clopper–Pearson beat the textbook interval near 0 and 1. [doi:10.1214/ss/1009213286](https://doi.org/10.1214/ss/1009213286)
- Wang & Manning (2012), "Baselines and bigrams: simple, good sentiment and topic classification", *ACL 2012*. how strong linear models on bags of words and bigrams are. [aclanthology.org/P12-2018](https://aclanthology.org/P12-2018/)
