import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import re

    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F

    from mlfs import checks, lesson
    from mlfs.utils import as_rng  # yours, from ch 00 · 05


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # evaluation · build

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `loglikelihood`: how likely a model finds a continuation of a context → `mlfs.llm` |
    | 2 | `evaluate_multiple_choice`: a benchmark scored by likelihood → `mlfs.llm` |
    | 3 | `accuracy_ci`, `paired_accuracy_diff`: how sure you can be about a score, and a difference |
    | 4 | `ngrams`, `contamination_flags`: did the test set leak into training? |
    | 5 | `pairwise_judge`: an LLM judge, with its position bias cancelled |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
@torch.no_grad()
def loglikelihood(model, tokenizer, context, continuation, *, device="cpu"):
    """stage 1: (sum log p(continuation tokens | context), is_greedy) with context and continuation encoded separately."""
    raise NotImplementedError("stage1: implement loglikelihood")


@app.cell(hide_code=True)
def _():
    checks.run(1, loglikelihood=loglikelihood)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def evaluate_multiple_choice(model, tokenizer, items, *, normalize=True, device="cpu"):
    """stage 2: accuracy of argmax over choices of loglikelihood(question, choice) (divided by its token count if normalize)."""
    raise NotImplementedError("stage2: implement evaluate_multiple_choice")


@app.cell(hide_code=True)
def _():
    checks.run(2, evaluate_multiple_choice=evaluate_multiple_choice, loglikelihood=loglikelihood)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def accuracy_ci(correct, *, n_boot=2000, alpha=0.05, rng=None):
    """stage 3: (accuracy, lo, hi): mean of the 0/1 array and a percentile bootstrap interval over items."""
    raise NotImplementedError("stage3: implement accuracy_ci")


@app.function
def paired_accuracy_diff(correct_a, correct_b, *, n_boot=2000, alpha=0.05, rng=None):
    """stage 3: (acc_a - acc_b, lo, hi) with a PAIRED bootstrap: the same resampled items for both models."""
    raise NotImplementedError("stage3: implement paired_accuracy_diff")


@app.cell(hide_code=True)
def _():
    checks.run(3, accuracy_ci=accuracy_ci, paired_accuracy_diff=paired_accuracy_diff)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def ngrams(text, n=13):
    """stage 4: set of word n-gram tuples of text, lowercased, punctuation stripped (words = runs of letters, digits, ')."""
    raise NotImplementedError("stage4: implement ngrams")


@app.function
def contamination_flags(test_texts, train_texts, *, n=13):
    """stage 4: bool ndarray (len(test_texts),): True where a test text shares at least one word n-gram with the training texts."""
    raise NotImplementedError("stage4: implement contamination_flags")


@app.cell(hide_code=True)
def _():
    checks.run(4, ngrams=ngrams, contamination_flags=contamination_flags)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def pairwise_judge(judge, prompt, response_a, response_b):
    """stage 5: P(a better than b), averaged over both orders: (judge(prompt, a, b) + 1 - judge(prompt, b, a)) / 2."""
    raise NotImplementedError("stage5: implement pairwise_judge")


@app.cell(hide_code=True)
def _():
    checks.run(5, pairwise_judge=pairwise_judge)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.llm import loglikelihood, evaluate_multiple_choice` scores any causal LM on any
    multiple-choice benchmark; the capstone's model card reports its numbers with your
    `accuracy_ci`.
    """)
    return


if __name__ == "__main__":
    app.run()
