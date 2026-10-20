import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import torch
    from torch import nn

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # speech recognition with CTC · build

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design. label 0 is the blank unless a `blank=` argument says
    otherwise.

    | stage | you build |
    |---|---|
    | 1 | `ctc_collapse`, `ctc_expand`, `min_frames`, `ctc_alignments`: the alignment space |
    | 2 | `ctc_forward`, `ctc_nll`: the forward algorithm, in log space (NumPy) |
    | 3 | `ctc_loss`: a batched, differentiable CTC loss in PyTorch that matches `F.ctc_loss` |
    | 4 | `greedy_decode`, `prefix_beam_search`: from frame posteriors to label sequences |
    | 5 | `edit_distance`, `error_rate`, `CTCRecognizer`: scoring, and a model to train |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def ctc_collapse(path, *, blank=0):
    """stage 1: the label sequence a frame path stands for: merge runs of repeats, then drop blanks; a list of ints."""
    raise NotImplementedError("stage1: implement ctc_collapse")


@app.function
def ctc_expand(labels, *, blank=0):
    """stage 1: the extended sequence [blank, l1, blank, l2, ..., lU, blank] (length 2U + 1), a list of ints."""
    raise NotImplementedError("stage1: implement ctc_expand")


@app.function
def min_frames(labels):
    """stage 1: the fewest frames any path for `labels` needs: U plus the number of adjacent equal pairs."""
    raise NotImplementedError("stage1: implement min_frames")


@app.function
def ctc_alignments(labels, T, *, blank=0):
    """stage 1: every length-T frame path (tuple of ints) that collapses to `labels`, sorted."""
    raise NotImplementedError("stage1: implement ctc_alignments")


@app.cell(hide_code=True)
def _():
    checks.run(1, ctc_collapse=ctc_collapse, ctc_expand=ctc_expand, min_frames=min_frames,
               ctc_alignments=ctc_alignments)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def ctc_forward(log_probs, labels, *, blank=0):
    """stage 2: log alpha, shape (T, 2U + 1): log-prob of all path prefixes ending at time t in extended state s."""
    raise NotImplementedError("stage2: implement ctc_forward")


@app.function
def ctc_nll(log_probs, labels, *, blank=0):
    """stage 2: -log p(labels | x), summed over all alignments; inf when no alignment fits in T frames."""
    raise NotImplementedError("stage2: implement ctc_nll")


@app.cell(hide_code=True)
def _():
    checks.run(2, ctc_forward=ctc_forward, ctc_nll=ctc_nll, ctc_alignments=ctc_alignments)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def ctc_loss(log_probs, targets, input_lengths, target_lengths, *, blank=0, reduction="mean", zero_infinity=False):
    """stage 3: batched CTC loss, log_probs (B, T, V), padded targets (B, S); matches F.ctc_loss (which wants (T, B, V))."""
    raise NotImplementedError("stage3: implement ctc_loss")


@app.cell(hide_code=True)
def _():
    checks.run(3, ctc_loss=ctc_loss)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def greedy_decode(log_probs, *, blank=0):
    """stage 4: best-path decoding: the argmax label of every frame of (T, V) log_probs, collapsed; a list of ints."""
    raise NotImplementedError("stage4: implement greedy_decode")


@app.function
def prefix_beam_search(log_probs, *, beam_size=8, blank=0):
    """stage 4: CTC prefix beam search over (T, V) log_probs; list of (prefix tuple, log p(prefix)) best first."""
    raise NotImplementedError("stage4: implement prefix_beam_search")


@app.cell(hide_code=True)
def _():
    checks.run(4, greedy_decode=greedy_decode, prefix_beam_search=prefix_beam_search)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def edit_distance(ref, hyp):
    """stage 5: Levenshtein distance between two sequences (substitutions, insertions, deletions each cost 1)."""
    raise NotImplementedError("stage5: implement edit_distance")


@app.function
def error_rate(refs, hyps):
    """stage 5: corpus error rate, sum of edit distances / total reference length (WER/CER/PER)."""
    raise NotImplementedError("stage5: implement error_rate")


@app.class_definition
class CTCRecognizer(nn.Module):
    """acoustic model: features (B, T, n_features) -> log-probs (B, T, n_classes), blank included."""

    def __init__(self, n_features, n_classes, *, hidden_size=64, num_layers=2):
        """stage 5: a bidirectional nn.GRU (batch_first) over the features, then nn.Linear(2 * hidden_size, n_classes)."""
        raise NotImplementedError("stage5: implement CTCRecognizer.__init__")

    def forward(self, x):
        """stage 5: log_softmax over classes of the per-frame logits; the time axis is unchanged."""
        raise NotImplementedError("stage5: implement CTCRecognizer.forward")


@app.cell(hide_code=True)
def _():
    checks.run(5, edit_distance=edit_distance, error_rate=error_rate, CTCRecognizer=CTCRecognizer,
               ctc_loss=ctc_loss, greedy_decode=greedy_decode)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab (`marimo edit lab.py`). it builds a synthetic
    "phoneme" corpus from your topic 24 vowels and noise bursts, featurizes it with your log
    mel spectrogram, trains your `CTCRecognizer` with your `ctc_loss`, and asks what the
    blank is for.
    """)
    return


if __name__ == "__main__":
    app.run()
