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
    # speech recognition with CTC · build (reference solution)

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
    out, prev = [], None
    for p in (int(v) for v in path):
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return out


@app.function
def ctc_expand(labels, *, blank=0):
    """stage 1: the extended sequence [blank, l1, blank, l2, ..., lU, blank] (length 2U + 1), a list of ints."""
    out = [blank]
    for v in labels:
        out += [int(v), blank]
    return out


@app.function
def min_frames(labels):
    """stage 1: the fewest frames any path for `labels` needs: U plus the number of adjacent equal pairs."""
    labels = [int(v) for v in labels]
    return len(labels) + sum(a == b for a, b in zip(labels, labels[1:]))


@app.function
def ctc_alignments(labels, T, *, blank=0):
    """stage 1: every length-T frame path (tuple of ints) that collapses to `labels`, sorted."""
    ext = ctc_expand(labels, blank=blank)
    S = len(ext)
    out = []

    def walk(t, s, path):
        if t == T:
            if s >= S - 2:
                out.append(tuple(path))
            return
        for nxt in (s, s + 1, s + 2):
            if nxt >= S or (nxt == s + 2 and (ext[nxt] == blank or ext[nxt] == ext[s])):
                continue
            path.append(ext[nxt])
            walk(t + 1, nxt, path)
            path.pop()

    if T > 0:
        for s0 in (0, 1)[:S]:
            walk(1, s0, [ext[s0]])
    return sorted(set(out))


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
    lp = np.asarray(log_probs, dtype=float)
    ext = ctc_expand(labels, blank=blank)
    T, S = lp.shape[0], len(ext)
    alpha = np.full((T, S), -np.inf)
    alpha[0, 0] = lp[0, ext[0]]
    if S > 1:
        alpha[0, 1] = lp[0, ext[1]]
    skip = np.array([s >= 2 and ext[s] != blank and ext[s] != ext[s - 2] for s in range(S)])
    for t in range(1, T):
        prev = alpha[t - 1]
        a = prev.copy()
        a[1:] = np.logaddexp(a[1:], prev[:-1])
        a[2:] = np.where(skip[2:], np.logaddexp(a[2:], prev[:-2]), a[2:])
        alpha[t] = a + lp[t, ext]
    return alpha


@app.function
def ctc_nll(log_probs, labels, *, blank=0):
    """stage 2: -log p(labels | x), summed over all alignments; inf when no alignment fits in T frames."""
    alpha = ctc_forward(log_probs, labels, blank=blank)
    last = alpha[-1, -1] if len(labels) == 0 else np.logaddexp(alpha[-1, -1], alpha[-1, -2])
    return float(-last)


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
    B, T, V = log_probs.shape
    input_lengths = torch.as_tensor(input_lengths, dtype=torch.long, device=log_probs.device)
    target_lengths = torch.as_tensor(target_lengths, dtype=torch.long, device=log_probs.device)
    targets = torch.as_tensor(targets, dtype=torch.long, device=log_probs.device).reshape(B, -1)
    U = targets.shape[1]
    S = 2 * U + 1
    neg = torch.tensor(-1e30, dtype=log_probs.dtype, device=log_probs.device)
    ext = torch.full((B, S), blank, dtype=torch.long, device=log_probs.device)
    if U > 0:
        ext[:, 1::2] = targets.clamp(0, V - 1)
    skip = torch.zeros(B, S, dtype=torch.bool, device=log_probs.device)
    if U > 1:
        skip[:, 3::2] = targets[:, 1:] != targets[:, :-1]
    emit = torch.gather(log_probs, 2, ext.unsqueeze(1).expand(B, T, S))  # (B, T, S)
    alpha = torch.full((B, S), -1e30, dtype=log_probs.dtype, device=log_probs.device)
    alpha = torch.cat([emit[:, 0, :2], alpha[:, 2:]], dim=1) if S > 1 else emit[:, 0, :1]
    for t in range(1, T):
        prev1 = torch.cat([neg.expand(B, 1), alpha[:, :-1]], dim=1)
        prev2 = torch.cat([neg.expand(B, 2), alpha[:, :-2]], dim=1)
        prev2 = torch.where(skip, prev2, neg)
        new = torch.logsumexp(torch.stack([alpha, prev1, prev2]), dim=0) + emit[:, t]
        alpha = torch.where((t < input_lengths).unsqueeze(1), new, alpha)
    last = (2 * target_lengths).unsqueeze(1)
    end1 = alpha.gather(1, last).squeeze(1)
    end2 = alpha.gather(1, (last - 1).clamp(min=0)).squeeze(1)
    end2 = torch.where(target_lengths > 0, end2, neg)
    loss = -torch.logaddexp(end1, end2)
    loss = torch.where(input_lengths == 0, torch.zeros_like(loss), loss)  # no frames, no labels: p = 1
    needed = torch.tensor([min_frames(targets[b, :target_lengths[b]].tolist()) for b in range(B)],
                          device=log_probs.device)
    impossible = input_lengths < needed
    inf = torch.tensor(float("inf"), dtype=loss.dtype, device=loss.device)
    loss = torch.where(impossible, torch.zeros_like(loss) if zero_infinity else inf, loss)
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return (loss / target_lengths.clamp(min=1).to(loss.dtype)).mean()
    raise ValueError(f"unknown reduction {reduction!r}")


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
    lp = log_probs.detach().cpu().numpy() if torch.is_tensor(log_probs) else np.asarray(log_probs)
    return ctc_collapse(lp.argmax(axis=1), blank=blank)


@app.function
def prefix_beam_search(log_probs, *, beam_size=8, blank=0):
    """stage 4: CTC prefix beam search over (T, V) log_probs; list of (prefix tuple, log p(prefix)) best first."""
    lp = log_probs.detach().cpu().numpy() if torch.is_tensor(log_probs) else np.asarray(log_probs, dtype=float)
    T, V = lp.shape
    beams = {(): (0.0, -np.inf)}  # prefix -> (log p ending in blank, log p ending in a label)
    for t in range(T):
        nxt = {}

        def add(prefix, pb=-np.inf, pnb=-np.inf):
            ob, onb = nxt.get(prefix, (-np.inf, -np.inf))
            nxt[prefix] = (np.logaddexp(ob, pb), np.logaddexp(onb, pnb))

        for prefix, (pb, pnb) in beams.items():
            total = np.logaddexp(pb, pnb)
            add(prefix, pb=total + lp[t, blank])
            for c in range(V):
                if c == blank:
                    continue
                if prefix and prefix[-1] == c:
                    add(prefix + (c,), pnb=pb + lp[t, c])
                    add(prefix, pnb=pnb + lp[t, c])
                else:
                    add(prefix + (c,), pnb=total + lp[t, c])
        ranked = sorted(nxt.items(), key=lambda kv: -np.logaddexp(*kv[1]))
        beams = dict(ranked[:beam_size])
    out = [(prefix, float(np.logaddexp(pb, pnb))) for prefix, (pb, pnb) in beams.items()]
    return sorted([kv for kv in out if np.isfinite(kv[1])], key=lambda kv: -kv[1])


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
    ref, hyp = list(ref), list(hyp)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


@app.function
def error_rate(refs, hyps):
    """stage 5: corpus error rate, sum of edit distances / total reference length (WER/CER/PER)."""
    refs, hyps = list(refs), list(hyps)
    if len(refs) != len(hyps):
        raise ValueError(f"{len(refs)} references but {len(hyps)} hypotheses")
    total = sum(len(r) for r in refs)
    if total == 0:
        raise ValueError("references are all empty")
    return sum(edit_distance(r, h) for r, h in zip(refs, hyps)) / total


@app.class_definition
class CTCRecognizer(nn.Module):
    """acoustic model: features (B, T, n_features) -> log-probs (B, T, n_classes), blank included."""

    def __init__(self, n_features, n_classes, *, hidden_size=64, num_layers=2):
        """stage 5: a bidirectional nn.GRU (batch_first) over the features, then nn.Linear(2 * hidden_size, n_classes)."""
        super().__init__()
        self.rnn = nn.GRU(n_features, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * hidden_size, n_classes)

    def forward(self, x):
        """stage 5: log_softmax over classes of the per-frame logits; the time axis is unchanged."""
        h, _ = self.rnn(x)
        return torch.log_softmax(self.out(h), dim=-1)


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
