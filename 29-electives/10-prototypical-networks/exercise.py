import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # prototypical networks · build

    spec-driven (L4): the **spec** boxes in the lesson define every function, the checks
    enforce them, and the internals are yours. a dataset is a tensor `X` (any shape after the
    first axis) with integer labels `y (n,)`; an **episode** is a dict of index tensors into it,
    so that the same episode can be replayed with any embedding. `embed` is any `nn.Module`
    mapping `(B, ...)` inputs to `(B, D)` embeddings.

    | stage | you build |
    |---|---|
    | 1 | `split_classes`, `sample_episode`: disjoint meta-train/meta-test classes and N-way K-shot episodes |
    | 2 | `prototypes`, `proto_logits`: class means, and negative squared distances (or scaled cosines) as logits |
    | 3 | `episode_loss`: the prototypical loss and accuracy of one episode |
    | 4 | `train_protonet`, `evaluate_protonet`: episodic training, and few-shot accuracy with a 95% interval |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def split_classes(classes, n_test, *, generator=None):
    """stage 1: shuffle the distinct class ids, return (train, test) as sorted tuples of ints, len(test) = n_test."""
    raise NotImplementedError("stage1: implement split_classes")


@app.function
def sample_episode(y, classes, n_way, k_shot, n_query, *, generator=None):
    """stage 1: dict of int64 tensors support (N*K,), query (N*Q,) indices into y, support_labels, query_labels in 0..N-1, classes (N,)."""
    raise NotImplementedError("stage1: implement sample_episode")


@app.cell(hide_code=True)
def _():
    checks.run(1, split_classes=split_classes, sample_episode=sample_episode)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def prototypes(z, labels, n_way):
    """stage 2: (N, D) mean embedding of each class 0..N-1; labels in any order, counts may differ."""
    raise NotImplementedError("stage2: implement prototypes")


@app.function
def proto_logits(zq, protos, *, metric="sqeuclidean", scale=1.0):
    """stage 2: (Q, N) logits: -scale * squared Euclidean distance, or scale * cosine similarity."""
    raise NotImplementedError("stage2: implement proto_logits")


@app.cell(hide_code=True)
def _():
    checks.run(2, prototypes=prototypes, proto_logits=proto_logits)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def episode_loss(embed, X, episode, *, metric="sqeuclidean", scale=1.0):
    """stage 3: (cross-entropy of the query logits as a scalar tensor, query accuracy as a float); one embed call on support then query."""
    raise NotImplementedError("stage3: implement episode_loss")


@app.cell(hide_code=True)
def _():
    checks.run(3, prototypes=prototypes, proto_logits=proto_logits, episode_loss=episode_loss)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def train_protonet(embed, X, y, classes, *, n_way, k_shot, n_query, n_episodes, lr=1e-3, metric="sqeuclidean",
                   scale=1.0, seed=0):
    """stage 4: Adam on episode_loss over n_episodes episodes drawn from `classes` (train mode); returns the list of losses."""
    raise NotImplementedError("stage4: implement train_protonet")


@app.function
def evaluate_protonet(embed, X, y, classes, *, n_way, k_shot, n_query=15, n_episodes=600, metric="sqeuclidean",
                      scale=1.0, seed=0):
    """stage 4: (mean query accuracy over episodes, 95% half-width 1.96 * sd / sqrt(n_episodes)), eval mode, no gradients."""
    raise NotImplementedError("stage4: implement evaluate_protonet")


@app.cell(hide_code=True)
def _():
    checks.run(4, sample_episode=sample_episode, prototypes=prototypes, proto_logits=proto_logits,
               episode_loss=episode_loss, train_protonet=train_protonet, evaluate_protonet=evaluate_protonet)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. it trains your prototypical
    networks on the faces of 28 people and tests them on 12 people they've never seen.
    """)
    return


if __name__ == "__main__":
    app.run()
