import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import torch
    from torch import nn

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # superposition & dictionary learning · build

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `sparse_feature_batch`: synthetic data made of sparse features → `mlfs.safety` |
    | 2 | `ToySuperposition`: the toy model of superposition → `mlfs.safety` |
    | 3 | `superposition_loss`, `train_toy_model`, `feature_dimensionality`: train it and measure how features share dimensions |
    | 4 | `SparseAutoencoder`: a sparse dictionary for a model's activations → `mlfs.safety` |
    | 5 | `train_sae`, `mmcs`, `sae_stats`: train the dictionary and judge what it found |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def sparse_feature_batch(n, n_features, sparsity, *, generator=None):
    """stage 1: (n, n_features) float32: uniform [0, 1) values, each independently zero with probability sparsity."""
    raise NotImplementedError("stage1: implement sparse_feature_batch")


@app.cell(hide_code=True)
def _():
    checks.run(1, sparse_feature_batch=sparse_feature_batch)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class ToySuperposition(nn.Module):
    """Elhage et al. (2022): n_features squeezed through n_hidden dimensions and read back with a ReLU."""

    def __init__(self, n_features, n_hidden, *, importance=None):
        """stage 2: W (n_hidden, n_features) Xavier-normal, b zeros (n_features,), buffer importance (default ones)."""
        raise NotImplementedError("stage2: implement ToySuperposition.__init__")

    def forward(self, x):
        """stage 2: x (B, n_features) -> ReLU(W^T W x + b), (B, n_features)."""
        raise NotImplementedError("stage2: implement ToySuperposition.forward")


@app.cell(hide_code=True)
def _():
    checks.run(2, ToySuperposition=ToySuperposition, sparse_feature_batch=sparse_feature_batch)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def superposition_loss(model, x):
    """stage 3: importance-weighted squared error: mean over the batch of sum_i importance_i (x_i - model(x)_i)^2."""
    raise NotImplementedError("stage3: implement superposition_loss")


@app.function
def train_toy_model(model, sparsity, *, steps=3000, batch_size=1024, lr=1e-3, generator=None):
    """stage 3: Adam on fresh sparse_feature_batch draws each step; returns the list of per-step losses (floats)."""
    raise NotImplementedError("stage3: implement train_toy_model")


@app.function
def feature_dimensionality(W):
    """stage 3: D_i = ||W_i||^2 / sum_j (W_i_hat . W_j)^2 for the columns W_i of W (n_hidden, n_features); 0 for zero columns."""
    raise NotImplementedError("stage3: implement feature_dimensionality")


@app.cell(hide_code=True)
def _():
    checks.run(3, superposition_loss=superposition_loss, train_toy_model=train_toy_model,
               feature_dimensionality=feature_dimensionality, ToySuperposition=ToySuperposition,
               sparse_feature_batch=sparse_feature_batch)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.class_definition
class SparseAutoencoder(nn.Module):
    """a one-layer sparse autoencoder: ReLU features over an overcomplete dictionary of unit-norm directions."""

    def __init__(self, d_in, d_hidden, *, l1_coeff=0.001):
        """stage 4: W_dec (d_in, d_hidden) random unit columns, W_enc = W_dec copy, b_enc (d_hidden,), b_dec (d_in,) zeros."""
        raise NotImplementedError("stage4: implement SparseAutoencoder.__init__")

    def dictionary(self):
        """stage 4: the decoder directions with unit-norm columns, (d_in, d_hidden)."""
        raise NotImplementedError("stage4: implement SparseAutoencoder.dictionary")

    def encode(self, x):
        """stage 4: f = ReLU((x - b_dec) W_enc + b_enc), (B, d_hidden)."""
        raise NotImplementedError("stage4: implement SparseAutoencoder.encode")

    def decode(self, f):
        """stage 4: x_hat = f dictionary()^T + b_dec, (B, d_in)."""
        raise NotImplementedError("stage4: implement SparseAutoencoder.decode")

    def forward(self, x):
        """stage 4: (x_hat, f, loss) with loss = mean_b ||x_hat - x||^2 + l1_coeff mean_b ||f||_1."""
        raise NotImplementedError("stage4: implement SparseAutoencoder.forward")


@app.cell(hide_code=True)
def _():
    checks.run(4, SparseAutoencoder=SparseAutoencoder)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def train_sae(sae, acts, *, steps=2000, batch_size=256, lr=1e-3, generator=None):
    """stage 5: Adam on minibatches of rows of acts (drawn with generator); returns the list of per-step losses."""
    raise NotImplementedError("stage5: implement train_sae")


@app.function
def mmcs(learned, true):
    """stage 5: mean over the columns of true (d, n) of the max cosine similarity with a column of learned (d, m)."""
    raise NotImplementedError("stage5: implement mmcs")


@app.function
def sae_stats(sae, acts):
    """stage 5: {'l0': mean active features per row, 'dead': fraction never active, 'fvu': ||x - x_hat||^2 / ||x - mean||^2}."""
    raise NotImplementedError("stage5: implement sae_stats")


@app.cell(hide_code=True)
def _():
    checks.run(5, train_sae=train_sae, mmcs=mmcs, sae_stats=sae_stats, SparseAutoencoder=SparseAutoencoder)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.safety import sparse_feature_batch, ToySuperposition, SparseAutoencoder` gives
    you the toy model and the dictionary learner (SAELens-style libraries are unlocked).
    topic 04 turns from reading models to attacking them.
    """)
    return


if __name__ == "__main__":
    app.run()
