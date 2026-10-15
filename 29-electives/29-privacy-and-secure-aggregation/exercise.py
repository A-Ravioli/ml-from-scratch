import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import numpy as np
    import torch

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # privacy & secure aggregation · build

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `encode`, `decode`: real vectors as integers mod $2^{32}$ |
    | 2 | `public_key`, `shared_secret`, `prg`, `mask`: pairwise masks that cancel in the sum |
    | 3 | `shamir_share`, `shamir_reconstruct`, `secure_aggregate`, `secagg_upload_bytes`: surviving dropouts |
    | 4 | `clip_update`, `dp_aggregate`: the Gaussian mechanism on a sum of client updates |
    | 5 | `rdp_sampled_gaussian`, `rdp_to_dp`, `compute_epsilon`: a privacy accountant |
    | 6 | `dp_fedavg`: user-level differentially private federated averaging, with ε and bytes per round |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def encode(x, *, scale=2**16, modulus=2**32):
    """stage 1: round(x * scale) mod modulus, as an int64 NumPy array with entries in [0, modulus)."""
    raise NotImplementedError("stage1: implement encode")


@app.function
def decode(z, *, scale=2**16, modulus=2**32):
    """stage 1: invert encode: entries >= modulus / 2 are negative; returns float64."""
    raise NotImplementedError("stage1: implement decode")


@app.cell(hide_code=True)
def _():
    checks.run(1, encode=encode, decode=decode)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def public_key(secret, *, prime=2**61 - 1, generator=3):
    """stage 2: generator ** secret mod prime (a toy Diffie-Hellman public key)."""
    raise NotImplementedError("stage2: implement public_key")


@app.function
def shared_secret(my_secret, their_public, *, prime=2**61 - 1):
    """stage 2: their_public ** my_secret mod prime; the same number from both ends."""
    raise NotImplementedError("stage2: implement shared_secret")


@app.function
def prg(seed, dim, *, modulus=2**32):
    """stage 2: dim pseudorandom integers, uniform in [0, modulus), from np.random.default_rng(seed)."""
    raise NotImplementedError("stage2: implement prg")


@app.function
def mask(i, z, my_secret, publics, *, modulus=2**32, prime=2**61 - 1):
    """stage 2: z + sum_{j > i} prg(s_ij) - sum_{j < i} prg(s_ij) mod modulus, over the other ids in publics."""
    raise NotImplementedError("stage2: implement mask")


@app.cell(hide_code=True)
def _():
    checks.run(2, public_key=public_key, shared_secret=shared_secret, prg=prg, mask=mask)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def shamir_share(secret, n, t, *, rng, prime=2**61 - 1):
    """stage 3: n shares (x, f(x)), x = 1..n, of a random degree-(t-1) polynomial f mod prime with f(0) = secret."""
    raise NotImplementedError("stage3: implement shamir_share")


@app.function
def shamir_reconstruct(shares, *, prime=2**61 - 1):
    """stage 3: f(0) by Lagrange interpolation mod prime from a list of (x, y) shares."""
    raise NotImplementedError("stage3: implement shamir_reconstruct")


@app.function
def secure_aggregate(updates, *, threshold, dropped=(), scale=2**16, modulus=2**32, prime=2**61 - 1, seed=None,
                     uploads=None, revealed=None):
    """stage 3: simulate one round of secure aggregation; returns the float64 sum of the surviving clients' updates."""
    raise NotImplementedError("stage3: implement secure_aggregate")


@app.function
def secagg_upload_bytes(n_clients, dim, *, n_dropped=0, entry_bytes=4, key_bytes=8):
    """stage 3: bytes one surviving client uploads in a secure-aggregation round (masked input, key, shares)."""
    raise NotImplementedError("stage3: implement secagg_upload_bytes")


@app.cell(hide_code=True)
def _():
    checks.run(3, shamir_share=shamir_share, shamir_reconstruct=shamir_reconstruct,
               secure_aggregate=secure_aggregate, secagg_upload_bytes=secagg_upload_bytes)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def clip_update(delta, clip):
    """stage 4: delta scaled by min(1, clip / ||delta||_2)."""
    raise NotImplementedError("stage4: implement clip_update")


@app.function
def dp_aggregate(deltas, *, clip, noise_multiplier, expected_clients, generator=None):
    """stage 4: (sum_k clip(delta_k) + N(0, (noise_multiplier * clip)^2 I)) / expected_clients."""
    raise NotImplementedError("stage4: implement dp_aggregate")


@app.cell(hide_code=True)
def _():
    checks.run(4, clip_update=clip_update, dp_aggregate=dp_aggregate)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def rdp_sampled_gaussian(q, sigma, orders):
    """stage 5: RDP epsilon(alpha) of one Poisson-subsampled Gaussian step (sensitivity 1), for integer orders >= 2."""
    raise NotImplementedError("stage5: implement rdp_sampled_gaussian")


@app.function
def rdp_to_dp(rdp, orders, delta):
    """stage 5: (epsilon, best order) = min over orders of rdp + log(1 / delta) / (order - 1)."""
    raise NotImplementedError("stage5: implement rdp_to_dp")


@app.function
def compute_epsilon(q, sigma, steps, delta, orders=tuple(range(2, 257))):
    """stage 5: epsilon after `steps` compositions of the subsampled Gaussian, at this delta."""
    raise NotImplementedError("stage5: implement compute_epsilon")


@app.cell(hide_code=True)
def _():
    checks.run(5, rdp_sampled_gaussian=rdp_sampled_gaussian, rdp_to_dp=rdp_to_dp, compute_epsilon=compute_epsilon)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 6")
    return


@app.function
def dp_fedavg(model, client_data, local_update, *, rounds, sampling_rate, clip, noise_multiplier, delta,
              server_lr=1.0, local_kwargs=None, eval_fn=None, eval_every=1, seed=None):
    """stage 6: DP-FedAvg (McMahan et al., 2018) with Poisson client sampling; trains model in place, returns history."""
    raise NotImplementedError("stage6: implement dp_fedavg")


@app.cell(hide_code=True)
def _():
    checks.run(6, dp_fedavg=dp_fedavg)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab (`marimo edit lab.py`): it checks your secure
    aggregation at scale, compares your accountant with naive composition, and ends with the
    block's project, the privacy–accuracy–communication triangle.
    """)
    return


if __name__ == "__main__":
    app.run()
