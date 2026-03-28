import torch

from exercise import (
    VAEAnalyzer,
    VAETheory,
    VariationalAutoencoder,
    VariationalDecoder,
    VariationalEncoder,
    create_synthetic_vae_data,
)


def test_compute_elbo():
    recon = torch.tensor(2.0)
    kl = torch.tensor(0.5)
    elbo = VAETheory.compute_elbo(recon, kl)
    assert torch.isclose(elbo, torch.tensor(-2.5))


def test_encoder_decoder_shapes():
    encoder = VariationalEncoder(6, 8, 3)
    decoder = VariationalDecoder(3, 8, 6)
    x = torch.randn(4, 6)
    mu, logvar = encoder(x)
    recon = decoder(mu)
    assert mu.shape == (4, 3)
    assert logvar.shape == (4, 3)
    assert recon.shape == (4, 6)


def test_variational_autoencoder_forward():
    vae = VariationalAutoencoder(6, 8, 3)
    x = torch.randn(5, 6)
    recon, mu, logvar = vae(x)
    loss = vae.loss_function(recon, x, mu, logvar)
    assert recon.shape == x.shape
    assert loss["loss"].item() >= 0.0


def test_analysis_helpers():
    data = create_synthetic_vae_data(n_samples=12, data_type="gaussian_mixture")
    vae = VariationalAutoencoder(data.shape[1], 10, 2)
    stats = VAEAnalyzer.posterior_collapse_detection(vae, data)
    assert "collapsed_dims" in stats
