import jax
import jax.numpy as jnp
import haiku as hk
import configlib

from models.MainModel import MainModel

class Model(hk.Module):
    def __init__(self,
                 config: configlib.Config,
                 encoder: MainModel,
                 decoder: MainModel):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def __call__(self, x, latents_rng):
        batch_size = x.shape[0]
        means, logvars = self.encoder(x)
        latent_vars = self.reparameterize(batch_size, latents_rng, means, logvars)
        recon_x = self.decoder(latent_vars)
        return recon_x, means, logvars
        
        
    def reparameterize(self, batch_size, latent_rng, means, logvars):
        std = jnp.exp(0.5 * logvars)
        latent_samples = means + jax.random.normal(latent_rng, means.shape) * std
        return latent_samples
        