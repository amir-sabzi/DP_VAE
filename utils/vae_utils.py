import jax
import jax.numpy as jnp
import configlib
from models.VAEsimple import Model as VAEsimple
from models.encoders import EncoderMLP
from models.decoders import DecoderMLP

def kl_divergence_normal(mean, logvar):
    return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))

def binary_cross_entropy(recon_x, x):
    ## Mapping the real values back to probability
    logits_x = jax.nn.log_sigmoid(recon_x)
    return -jnp.sum(x * logits_x + (1. - x) * jnp.log(-jnp.expm1(x)))

def get_vae_model(config: configlib.Config, model_vae):
    def VAE(x, latent_rng):
        encoder = get_encoder(config)
        decoder = get_decoder(config)
        vae = model_vae(config, encoder, decoder)
        return vae(x, latent_rng)
    return VAE


def get_vae(config: configlib.Config):
    if config.vae_model == "VAE_simple":
        return get_vae_model(config, VAEsimple)
    else:
        raise NotImplementedError("The VAE model is not implemented!")

def get_elbo_loss(config: configlib.Config, model_vae):
    @jax.jit
    def elbo_loss(params, batch, z_rng):
        recon_x, means, logvars = model_vae.apply(params, batch, z_rng)
        recon_loss = binary_cross_entropy(recon_x, batch).mean()
        kl_loss = kl_divergence_normal(means, logvars).mean()
        return recon_loss, kl_loss
    return elbo_loss
    
# Getting the encoder given the encoder model
def get_encoder_model(config: configlib.Config, model_encoder):
    def encoder(x):
        enc = model_encoder(config)
        return enc(x)
    return encoder


# Selecting the encoder model from all encoders
def get_encoder(config: configlib.Config):
    if config.encoder_model == "mlp":
        return get_encoder_model(config, EncoderMLP)
    else:
        raise NotImplementedError("The encoder is not implemented!")
        

# Getting the decoder given the decoder model
def get_decoder_model(config: configlib.Config, model_decoder):
    def decoder(x):
        dec = model_decoder(config)
        return dec(x)
    return decoder

# Selecting the decoder model from all decoders
def get_decoder(config: configlib.Config):
    if config.decoder_model == "mlp":
        return get_decoder_model(config, DecoderMLP)
    else:
        raise NotImplementedError("The encoder is not implemented!")


def get_vae_loss(config: configlib.Config):
    if config.vae_loss == "ELBO":
        return get_elbo_loss(config)
    else:
        raise NotImplementedError("The loss function is not implemented!")