import jax
import jax.numpy as jnp
import optax


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


def gaussian_log_likelihood(recon_means_x, recon_logvars_x, x):
    
    # loss_aux = -jnp.sum(((-0.5 / jnp.exp(recon_logvars_x)) * (x - recon_means_x) ** 2.0))
    loss_aux = jnp.max(recon_logvars_x)
    loss = -jnp.sum(
            (-0.5 * jnp.log(2.0 * jnp.pi))
            + (-0.5 * recon_logvars_x)
            + ((-0.5 / jnp.exp(recon_logvars_x)) * (x - recon_means_x) ** 2.0),
        )
    return loss, loss_aux

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

def get_elbo_discrete_loss_fn(config: configlib.Config, model_vae):
    @jax.jit
    def elbo_loss(params, batch, z_rng):
        recon_x, latent_means, latent_logvars = model_vae.apply(params, batch, z_rng)
        recon_loss = binary_cross_entropy(recon_x, batch).mean()
        kl_loss = kl_divergence_normal(latent_means, latent_logvars).mean()
        return recon_loss + kl_loss
    return elbo_loss


def get_elbo_continuous_loss_fn(config: configlib.Config, model_vae):
    @jax.jit 
    def elbo_loss(params, batch, z_rng):
        recon_means_x, recon_logvars_x, latent_means, latent_logvars = model_vae.apply(params, batch, z_rng)
        recon_loss, recon_loss_aux = gaussian_log_likelihood(recon_means_x, recon_logvars_x, jnp.reshape(batch, (batch.shape[0], -1)))
        kl_loss = kl_divergence_normal(latent_means, latent_logvars)
        aux = {'recon_loss': recon_loss, 'kl_loss': kl_loss, 'recon_loss_aux': recon_loss_aux}
        return recon_loss + kl_loss, aux
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


def get_vae_loss_fn(config: configlib.Config, model_vae):
    if config.vae_loss == "elbo_discrete":
        return get_elbo_discrete_loss_fn(config, model_vae)
    if config.vae_loss == "elbo_continuous":
        return get_elbo_continuous_loss_fn(config, model_vae)
    else:
        raise NotImplementedError("The loss function is not implemented!")
    
def get_vae_update_fn(loss_fn, opt):
    def update_model(opt_state, params, prng_key, batch):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch, prng_key)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return loss, aux, params, opt_state
    return update_model 


