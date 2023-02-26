import jax
import jax.numpy as jnp
import haiku as hk

import configlib

class MainModel(hk.Module):
    def __init__(self, config: configlib.Config):
        super().__init__()
        self.config = config
        self.latent_dim = config.latent_dim
        self.decoder_output_dim = config.decoder_output_dim
         
        if config.activation == "relu":
            self.activation = jax.nn.relu
        elif config.activation == "leaky_relu":
            self.activation = lambda x: jax.nn.leaky_relu(x, negative_slope=config.negative_slope)
        elif config.activation == "tanh":
            self.activation = jnp.tanh
        elif config.activation == "elu":
            self.activation = lambda x: jax.nn.elu(x, alpha=config.elu_alpha)
        else:
            raise NotImplementedError

