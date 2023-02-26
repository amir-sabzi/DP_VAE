import jax
import jax.numpy as jnp
import haiku as hk
from models.MainModel import MainModel

import configlib

class EncoderMLP(MainModel):
    def __call__(self, x):
        batch_size = x.shape[0]
        out_dim_l1 = 500
        out_dim_l2 = 300

        print_shapes=False

        # Flatten layer
        if print_shapes: print(x.shape) 
        x = hk.Flatten()(x)

        # Dense layer 1
        if print_shapes: print(x.shape)
        x = hk.Linear(out_dim_l1, name="fc1")(x)
        x = self.activation(x) 
        
        # Dense layer 2
        if print_shapes: print(x.shape)
        x = hk.Linear(out_dim_l2, name="fc2")(x)
        x = self.activation(x)  
        
        # Mean of latent variables
        if print_shapes: print(x.shape)
        mean_x = hk.Linear(self.latent_dim, name="fc3_mean")(x)
        logvar_x = hk.Linear(self.latent_dim, name="fc3_logvar")(x)
        
        mean_x = mean_x.reshape(batch_size, -1)
        logvar_x = logvar_x.reshape(batch_size, -1) 
        return mean_x, logvar_x


