import jax
import jax.numpy as jnp
import haiku as hk
from models.MainModel import MainModel

import configlib

class DecoderMLP(MainModel):
    def __call__(self, x):
        batch_size = x.shape[0]
        out_dim_l1 = 300
        out_dim_l2 = 500
        max_var = 10
        print_shapes=False

        # Dense layer 1
        if print_shapes: print(x.shape)
        x = hk.Linear(out_dim_l1, name="fc1")(x)
        x = self.activation(x) 
        
        # Dense layer 2
        if print_shapes: print(x.shape)
        x = hk.Linear(out_dim_l2, name="fc2")(x)
        x = self.activation(x)  
        
        # Reconstruction layer
        if print_shapes: print(x.shape)

        means_x = hk.Linear(self.decoder_output_dim, name="fc_final_means")(x)
        # means_x = jax.nn.tanh(means_x)

        logvars_x = hk.Linear(self.decoder_output_dim, name="fc_final_logvars")(x)
        logvars_x = jax.nn.tanh(logvars_x) * 10 

        means_x = means_x.reshape(batch_size, -1) 
        logvars_x = logvars_x.reshape(batch_size, -1) 
        return means_x, logvars_x


