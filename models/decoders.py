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
        print_shapes=True

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
        x = hk.Linear(self.decoder_output_dim, name="fc_final")(x)
        x = self.activation(x)  

        x = x.reshape(batch_size, -1) 
        return x


