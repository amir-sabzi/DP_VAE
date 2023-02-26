import haiku as hk

import configlib
from utils.vae_utils import get_vae
from data_utils.datasets import get_dataset

def train_vae(config: configlib.Config,
              vae_prng_seed = 0,
              latent_prng_seed = 1):
    
    # Getting the data set 
    train_dataset, test_dataset, train_loader, test_loader = get_dataset(config)
    dataset_size = len(train_dataset)

    # Getting our Variational Auto-Encoder 
    vae = get_vae(config)

    # Initializing the parameters of VAE 
    batch_size = config.vae_batch_size

    vae_prng_seq = hk.PRNGSequence(vae_prng_seed)
    model_vae = hk.without_apply_rng(hk.transform(vae))
    latent_prng_seq = hk.PRNGSequence(latent_prng_seed)
    sample_img, _ = next(iter(train_loader))
    params_vae = model_vae.init(rng=next(vae_prng_seq),
                                x=sample_img,
                                latent_rng=next(latent_prng_seq))
    
    
    