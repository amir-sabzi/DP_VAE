import haiku as hk
import optax
from tqdm import tqdm

import configlib
from utils.vae_utils import get_vae, get_vae_loss_fn, get_vae_update_fn
from data_utils.datasets import get_dataset
from utils.logging_utils import save_results
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
    
    
    # Optimizer (for now just go with Adam, we'll write a getter for different optimizers in future)
    opt_vae = optax.adam(learning_rate=config.vae_lr)
    opt_state = opt_vae.init(params_vae)
    
    # Getting the VAE loss function
    loss_fn_vae = get_vae_loss_fn(config, model_vae)
    
    # Getting the update function
    update_fn = get_vae_update_fn(loss_fn_vae, opt_vae)
    
    results = {'losses': []}
    for e in range(config.vae_epochs):
        total_loss = 0
        pbar_it = tqdm(train_loader)
        for img, label in pbar_it:
            # print(img[0])
            loss, aux, params_vae, opt_state = update_fn(opt_state, params_vae, next(vae_prng_seq), img)
            # print("-------------")
            recon_loss = aux['recon_loss']
            kl_loss = aux['kl_loss']
            recon_loss_aux = aux['recon_loss_aux']
        print(f'loss: {loss}')
            # print(f'recon_loss {recon_loss}') 
            # print(f'kl_loss {kl_loss}') 
            # print(f'recon_loss_aux {recon_loss_aux}') 
        
        results['losses'].append(loss)
         
    save_results(config, results)
            
    
            