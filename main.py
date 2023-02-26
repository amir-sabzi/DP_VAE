import os

import configlib
import training.vae 


# Do not allocate all GPU memory to this program
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'

def main():
    # load the config file
    config = configlib.parse(save_fname="tmp.txt")
    configlib.print_config(config) 
    print(config.load_json)
    if config.load_json:
        training.vae.train_vae(config)
        
  

if __name__ == "__main__":
    main()