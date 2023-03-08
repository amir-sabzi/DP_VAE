import pickle
import os
import datetime
from pathlib import Path


import configlib



def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_results(config: configlib.Config, results_dict):
   ensure_dir()
   
   
def save_results(config: configlib.Config, results):
    if not os.path.exists(config.results_dir):
        os.mkdir(config.results_dir)

    ensure_dir(config.results_dir)

    # Create a directory based on the experiment date and time
    now = datetime.datetime.now()
    child_dir = child_dir = config.results_dir + config.experiment + "_(" + now.strftime("%Y-%m-%d_%H-%M") + ")/" 
    os.mkdir(child_dir)
    
    # save the config file in the results directory if results 
    with open(child_dir + "config.json", "w") as f: 
        f.write(str(config)) 
    
    # Save the results 
    with open(child_dir + "results.pkl", "wb") as f:
        pickle.dump(results, f)
   