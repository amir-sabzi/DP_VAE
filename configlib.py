import argparse
import pprint
import json
from typing import Dict, Any

# Defining a global parser
parser = argparse.ArgumentParser(description="parsing simulator configs", argument_default=argparse.SUPPRESS)

# Adding arguments to the parser
p = parser.add_argument_group('configs')
p.add_argument('--load_json', type=str, metavar='PATH', help='Path to the json file containing the configs')
p.add_argument("--train_vae", default=False, action='store_true',
        help="Train the classifier.")


# With this class, instead of accessing members of dict with dict['key'], we can use dict.key to access them.

class Config(dict):
    """This is just a dict that also supports ``.key'' access, making calling config
    values less cumbersome."""
    __slots__ = []

    def __init__(self, *arg, **kw):
        super(Config, self).__init__(*arg, **kw)

    def __getattr__(self, key: str) -> Any:
        if key in self:
            return self[key]
        else:
            return None

def parse(config: Config = None, save_fname: str = "") -> Dict[str, Any]:
    """Parse given arguments."""
    if config is None:
        config = Config()

    args = parser.parse_args()
    if "load_json" in args and args.load_json:
        # Load json config if there is one.
        with open(args.load_json, 'rt') as f:
            config.update(json.load(f))
    
    
    # Parse command line again, with defaults set to the values from the json
    # config. This is needed to allow command line arguments to override the
    # json config.
    parser.set_defaults(**config)
    args = parser.parse_args()
    config.update(vars(args)) 

    # Optionally save passed arguments

    if config.save_json:
        with open(config.save_json, 'wt') as f:
            json.dumps(args, f, indent=4, sort_keys=True)

    return config

def print_config(config: Config):
    """Print the current config to stdout."""
    pprint.pprint(config)