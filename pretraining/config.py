from pathlib import Path
import yaml

# read config file path from arguement
# this avoids hardcoding config paths and makes script invocation path agnostic

# the input can either be a str or a Path object, we return a dict
def load_config(config_path: str | Path) -> dict:
    path = Path(config_path) 

    # check if file exists at path
    if not path.is_file():
        raise FileNotFoundError(
            f"dllm: config file not found : {path}"
        )

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) # load YAML file to python dict

    # check syntax of config (yaml) file post load : must map to dict
    if not isinstance(config, dict):	
        raise ValueError(
            "Config file must be a YAML mapping"
        )

    # return the dict
    return config

if __name__ == "__main__": 

    import argparse # to enable command line arguements

    parser = argparse.ArgumentParser(
        # default
        description = "Load a demosthenes experiment config"
    )

    parser.add_argument(
        "--config",
        required = True,
        help = "Path to experiment YAML config for training"
    )

    args = parser.parse_args()
    config = load_config(args.config)

    # access nested values using dictionary keys
    exp_name      = config["experiment"]["name"]
    batch_size    = config["data"]["batch_size"]
    learning_rate = config["optimizer"]["learning_rate"]
    device        = config["training"]["device"]

    # print values from the dict of the YAML
    print(f"Experiment: {exp_name}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Device: {device}")
    

'''
 # - old hardcoded code: 

 1. Load the YAML into a python dictionary
config_path = Path("./configs/wiki-124m.yaml")
print(config_path)

with config_path.open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f) # load YAML file to python dict

# 2. Access nested values using dictionary keys
exp_name   = config["experiment"]["name"]
batch_size = config["data"]["batch_size"]
learning_rate = config["optimizer"]["learning_rate"]
device = config["training"]["device"]

# 3. Print values
print(f"Experiment: {exp_name}")
print(f"Batch Size: {batch_size}")
print(f"Learning Rate: {learning_rate}")
print(f"Device: {device}")
'''


