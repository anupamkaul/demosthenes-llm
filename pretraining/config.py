from pathlib import Path
import yaml

# 1. Load the YAML into a python dictionary
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


