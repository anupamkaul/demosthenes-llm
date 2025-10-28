Notes: for cloud based training to utilize GPUs: images/README.md

Perusal order:

(1) Dataset prep:

Code:
download_dataset.py (generates instruction-data.json and logs)
stylize_prompts.py  (next we can reformat the training sets to a specific prompt style (alpaca, phi-3 etc))
dataset_tuning.py   (all other operations to prepare data for the model, including batching, optimizations etc)

Tests:
test_inspect_dataset.py
test_stylize_prompts.py
test_custom_collate.py





