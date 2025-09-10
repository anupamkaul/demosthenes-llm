from gpt_download import download_and_load_gpt2
# download_and_load_gpt2 loads all params into ram (python)

settings, params = download_and_load_gpt2(
    model_size="124M", models_dir="gpt2"
)

# lets inspect settings and parameter dictionary keys

print("Settings: ", settings)
print("Params: ", params.keys())

#print(params) # all params (big data of weights)

print("The weights of the token embedding layers are:", params["wte"])
print("The dims are:", params["wte"].shape)

# now that the params (weights) and settings are in RAM, I need to move
# them to my GPTModel instance for demosthenes




