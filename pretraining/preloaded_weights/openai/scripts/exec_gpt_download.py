from gpt_download import download_and_load_gpt2

settings, params = download_and_load_gpt2(
    model_size="124M", models_dir="gpt2"
)

# lets inspect settings and parameter dictionary keys

print("Settings: ", settings)
print("Params: ", params.keys())
