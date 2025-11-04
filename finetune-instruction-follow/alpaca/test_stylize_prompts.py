from download_dataset import download_and_load_file
from stylize_prompts import format_input


file_path = "instruction-data.json"
url =  (
    "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
)

data = download_and_load_file(file_path, url)

for data_item in data:
   print(format_input(data_item), "<enter>")
   input()
   





