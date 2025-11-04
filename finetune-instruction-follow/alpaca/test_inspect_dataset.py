from download_dataset import download_and_load_file

file_path = "instruction-data.json"
url =  (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)

data = download_and_load_file(file_path, url)
print("Number of entries:", len(data), " <enter>")
input()

# inspect the data
for dataitem in data:
    print(dataitem, " <enter>")
    input()

