from datasets import load_dataset

'''
# -- Had to abandon this code as new methods suggest using Parquet 

# Load a specific version of English Wikipedia
# Using 'streaming=True' is recommended for large datasets to avoid downloading 20GB+ at once

dataset = load_dataset("wikipedia", "20220301.en", split='train', streaming=True, trust_remote_code=True)

# add the trust_remote_code = True as HuggingFace recently disabled the execution of local or remote 
# python script for security reasons
'''

# Use the 'wikimedia' repo which doesn't use old loading scripts
dataset = load_dataset(
    "wikimedia/wikipedia", 
    "20231101.en",      # Newer date format
    split='train', 
    streaming=True
)

# Test it
for article in dataset.take(1):
    print(f"Success! Loaded: {article['title']}")

# Take a small sample (e.g., 1 articles) for testing
sample_data = dataset.take(1)

# print the contents
# (since the dataset is a collection of dictionaries, I can iterate through the sample to print
#  articles titles and some text of my choosing)

for i, articles in enumerate(sample_data):

    print(f"--- Article {i+1}: {article['title']} ---")
    print(article['text'][:1200])  # Print first 1200 characters
    print("\n")

print("done")






