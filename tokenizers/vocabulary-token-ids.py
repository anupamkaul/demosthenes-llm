# run as python vocabulary-token-ids.py| more

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

import re
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item for item in preprocessed if item.strip()] # remove extra white spaces (optional)

# next I will assign every unique token to its vocabulary (token id structure)

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size) # 1130 exactly from "the-verdict" story
print(all_words) # sorted list of all unique words

# create the vocabulary (python map)

vocab = { token: integer for integer,token in enumerate(all_words)  }
print("\nhere is the vocabulary or token ids:")
print(vocab)

# another way of displaying the info
print("\npretty print of words mapped to token ids:")
for i, item in enumerate(vocab.items()):
    print(item)








