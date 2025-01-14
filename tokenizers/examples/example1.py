# create vocabulary (from a reference text, or actually via a trained model)

'''
ToDo: The Vocab code can be containerized as a class (filepath) that returns a vocab object
'''

with open("../the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

import re
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item for item in preprocessed if item.strip()] # remove extra white spaces (optional)

# next I will assign every unique token to its vocabulary (token id structure)

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
#print(vocab_size) # 1130 exactly from "the-verdict" story
#print(all_words) # sorted list of all unique words

# create the vocabulary (python map)
vocab = { token: integer for integer,token in enumerate(all_words)  }

# create links of files from parent folder whose modules are to be imported here
# i.e. ln -s <path to original file> <path to link>

import SimpleTokenizerV1 as t

# Example 1:

tokenizer = t.SimpleTokenizerV1(vocab)
text = """It's the last he painted, you know,"
       Mrs Gisburn said with pardonable pride."""

ids = tokenizer.encode(text)
print(ids)

print(tokenizer.decode(ids))

# Example 2: 

text = " you like to have some tea ?"
print(tokenizer.encode(text))
print(tokenizer.decode(tokenizer.encode(text)))

# Example 3: 

text = "like you like to have some tea ?"
print(tokenizer.encode(text))
print(tokenizer.decode(tokenizer.encode(text)))

'''
# Example 4:  (Key-Error as "Like" is not in vocab, but like is)

text = "Like you like to have some tea ?"
print(tokenizer.encode(text))
print(tokenizer.decode(tokenizer.encode(text)))

# Example 5: (Key-Error as "Would" is not in vocab. Vocab was generated from "the-verdict.txt")  

text = "Would you like to have some tea ?"
print(tokenizer.encode(text))
print(tokenizer.decode(tokenizer.encode(text)))
'''






