# create vocabulary (from a reference text, or actually via a trained model)

# this example shows how the LLM (the tokenizer) will handle words it doesn't
# recognize in its vocabulary

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
all_words.extend(["<|endoftext|>" , "<|unk|>"])

vocab_size = len(all_words)
print(vocab_size) # 1130 + 2 exactly from "the-verdict" story (the extra 2 being endoftext and unk)

#print(all_words) # sorted list of all unique words

# create the vocabulary (python map)
vocab = { token: integer for integer,token in enumerate(all_words)  }

#print the last 5
for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)

exit

# create links of files from parent folder whose modules are to be imported here
# i.e. ln -s <path to original file> <path to link>

import SimpleTokenizerV2 as t

# Example 1:

tokenizer = t.SimpleTokenizerV2(vocab)
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

# Example 4 Fixed:  (Key-Error as "Like" is not in vocab, but like is) (replace with <|unk|>)

text = "Like you like to have some tea ?"
print(tokenizer.encode(text))
print(tokenizer.decode(tokenizer.encode(text)))

# Example 5: Fixed :(Key-Error as "Would" is not in vocab. Vocab was generated from "the-verdict.txt")  (replace with |<unk>|)

text = "Would you like to have some tea ?"
print(tokenizer.encode(text))
print(tokenizer.decode(tokenizer.encode(text)))

# Example 6 : use <|endoftext?>
print("\nexample of endoftext\n")
text1 = "Hello Anupam, do you like tea?" # "Hello" and "Anupam" not in vocab
text2 = "In the sunlit bodaceaous terraces of the palace" # the b word is not even a word
text = "<|endoftext|>".join((text1, text2)) # replicating logic that would isolate unrelated documents of text (or of context)
print(text, "\n")

import SimpleTokenizerV2 as V2
tokenizer = V2.SimpleTokenizerV2(vocab)
print("encoded: ")
print(tokenizer.encode(text))
print("decoded: ")
print(tokenizer.decode(tokenizer.encode(text)))







