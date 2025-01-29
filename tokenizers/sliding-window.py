with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

import tiktoken
# instantiate the BPE tokenizer from tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

import time
time.sleep(2)
print(enc_text)
