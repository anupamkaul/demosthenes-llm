'''
Data Sampling with a sliding window on the input text

Given a text sample, extract input blockers as subsamples
that serve as input to the LLM. The LLM's prediction task
during training is to predict the next word that follows the
input block. During training, we mask out all words that are
past the target. The text first undergoes tokenization
'''


with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

import tiktoken
# instantiate the BPE tokenizer from tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

'''
import time
time.sleep(2)
print(enc_text)
'''

# remove the first 50 tokens for demo purposes / more interesting samples
enc_sample = enc_text[50:]

# simplest input-target pairing : x and y
# x is inputs-token and y is targets (here inputs shifted by 1)

context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]

print(f"x:  {x}")
print(f"y:       {y}")

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "---->", desired)

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))



