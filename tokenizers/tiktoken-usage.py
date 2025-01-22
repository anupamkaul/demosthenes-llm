# from https://github.com/openai/tiktoken?tab=readme-ov-file
# code highlights usage of BPE tokenizer

# pip install tiktoken

from importlib.metadata import version
import tiktoken
print("tiktoken version: ", version("tiktoken"))

# instantiate the BPE tokenizer from tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

# usage is similar to SimpleTokenizerV2

text = (
    "Hello Anupam, do you like tea? <|endoftext|> In the sunlit terraces"
    "of someunknownPlace"
)

print("encoding ..\n")
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)

print("decoding ..\n")
strings = tokenizer.decode(integers)
print(strings)






