# from https://github.com/openai/tiktoken?tab=readme-ov-file
# code highlights usage of BPE tokenizer

# pip install tiktoken

from importlib.metadata import version
import tiktoken
print("tiktoken version: ", version("tiktoken"), "\n")

# instantiate the BPE tokenizer from tiktoken
print("instantiating Byte Pair Encoding tokenization (tiktoken)..\n")
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

# can see that endoftext is assigned id 50256. Total vocab is 50257

text = "Akwiewier"

print("\nencoding unknown words (Akwiewier)..\n")
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)

print("decoding ..\n")
strings = tokenizer.decode(integers)
print(strings)




