'''
Implementation of a simple text tokenizer with encode (to tokenIDs)
and decode (reverse IDs to text) methods

Remember vocab is a pythom map

V2: We handle unknown words. We also need to address the usage and addition
of special context tokensthat can enhance a model's understanding of context
or other relevant info in the text. Special tokens can include markers for 
unknown words and document boundaries. We will supply new tokens : <|unk|> 
and <|endoftext|>

We use <|unk|> token if it encouters a word not part of vocab. We add a token
between unrelated texts to indicate to the LLMs that some contexts might be
unrelated to each other.

Basically compared to V1, V2 replaces unknown words with <|unk|> tokens

'''

# store vocabulary as a class attribute for access in the encode/decode methods
import re

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items() }

    def encode(self, text):

        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item for item in preprocessed if item.strip()
        ] # remove extra white spaces (optional)

        # such a beautiful expression below: 
        preprocessed = [item if item in self.str_to_int
                        else "<|unk|>" for item in preprocessed]

        ids = [self.str_to_int[s] for s in preprocessed] 
        return ids

    def decode(self, ids):

        # convert token ids back into text
        text = " ".join( [ self.int_to_str[i] for i in ids ] )

        # remove spaces before the specified punctuation
        text = re.sub( r'\s+([,.?!"()\'])', r'\1', text )
        return text



