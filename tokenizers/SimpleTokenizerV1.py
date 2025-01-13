'''
Implementation of a simple text tokenizer with encode (to tokenIDs)
and decode (reverse IDs to text) methods

Remember vocab is a pythom map
'''

# store vocabulary as a class attribute for access in the encode/decode methods
import re

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items() }

    def encode(self, text):

        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item for item in preprocessed if item.strip()
        ] # remove extra white spaces (optional)

        ids = [self.str_to_int[s] for s in preprocessed] 
        return ids

    def decode(self, ids):

        text = " ".join( [ self.int_to_str[i] for i in ids ] )
        return text



