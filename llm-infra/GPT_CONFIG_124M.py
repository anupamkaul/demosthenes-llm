GPT_CONFIG_124M = {
    "vocab_size"     : 50257,       # vocabulary size
    "context_length" : 1024,        # context length
    "emb_dim"        : 768,         # embedding dimension
    "n_heads"        : 12,          # number of attention heads
    "n_layers"       : 12,          # number of layers
    "drop_rate"      : 0.1,         # dropout rate
    "qkv_bias"       : False        # query-key-value bias

}

def get_GPT_CONFIG_124M():
    return GPT_CONFIG_124M

'''
notes:

- vocab_size refers to a vocabulary of 50.527 words, as used by the BPE tokenizer from pytorch that we have
been using previously

- context_length denotes the maximum number of input tokens that the model can handle via positional embeddings
that we have used previously

- emb_dim represents the embedding size, transforming each token into a 768-dimension vector

- n_heads indicates the count of attention heads in the nulti-head attention mechanism

- n_layers specifies the number of transformer blocks in the model (new), which we will cover here

- drop_reate indicatesthe intensity of the dropout mechanism (0.1 implies a 10% random drop out of hidden units) 
and this is to prevent overfitting

- qkv_bias determines whether to include a bias vector in the Linear layers of the multi-head attention for query key and 
vaue computations. We will initially disable this, following the norms of modern LLMs and will later revisit this.

'''
