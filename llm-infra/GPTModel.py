'''
Replace DummyGPTModel.py with the real GPTModel class. See "GPT-Model-Arch.png" for arch.
The TransformerBlock is sequenced to run 12 times ('n_layers'). (Look at shortcut_example.py for good examples of NN layering)
'''

import torch
import torch.nn as nn
import TransformerBlock as trf
import LayerNorm as lnm

class GPTModel(nn.Module):

    '''
    The GPTModel class in this code defines a simplified version of a GPT-like
    model using PyTorch's neural network module (nn.Module). The model architecture 
    in the GPTModel class consists of token and positional embeddings, dropout,
    a series of transformer blockers (TransformerBlock), a final layer normalization
    (LayerNorm), and a linear output layer (out_head). The configuration is passed
    in via a Python dictionary, the GPT_CONFIG_124M dictionary that was created earlier
    '''

    def __init__(self, cfg):
        super().__init__()
    
        # initialize token, position and drop embeddings from config
        self.tok_emb  = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])  
        self.pos_emb  = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # set up transformer blocks sequentially n_layer times (usually 12 for GPT2)
        self.trf_blocks = nn.Sequential( *[trf.TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        # set up a final layer normalization (different from the layer-norm in transformerblock) 
        self.final_norm = lnm.LayerNorm(cfg["emb_dim"])

        # set up a final linear output layer (see GPT-Model-Arch.png) 
        self.out_head = nn.Linear( cfg["emb_dim"], cfg["vocab_size"], bias = False)


    '''
    The forward method describes the data flow through the model: it computes token
    and positional embeddings for the input indices, applies dropout, processes the data
    through the transformer blocks, applies normalization, and finally produces logits
    with the linear output layer
    '''

    def forward(self, in_idx):

        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


'''
Notes:

Thanks to the TransformerBlock class, the GPTModel class is relatively small and compact.

The __init__ constructor of this GPTModel class initializes the token and positional embedding 
layers using the configurations passed in via a Python dictionary, cfg. These embedding layers 
are responsible for converting input token indices into dense vectors and adding positional information
(see folder tokenizers)

Next, the __init__ method creates a sequential stack of TransformerBlock modules equal to the number of 
layers specified in cfg. Following the transformer blocks, a LayerNorm layer is applied, standardizing the 
outputs from the transformer blocks to stabilize the learning process. Finally, a linear output head without 
bias is defined, which projects the transformer’s output into the vocabulary space of the tokenizer to generate 
logits for each token in the vocabulary.

The forward method takes a batch of input token indices, computes their embeddings, applies the positional embeddings, 
passes the sequence through the transformer blocks, normalizes the final output, and then computes the logits, 
representing the next token’s unnormalized probabilities. We will convert these logits into tokens and text outputs 
in the next section.

'''
