'''
A placeholder GPT model architecture class

Calls DummyTransformerBlock, DummyLayerNorm,
Reads and applies values from config file
'''

import torch
import torch.nn as nn

class DummyGPTModel(nn.Module):

    '''
    The DummyGPTModel class in this code defines a simplified version of a GPT-like
    model using PyTorch's neural network module (nn.Module). The model architecture 
    in the DummyGPTModel class consists of token and positional embeddings, dropout,
    a series of transformer blockers (DummyTransformerBlock), a final layer normalization
    (DummyLayerNorm), and a linear output layer (out_head). The configuration is passed
    in via a Python dictionary, the GPT_CONFIG_124M dictionary that was created earlier
    '''

    def __init__(self, cfg):
        super().__init__()

        # get values from the config file
        self.tok_emb  = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])  
        self.pos_emb  = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # using a placeholder for TransformerBlock
        self.trf_blocks = nn.Sequential(
             *[DummyTransformerBlock(cfg)
              for _ in range(cfg["n_layers"])]
        )

        # using a placeholder for LayerNorm
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])

        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias = False
        )

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
Placeholders for DummyTransformerBlock 
and DummyLayerNorm
'''

class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return x

class DummyLayerNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return x



        


 

