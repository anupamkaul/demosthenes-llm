import torch
import torch.nn as nn

# a better version of generate_text_simple can adopt temperature scaling, top-k sampling etc

def generate_text_simple(model, idx, max_new_tokens, context_size):

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
    
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]   # remember its [batch_size, num_token, vocab_size] shape so taking the last row
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True) # this is greedy decoding, and prevents variety on repeated invokations
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

