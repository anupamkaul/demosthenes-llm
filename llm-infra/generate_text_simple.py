import torch
import torch.nn as nn

def generate_text_simple(model, idx, max_new_tokens, context_size):

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
    
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]   # remember its [batch_size, num_token, vocab_size] shape so taking the last row
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

# test this code:
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
print("encoded: ", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("encoded_tensor.shape: ", encoded_tensor.shape)

'''
encoded:  [15496, 11, 314, 716]
encoded_tensor.shape:  torch.Size([1, 4])
'''

import GPTModel as gpt
import GPT_CONFIG_124M as gpt2_cfg

torch.manual_seed(123)
#model = gpt.GPTModel(gpt2_cfg.get_GPT_CONFIG_GPT2_SMALL()) # Hello I am feature
model = gpt.GPTModel(gpt2_cfg.get_GPT_CONFIG_GPT2_MEDIUM()) # Hello I am turnover
#model = gpt.GPTModel(gpt2_cfg.get_GPT_CONFIG_GPT2_LARGE())  # Hello I amironically

model.eval()
out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=gpt2_cfg.get_GPT_CONFIG_GPT2_SMALL()["context_length"]  # or, 1024
        #context_size=1024
)

print("Output: ", out)
print("Output length:", len(out[0]))

decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)




