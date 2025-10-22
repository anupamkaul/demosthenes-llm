'''
Code to check inference (actually classify inputs as spam vs ham)
Can be fixed inputs or a chat format
'''

# load the model
import torch

import sys, os
sys.path.append( os.path.join( os.path.dirname(os.path.abspath(__file__)),  '../llm-infra/') )

import time
start_time = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"
FT_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True
}
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
FT_CONFIG.update(model_configs[CHOOSE_MODEL])

from GPTModel import GPTModel
model = GPTModel(FT_CONFIG)

model.out_head = torch.nn.Linear(
    in_features  = FT_CONFIG["emb_dim"],  # 768, as before
    out_features = 2                      # 2 now
)

model_state_dict = torch.load("./review_classifier.pth", map_location=device)
#print(model_state_dict)

model.load_state_dict(model_state_dict)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"model loaded in {execution_time_minutes:.2f} minutes.")

import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

def classify_review(
        text, model, tokenizer, device, max_length=None,
        pad_token_id=50256):

    model.eval()

    # prepare inputs to the model
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]

    # truncate sequences if they are too long
    input_ids = input_ids[:min(
        max_length, supported_context_length
    )]

    # pad sequences to the longest sequence
    input_ids += [pad_token_id] * (max_length - len(input_ids))

    input_tensor = torch.tensor(
        input_ids, device=device
    ).unsqueeze(0)              # adds batch dimension

    with torch.no_grad():       # models inference without gradient tracking
        logits = model(input_tensor)[:, -1, :]    # logits of the last output token
    predicted_label = torch.argmax(logits, dim=-1).item()

    return "spam" if predicted_label == 1 else "not spam"  


# let's test classification now:


import spam_dataloader
from spam_dataloader import train_dataset

text_1 = (
    "You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award."
)

print("\nCheck for Spam: ", text_1)
print(classify_review(
    text_1, model, tokenizer, device, max_length=train_dataset.max_length
))

text_2 = (
    "Hey, just wanted to check if we're still on"
    " for dinner tonight? Let me know!"
)

print("\nCheck for Spam: ", text_2)
print(classify_review(
    text_2, model, tokenizer, device, max_length=train_dataset.max_length
))



