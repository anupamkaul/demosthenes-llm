'''
Now that we know how to calculate losses across the batch-sets of training and validation
data (see loss-training-validation.py) we will implement the code for pretraining my LLM (GPTModel)
'''

import torch

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):

    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):

        # see parent-child-basics.py : 
        # train is a method of nn (parent of GPTModel, see GPTModel.py in llm-infra)

        model.train()
    
        for input_batch, target_batch in train_loader:
 
            # reset the loss gradient from the previous batch iteration
            optimizer.zero_grad() 

            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )

            # calculate new gradients using back-prop
            loss.backward()

            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1

            # optional evaluation step
            if global_step % eval_freq == 0:

                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter 
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}"
                     )

                # pause for user to ack and continue (just because I am printing a lot of stuff currently)
 
                # TODO : good inspection to evaluate training outputs per steps of epoch. Disabling for now
                # This pause should be configurable in the code

                # input("Press enter to continue..")

            # print a sample text after each iteration to show visual/understandable progress (!) 
            generate_and_print_sample(model, tokenizer, device, start_context)

            # input("Press enter to continue..")

    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):

    model.eval()
 
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )    

    with torch.no_grad():
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )    
    
    model.train()
    
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):

    model.eval()

    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
            
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
   
    model.train()

# main
import sys, os
sys.path.append( os.path.join( os.path.dirname(os.path.abspath(__file__)),  '../llm-infra/') )

from GPTModel import GPTModel

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12, 
    "drop_rate": 0.1,
    "qkv_bias": False
}

torch.manual_seed(123)

model = GPTModel(GPT_CONFIG_124M)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)

model.to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0004, weight_decay=0.1     
)

num_epochs=10

# import everything that I need for this code to compile..

import ltv # had to shorten the name loss-training-validation for python import syntax 
from ltv import train_loader, val_loader, calc_loss_batch, calc_loss_loader

import textgenerate
from textgenerate import text_to_token_ids, generate_text_simple, token_ids_to_text

import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)


