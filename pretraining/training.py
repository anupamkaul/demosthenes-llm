'''
Now that we know how to calculate losses across the batch-sets of training and validation
data (see loss-training-validation.py) we will implement the code for pretraining my LLM (GPTModel)
'''

import torch

from dual_writer import DualWriter
import sys
sys.stdout = DualWriter("dump_training.txt")

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):

    train_losses, val_losses, track_tokens_seen = [], [], []  # initialize lists to track losses and tokens seen
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):                           # main training loop

        # see parent-child-basics.py : 
        # train is a method of nn (parent of GPTModel, see GPTModel.py in llm-infra)

        model.train() # set up training mode, i.e. enable dropouts, batch normalization (more memory computes)
    
        for input_batch, target_batch in train_loader:

            print("debug: len input batch: ", len(input_batch), " len target batch: ", len(target_batch), " len train loader : ", len(train_loader))
 
            # reset the loss gradient from the previous batch iteration
            optimizer.zero_grad()                             # reset loss gradients from previous batch iteration

            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )

            # calculate new gradients using back-prop
            loss.backward()                                   # calculate loss gradients

            optimizer.step()                                  # update weight model using loss gradients

            #print("input batch's numel : ", input_batch.numel(), "\n") (this is normally 512)

            tokens_seen += input_batch.numel()
            global_step += 1

            print("debug: global_step : ", global_step, " tokens seen: ", tokens_seen)

            # optional evaluation step
            if global_step % eval_freq == 0:                  # optional evaluation step

                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter 
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                print(f"Epoch {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}"
                     )

                # pause for user to ack and continue (just because I am printing a lot of stuff currently)
                # TODO : good inspection to evaluate training outputs per steps of epoch. Disabling for now
                # This pause should be configurable in the code

                # input("Press enter to continue..")

        print("\nout of inner input_batch loop..")
        # print a sample text after each iteration to show visual/understandable progress (!) 
        generate_and_print_sample(model, tokenizer, device, start_context) # print a sample text after each epoch

        # input("Press enter to continue..")

    # save the model before we return
    torch.save(model.state_dict(), "./model/model.pth")
    print("model saved\n")

    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):

    model.eval() # turn off training mode (disable dropout, batch normalization for better calc efficiency, lower mem)
 
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )    

    with torch.no_grad():
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )    
    
    model.train() # turn training mode back on
    
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):

    print("generate and print sample..\n")
    model.eval() # turn off training mode (see above)

    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
            
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
   
    model.train() # turn back training mode

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


# load previously saved instance of the model (to continue training) and the training states
# enable this once I save it first time (I don't know the format, so let the output drive the input)

try:
    model.load_state_dict(torch.load("./model/model.pth", map_location=device))
    print("loaded previously saved model to continue training..")
    input()
except FileNotFoundError:
    print("model not found on disk. monitor as a one time thing, error out if repeats")


#num_epochs=10
num_epochs=20
#num_epochs=2 # just to debug plot_losses 

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

print("tokens seen: ", tokens_seen)  # debug why tokens_seen is incorrect

# plot out a graph that shows training and validation losses side by side
# (to help detect if we are overfitting etc)

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):

    #print("In plot_losses\n")
    #print("len of epochs seen (X of 1st axis)", epochs_seen.shape)
    #print("len of tokens seen (X of 2nd axis)", len(tokens_seen))
    #print("len of train_losses(Y of twiny)", len(train_losses))
    #print("len of val_losses(Y of twiny)", len(val_losses))
   

    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = ax1.twiny()    # twiny in pyplot creates a second axis object that shares the y-axis with existing axes object
                         # so the y-axis have to be the same
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()

    plt.show()

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses)) 
#linspace creates 1 dim tensor (start, end, steps=STEPS, out=None, dtype=None, ..etc)

sys.stdout.close() # Close the file at the end

plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

