'''
Lets now prep up the infra for writing up the loss function
and dividing the data set into training and validation sizes
'''

import torch

'''
Implement a utility function to calculate the cross entropy loss
of a given batch returned via the training and validation loader

(This is where the model's forward is called for the AI processing)
'''

def calc_loss_batch(input_batch, target_batch, model, device):

    input_batch =  input_batch.to(device) # optimize
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )

    return loss

'''
We now use the above loss utility function and implement the following
calc_loss_loader function that computes the loss over all the batches
that are sampled by a given data loader
'''

def calc_loss_loader(data_loader, model, device, num_batches=None):

    total_loss = 0

    # sanity checks
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)    
            total_loss += loss.item() # sum the loss for each batch
        else:
            break

    # return the average loss over all batches
    return total_loss / num_batches 

# evaluate the model being trained
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

# generate encoded text
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # unsqueeze(0) adds the batch dimension
    return encoded_tensor

# generate decoded text
def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

# generate_text_simple
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

# print decoded text
def generate_and_print_sample(model, tokenizer, device, start_context):

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

