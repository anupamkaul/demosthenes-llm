'''

prepare the actual openAI GPT2 wt trained demosthenes model now.
next, replace out_head to classify only 2 output variables that
are required for classification finetuning

First, some choice configurations

'''

CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"
BASE_CONFIG = {
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
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

'''
Next, we import the download_and_load_gpt2 function from gpt_download.py and 
we reuse the GPTModel class (demosthenes) and load_weights_into_gpt function
from pretraining to download the weights into the GPT model. A hackier way is
also to simply read the model that I stored to the disk but I will skip that
for now and re-create the pretrained demosthenes model now

Starting point is ../pretraining/preloaded_weights/openai/scripts
'''

import sys, os
sys.path.append( os.path.join( os.path.dirname(os.path.abspath(__file__)),  '../pretraining/preloaded_weights/openai/scripts') )
sys.path.append( os.path.join( os.path.dirname(os.path.abspath(__file__)),  '../llm-infra/') )

from gpt_download import download_and_load_gpt2
from load_wts_to_gpt import load_weights_into_gpt
from GPTModel import GPTModel

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(
    model_size=model_size, models_dir="gpt2"
)

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()
print("model with loaded weights is ready")

'''
After loading the model weights into the GPTModel, we reuse the text generation utility function 
from previous work to ensure that the model generates coherent text:
'''

from textgenerate import text_to_token_ids, generate_text_simple, token_ids_to_text, generate

import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

text_1 = "Every effort moves you"
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text_1, tokenizer),
    max_new_tokens=15,
    context_size=BASE_CONFIG["context_length"]
)
print(token_ids_to_text(token_ids, tokenizer))

# now check if classification prompts work (they won't as we
# haven't fine-tuned the model for classification prompts yet)

print("\n\nnow checking for classification prompts:")

text_2 = (
    "Is the following text 'spam'? Answer with 'yes' or 'no':"
    " 'You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award.'"
)
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text_2, tokenizer),
    max_new_tokens=23,
    context_size=BASE_CONFIG["context_length"]
)
print(token_ids_to_text(token_ids, tokenizer))

'''
To prep the model for classification fine tuning, we replace the 
original output layer, which maps the hidden representation of 768
nodes to 50,257 (the token vocabulary) to 2 classes (0 - not spam, and
1 - spam). We use the same model as above which is pretrained, except
that we will replace the output layer and then train this model such
that only the edge most nodes of the outermost layer that we have replaced,
are tuned, thus achieving "fine tuning" of this model geared towards
classification.

Fine-tuning selected layers vs. all layers

Since we start with a pretrained model, it’s not necessary to fine-tune 
all model layers. In neural network-based language models, the lower layers 
generally capture basic language structures and semantics applicable across 
a wide range of tasks and datasets. So, fine-tuning only the last layers 
(i.e., layers near the output), which are more specific to nuanced linguistic 
patterns and task-specific features, is often sufficient to adapt the model to 
new tasks. A nice side effect is that it is computationally more efficient to 
fine-tune only a small number of layers.

'''

print("\n", model)

# first, freeze the model, meaning that we make all layers non-trainable

for param in model.parameters():
    param.requires_grad = False

# next, we replace the output layer (model.out_head -- see logs.txt) which
# originally maps to the size of the vocab, to 2

import torch
torch.manual_seed(123)
num_classes = 2

model.out_head = torch.nn.Linear(
    in_features  = BASE_CONFIG["emb_dim"],  # 768, as before
    out_features = num_classes              # 2 now
)

# note that this new model_out.head has requires_grad set to True by default
# so this will be the only layer in the model that will be updated during
# training. We also configure the last transformer block (accessed as -1) 
# and the final LayerNorm module, which connects this block to the output layer,
# to be trainable.

for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True
for param in model.final_norm.parameters():
    param.requires_grad = True

print(model)

'''
above shows that model.out_head maps to 2. At this time requires_grad field for
params is false for every layer except the last trf and layer_norm layers and the
final output block.
'''

'''
Lets now check that freezing the model doesn't impact the model's operations, i.e.
I should still be able to feed the model text, and grab out all of the output layers
with the difference being that the final tensor dims should be 2 instead of 50247
'''

print("\ncheck that the frozen model (except the final layers) is still able to operate on text as expected\n")

inputs = tokenizer.encode("Do you have time")	
inputs = torch.tensor(inputs).unsqueeze(0)
print("Inputs:", inputs)
print("Input dims:", inputs.shape)    # shape(batch, size, num_tokens)

'''
The print output shows that the preceding code encodes the inputs into a tensor consisting of four input tokens:

Inputs: tensor([[5211,  345,  423,  640]])
Inputs dimensions: torch.Size([1, 4])

'''

# now we pass the encoded text into the frozen model and get the output tensors

with torch.no_grad():
    output = model(inputs)

print("Output: ", output)
print("Output dims", output.shape)    # shape(batch, size, num_tokens)

'''
The output tensor looks like the following:

Outputs:
 tensor([[[-1.5854,  0.9904],
          [-3.7235,  7.4548],
          [-2.2661,  6.6049],
          [-3.5983,  3.9902]]])
Outputs dimensions: torch.Size([1, 4, 2])

A similar input would have previously produced an output tensor of [1, 4, 50257], where 50257 
represents the vocabulary size. The number of output rows corresponds to the number of input tokens 
(in this case, four). However, each output’s embedding dimension (the number of columns) is now 2 
instead of 50,257 since we replaced the output layer of the model.
'''

'''

Given how causal attention works, the last token actually holds all of the attention query info
from the previous tokens. So we don't need to fine tune the 4 rows (output). Instead we will
focus on the last row corresponding to the last output token (see last_output_token.png)

Given the causal attention mask setup, the last token in a sequence accumulates the most information 
since it is the only token with access to data from all the previous tokens. Therefore, in our spam 
classification task, we focus on this last token during the fine-tuning process.

'''

print("last output token: ", output[:, -1, :])  # the : is don't care or for-all, the -1 indicates last..

'''
Last output token: tensor([[-3.5983,  3.9902]])

Next: we convert this last token to class prediction labels
(we use softmax and argmax as follows)
'''

probas = torch.softmax(output[:, -1, :], dim=-1)
label = torch.argmax(probas)
print("Class label:", label.item())

'''
Using the softmax function here is optional because the largest outputs directly correspond to the highest 
probability scores. Hence, we can simplify the code without using softmax:
'''

logits = output[:, -1, :]
label = torch.argmax(logits)
print("Class label:", label.item())

'''
Both of the above return the class label 1 (spam)

This concept can also be used to compute the classification accuracy, which measures the percentage of correct 
predictions across a dataset.

To determine the classification accuracy, we apply the argmax-based prediction code to all examples in the dataset 
and calculate the proportion of correct predictions by defining a calc_accuracy_loader function, like so:
'''

def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)


            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]    # last output token
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (
                (predicted_labels == target_batch).sum().item()
            )

        else:
            break
    return correct_predictions / num_examples

import spam_dataloader
from spam_dataloader import train_loader, val_loader, test_loader

# quick check..
print(f"{len(train_loader)} training batches")
print(f"{len(val_loader)} validation batches")
print(f"{len(test_loader)} test batches")

'''
Let's use the above function to determine the classification accuracies from a bunch
of 10 batches from the 3 datasets, for efficiency:
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

torch.manual_seed(123)
train_accuracy = calc_accuracy_loader(
    train_loader, model, device, num_batches=10
)
val_accuracy = calc_accuracy_loader(
    val_loader, model, device, num_batches=10
)
test_accuracy = calc_accuracy_loader(
    test_loader, model, device, num_batches=10
)

print(f"Training accuracy: {train_accuracy*100:.2f}%")
print(f"Validation accuracy: {val_accuracy*100:.2f}%")
print(f"Test accuracy: {test_accuracy*100:.2f}%")

'''
Training accuracy: 46.25%
Validation accuracy: 45.00%
Test accuracy: 48.75%

we get close to random probabities (50%ish) and this is
because we haven't fine tuned the model for classification
yet

the next step prior to training would be to figure out
how we calculate the loss
'''

'''
Because classification accuracy is not a differentiable function, 
we use cross-entropy loss as a proxy to maximize accuracy. Accordingly, 
the calc_loss_batch function remains the same, with one adjustment: 
we focus on optimizing only the last token, model(input_batch)[:, -1, :], 
rather than all tokens, model(input_batch):
'''

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

# Similar to calculating the training accuracy, we now compute the initial loss for each data set:

with torch.no_grad():
    train_loss = calc_loss_loader(
        train_loader, model, device, num_batches=5
    )
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
    test_loss = calc_loss_loader(test_loader, model, device, num_batches=5)

print(f"Training loss: {train_loss:.3f}")
print(f"Validation loss: {val_loss:.3f}")
print(f"Test loss: {test_loss:.3f}")

# now that we have prepared everything, we fine tune the demosthenes model:

'''
The training function implementing the concepts shown in images/finetune.png 
also closely mirrors the train_model_simple function used for pretraining the model. 

The only two distinctions are that we now track the number of training examples seen 
(examples_seen) instead of the number of tokens, and we calculate the accuracy after 
each epoch instead of printing a sample text.
'''

def train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs, eval_freq, eval_iter):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.shape[0]
            global_step += 1


            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}"
                )


        train_accuracy = calc_accuracy_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_accuracy = calc_accuracy_loader(
            val_loader, model, device, num_batches=eval_iter
        )

        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen

# The evaluate_model function is identical to the one we used for pretraining:

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss

'''
Next, we initialize the optimizer, set the number of training epochs, and initiate 
the training using the train_classifier_simple function. The training takes about 6 
minutes on an M3 MacBook Air laptop computer and less than half a minute on a V100a
a or A100 GPU: (Here I note my local time on my intel mac)
'''

import time

start_time = time.time()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
num_epochs = 5

train_losses, val_losses, train_accs, val_accs, examples_seen = \
    train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=50,
        eval_iter=5
    )

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

# next, I plot the classification losses across the training process/dataset:

import matplotlib.pyplot as plt

def plot_values(
        epochs_seen, examples_seen, train_values, val_values,
        label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))


    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(
        epochs_seen, val_values, linestyle="-.",
        label=f"Validation {label}"
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()


    ax2 = ax1.twiny()
    ax2.plot(examples_seen, train_values, alpha=0)
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()
    plt.savefig(f"{label}-plot.pdf")
    plt.show()

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))

plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)

#Using the same plot_values function, let’s now plot the classification accuracies:

epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))

plot_values(
    epochs_tensor, examples_seen_tensor, train_accs, val_accs,
    label="accuracy"
)

'''
Now we must calculate the performance metrics for the training, validation, 
and test sets across the entire dataset by running the following code, 
this time without defining the eval_iter value:
'''

train_accuracy = calc_accuracy_loader(train_loader, model, device)
val_accuracy = calc_accuracy_loader(val_loader, model, device)
test_accuracy = calc_accuracy_loader(test_loader, model, device)

print(f"Training accuracy: {train_accuracy*100:.2f}%")
print(f"Validation accuracy: {val_accuracy*100:.2f}%")
print(f"Test accuracy: {test_accuracy*100:.2f}%")

torch.save(model.state_dict(), "review_classifier.pth")
print("model saved!\n")

