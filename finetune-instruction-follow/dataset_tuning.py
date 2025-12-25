'''
Construct the training batches and prepare everything
with the data until the point we need to prepare the 
model and fine tune it (this comes after download_dataset.py
and stylize_prompts.py)

Before we move on to setting up the PyTorch data loaders 
let’s divide the dataset into training, validation, and test sets 
analogous to what we have done with the spam classification dataset. 

'''

from download_dataset import download_and_load_file

file_path = "instruction-data.json"
url =  (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)

data = download_and_load_file(file_path, url)
print("Len of data: ", len(data))

# lets partition training 85%, test 10% and validation as 5%
train_portion = int(len(data) * 0.85)
test_portion  = int(len(data) * 0.1)
val_portion   = len(data) - train_portion - test_portion

train_data    = data[ : train_portion]
test_data     = data[train_portion : train_portion + test_portion]
val_data      = data[train_portion+test_portion : ]

print("Training set length:", len(train_data))
print("Test set length:", len(test_data))
print("Validation set length:", len(val_data))

'''
Next, we will create training batches. 

Previously the training batches were created automatically by the 
PyTorch DataLoader class, which employs a default collate function 
to combine lists of samples into batches. A collate function is 
responsible for taking a list of individual data samples and merging them 
into a single batch that can be processed efficiently by the model during training.

However, the batching process for instruction fine-tuning is a bit more involved 
and requires us to create our own custom collate function that we will later plug 
into the DataLoader. We implement this custom collate function to handle the specific 
requirements and formatting of our instruction fine-tuning dataset.

(see images/custom-collation-training-batches.png)

Let's do this in several steps. First, code an InstructionDataset class that 
applies 'format_input" (stylize_prompts) and "pretokenizes" all inputs in the
dataset, similar to SpamDataset. This is implemented in the __init__ constructor
of the following class:

'''

import torch
from torch.utils.data import Dataset
from stylize_prompts import format_input

# first we implement steps 2.1 and 2.2 from
# images/customer-collation-training-batches.png

class InstructionDataset(Dataset):

    def __init__(self, data, tokenizer):

        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)

'''

Similar to the approach for classification fine tuning, we will accelerate
training by collecting multiple traininig samples in a batch, which means
we have to pad inputs to a similar length. Similar to classification ft, we 
will use <|endoftext|> token as a padding token.

But instead of appending this marker to the input tokens, we will directly append
it to the pretokenized inputs. We can use the tokenizer's encode method on an
<|endoftext|> marker to remind us which tokenID we should use:

'''

import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))
# this prints [50256]

'''
Moving on to step 2.3 of the process (see images/collate_within_batches.png), 
we adopt a more sophisticated approach by developing a custom collate function 
that we can pass to the data loader. 

This custom collate function pads the training examples in each batch to the same 
length while allowing different batches to have different lengths.

This approach minimizes unnecessary padding by only extending sequences to match 
the longest one in EACH batch, NOT the whole dataset.
'''

def custom_collate_draft_1(
    batch,    # batch allows different padding equivalencies amongst batches
    pad_token_id=50256,
    device="cpu"
):

    # calc max len for this batch:
    batch_max_length = max(len(item)+1 for item in batch) 

    inputs_lst = []

    # pad and prepare inputs:
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = (
            new_item + [pad_token_id] * 
            (batch_max_length - len(new_item))
        )

        # remove extra padded token added earlier
        inputs = torch.tensor(padded[:-1])
        inputs_lst.append(inputs)

    # convert list of inputs into tensor and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    return inputs_tensor

'''

Above was for INPUT batches, but we also need to create OUTPUT
batches that stand for target tokens. Initially they will be right
shifted by one, as that is how LLMs/attention mechanisms work. The 
training/backprop will add weights that will then actually strengthen
the model to predict the next token. Shifting right by one is CORRECT
because in a sense these are already the generated outputs, and these
sentences are the "correct" representations of an output from the model
already (these are ideal outputs in a sense)

draft_2 version adds target tokens to target batches, using the same
padding strategy as described above, for draft_1:

'''

def custom_collate_draft_2(
    batch,
    pad_token_id=50256,
    device="cpu"
):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = (
            new_item + [pad_token_id] * 
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:]) # target shifts by 1 to left (its the "next" of sequence)

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor

'''

In the next step, I assign a -100 placeholder value to all padding tokens, 
as highlighted in minus-100-token.png. This special value allows me to exclude 
these padding tokens from contributing to the training loss calculation, 
ensuring that only meaningful data influences model learning. (-100 because this
is an "ignore-token" option during loss calculations using cross_entropy in pytorch):

"cross_entropy(..., ignore_index=-100)" 

But along with the -100 tokens, I still keep the first end of marker 50256 so that 
the training can remember where a logical sentence ends (and distinguish them from
pads that I'm "masking" with -100 so as not to convolute the training (no pun intended))

i.e. retaining the last eom allows the LLM to learn when to generate an end-of-text token in 
response to instructions, which I use as an indicator that the generated response is complete
(see retain_last_endoftext.png)

In the following fn I modify my custom collate function to replace tokens of ID 50256 
with -100 in the target lists. Additionally I introduce an allowed_ max_length parameter 
to optionally limit the length of the samples. 

This adjustment will be useful if I plan to work with my own datasets that exceed the 
1024-token context size supported by the GPT-2 model in demosthenes.

'''

def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):

    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

'''
see test_loss_patterns.py : we choose not to apply masks
to instruction and inputs (the -100) because the paper in
images/instruction_tuning_w_loss proves that there is more
benefit to the LLM response in not applying masks to input/instruction
(but should check and confirm otherwise)
'''

'''
Now, onto DataLoaders. Will reuse the code above to basically plugin
both InstructionDataset objects and the custom_collate_fn into the 
Pytorch data loaders. The loaders will automatically shuffle and organize
batches for the LLM fine-tuning process.
'''

'''
To reuse the chosen device setting in custom_collate_fn when we plug it into 
the PyTorch DataLoader class, we use the partial function from Python’s functools 
standard library to create a new version of the function with the device argument 
prefilled. Additionally, we set the allowed_max_length to 1024, which truncates 
the data to the maximum context length supported by the GPT-2 model, which we will fine-tune later:
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# if torch.backends.mps.is_available():
#     device = torch.device("mps")"      

print("Device:", device)

from functools import partial

customized_collate_fn = partial(
    custom_collate_fn,
    device=device,
    allowed_max_length=1024
)

'''
Next, we can set up the data loaders as we did previously, but this time, we will 
use our custom collate function for the batching process.
'''

from torch.utils.data import DataLoader

num_workers = 0
#batch_size = 8
batch_size = 1 #ubuntu machines have only 16GB RAM, crash at batch_size=8

torch.manual_seed(123)

train_dataset = InstructionDataset(train_data, tokenizer)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers
)

val_dataset = InstructionDataset(val_data, tokenizer)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

test_dataset = InstructionDataset(test_data, tokenizer)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

