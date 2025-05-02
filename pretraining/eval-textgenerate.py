import torch

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
model.eval()
print(model)

# add utility functions for text to tokenID conversion

import tiktoken
from generate_text_simple import generate_text_simple

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # unsqueeze(0) adds the batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model = model,
    idx = text_to_token_ids(start_context, tokenizer),
    max_new_tokens = 10,
    context_size = GPT_CONFIG_124M["context_length"]
)

print("token ids: ", token_ids)
print("Output text: \n", token_ids_to_text(token_ids, tokenizer))

'''
Let's setup framework for numerically assessing the text quality generated
during training by calculating the 'text generation loss'. (This loss metric
will serve as progress indicator of the training progress) 

First, let's map the text generation process explicity to the 5 step process
shown in textgenerate.png
'''

'''
step1: use vocabulary to map the input text to tokenIDs
Consider these 2 inputs, which have already been mapped to their token IDs from the vocab
'''

inputs = torch.tensor([[16833, 3626, 6100],   # ["every effort moves",
                       [40,    1107, 588]])   #  "I really like"]

'''
matching to these inputs we define targets containing the tokenIDs that we want the model
to produce (after all a loss function must know what the goal is). We assume a sliding window
model where the next word to the right is the word we'd like the model to output (also see
DataLoader - this shifting strategy is crucial for teaching the model to predict the next token
in a sequence.
'''

targets = torch.tensor([[3626, 6100, 345  ],  # [" effort moves you",
                        [1107, 588, 11311]])  #  " really like chocolate"]

'''
Step 2: Obtain vocab-size-dim (50257) probability row vector for each input token via the Softmax function
'''

with torch.no_grad():
    logits = model(inputs)

probas = torch.softmax(logits, dim=-1)

print("shape of softmax probability tensor: ", probas.shape)
print("value of the softmax tensor: \n", probas)
# these will be really small values as the embedding is 50257, not 7 ! 

# see https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html

'''
torch.Size([2, 3, 50257])
the first number 2 corresponds to the two examples (rows) in the inputs, also known as batch size
the second number 3 corresponds to the number of tokens in each row
the last number corresponds to the embedding dimensionality, which is determined by vocabulary size
''' 

'''
Step 3: Locate the index position with the highest probability value in each row vector (done via argmax fn)
Step 4: Obtain thus all predicted tokenIDs as the index positions with the highest probabilities
'''
token_ids = torch.argmax( probas, dim=-1, keepdim=True )
print("Token IDs:\n", token_ids)

'''
Token IDs:
 tensor([[[16657],
         [  339],
         [42826]],

        [[49906],
         [29669],
         [41751]]])
'''

'''
Convert token IDs back to text
'''
print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
print(f"Outputs batch 1:"
      f" {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")

'''
Targets batch 1:  effort moves you
Outputs batch 1:  Armed heNetflix

The model produces random text that is different from the target text because it has not been trained yet. We now want to 
evaluate the performance of the model’s generated text numerically via a loss. Not only is this useful for measuring the quality
of the generated text, but it’s also a building block for implementing the training function, which we will use to update the 
model’s weight to improve the generated text.

See loss-calc.png. The steps are:
1. Get the logits
2. Use softmax to get the probabilities
3. Get the target probabilities (ought to be)
4. Now get the log probabilities
5. Average the log probability
6. Take the negative average log probability

(6) is what needs to go to zero, via backpropagation applied during training
'''

# now for each of the 2 text inputs I will print out the initial softmax probabilities
# corresponding to the target tokens (the first 3 as found out by argmax, assuming these were trained correctly)

text_idx = 0
target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 1:", target_probas_1)

text_idx = 1
target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 2:", target_probas_2)

'''
Text 1: tensor([7.4541e-05, 3.1061e-05, 1.1563e-05])
Text 2: tensor([1.0337e-05, 5.6776e-05, 4.7559e-06])

Next, we will calculate the loss for the probability scores of the two example batches, target_probas_1 and target_probas_2. 
The main steps are illustrated in loss-calc.png. Since we already applied steps 1 to 3 to obtain target_probas_1 and target_ probas_2, 
we proceed with step 4, applying the logarithm to the probability scores:
'''

# get the log probabilities of the 2 sample batches (step4):

log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print("log of probabilities: ", log_probas)

'''
log of probabilities:  tensor([ -9.5042, -10.3796, -11.3677, -11.4798,  -9.7764, -12.2561])

Next, Step5 and Step6: we combine the above 3 + 3 values into a single value (negative average)
and this is the metric that we then need to get to zero via backprop used in the training loop

'''

avg_log_probas = torch.mean(log_probas)
print("average log: ", avg_log_probas)

'''
The goal is to get the average log probability as close to 0 as possible by updating the model’s weights as part of the training process. 
However, in deep learning, the common practice isn’t to push the average log probability up to 0 but rather to bring the negative average 
log probability down to 0. The negative average log probability is simply the average log probability multiplied by –1, which corresponds to step 6
'''

neg_avg_log_probas = avg_log_probas * -1
print("negative average log probability: ", neg_avg_log_probas)

'''
negative average log probability:  tensor(10.7940)
'''

'''
All of steps 1-6 can be achieved by using cross-entropy

At its core, the cross entropy loss is a popular measure in machine learning and deep learning that measures the difference between two 
probability distributions—typically, the true distribution of labels (here, tokens in a dataset) and the predicted distribution from a model 
(for instance, the token probabilities generated by an LLM).

In the context of machine learning and specifically in frameworks like PyTorch, the cross_entropy function computes this measure for discrete outcomes, 
which is similar to the negative average log probability of the target tokens given the model’s generated token probabilities, making the terms 
“cross entropy” and “negative average log probability” related and often used interchangeably in practice.

So let's check if we get the same results with cross_entropy

'''

print("Logits shape:", logits.shape)
print("Targets shape:", targets.shape)

'''
Logits shape: torch.Size([2, 3, 50257])
Targets shape: torch.Size([2, 3])

As we can see the logits shape has 3 dims: batchsize, number of tokens and vocab size.
Target shape has 2 dims: batchsize and number of tokens

For cross entropy loss function for pytorch, we want to flatten these tensors by combining
them over the batch dimension

'''

logits_flat = logits.flatten(0, 1)  # merge 0 and 1 dims, i.e. batchsize and number of tokens
targets_flat = targets.flatten()    # do the same except don't need to specify as there are 2 dims only

print(logits_flat.shape)
print(targets_flat.shape)

'''
torch.Size([6, 50257])
torch.Size([6])
'''

loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print("cross entropy loss: ", loss)

'''
cross entropy loss:  tensor(10.7940)
This is the same as the negative average log probability
'''

perplexity = torch.exp(loss)
print("perplexity: ", perplexity)

'''
perplexity:  tensor(48725.8203)

Perplexity is a measure often used alongside cross entropy loss to evaluate the performance of models in tasks 
like language modeling. It can provide a more interpretable way to understand the uncertainty of a model in predicting 
the next token in a sequence.

Perplexity measures how well the probability distribution predicted by the model matches the actual distribution of the 
words in the dataset. Similar to the loss, a lower perplexity indicates that the model predictions are closer to the actual 
distribution.

Perplexity can be calculated as perplexity = torch.exp(loss), which returns tensor(48725.8203) when applied to the previously 
calculated loss.

Perplexity is often considered more interpretable than the raw loss value because it signifies the effective vocabulary size about 
which the model is uncertain at each step. In the given example, this would translate to the model being unsure about which among 
48,725 tokens in the vocabulary to generate as the next token.

With this I show 2 types of losses : cross_entropy (negative average log of probability) and perplexity (e^ n.a.l.p) with perplexity
intuitively showing the data size that has uncertainty. The lower the perplexity the better.

'''





