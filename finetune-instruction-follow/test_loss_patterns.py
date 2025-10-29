import torch

logits_1 = torch.tensor(
    [[-1.0, 1.0],
     [-0.5, 1.5]]
)

targets_1 = torch.tensor([0, 1]) # say these are correct token indices to generate

loss_1 = torch.nn.functional.cross_entropy(logits_1, targets_1)
print(loss_1)

# tensor(1.1269)

# add an additional logits_id, which should impact loss value
logits_2 = torch.tensor(
    [[-1.0, 1.0],
     [-0.5, 1.5],
     [-0.5, 1.5]] # the extra row
)
targets_2 = torch.tensor([0, 1, 1]) # the target correct values

loss_2 = torch.nn.functional.cross_entropy(logits_2, targets_2)
print(loss_2)

# tensor(0.7936)

# now check what happens with a -100 (should be nothing)
targets_3 = torch.tensor([0, 1, -100])
loss_3 = torch.nn.functional.cross_entropy(logits_2, targets_3)
print(loss_3)
print("loss_1 == loss_3:", loss_1 == loss_3)

# the existence of the extra row in logits_2 gets masked ! and make the 
# loss calculation actually behave like logits_1 (because of the -100 in the target
# which is an Ignore/mask) 

# tensor(1.1269)
# loss_1 == loss_3: tensor(True)

'''
The resulting loss on these three training examples is identical to the loss we calculated 
from the two training examples earlier. In other words, the cross entropy loss function ignored 
the third entry in the targets_3 vector, the token ID corresponding to -100. (

Interested readers can try to replace the -100 value with another token ID that is not 0 or 1; 
it will result in an error.)

So what’s so special about -100 that it’s ignored by the cross entropy loss? The default setting 
of the cross entropy function in PyTorch is cross_entropy(..., ignore_index=-100). This means that 
it ignores targets labeled with -100. We take advantage of this ignore_index to ignore the additional 
end-of-text (padding) tokens that we used to pad the training examples to have the same length in each 
batch. However, we want to keep one 50256 (end-of-text) token ID in the targets because it helps the 
LLM to learn to generate end-of-text tokens, which we can use as an indicator that a response is complete.

In addition to masking out padding tokens, it is also common to mask out the target token IDs that correspond 
to the instruction, as illustrated in images/mask_instruction_with_loss.png. By masking out the LLM’s target 
token IDs corresponding to the instruction, the cross entropy loss is only computed for the generated response 
target IDs. Thus, the model is trained to focus on generating accurate responses rather than memorizing instructions, 
which can help reduce overfitting.

'''
