Here I will augment the basic pretraining loop with:
1. Linear learning rate warmup
2. Learning rate cosine decay
3. Gradient clipping

Will do individual examples of the training loop with these characteristics for LR and gradients
(minus the actual calc_batch_loss (backprop) and gradient adjustments) and then incorporate all
three into the training loop)

Later can add inference optimizations as well, e.g. kv cache, other types of attention mechanisms etc.


