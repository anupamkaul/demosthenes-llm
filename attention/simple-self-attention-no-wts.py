'''
Self-attention is a mechanism that allows eaxch position in the INPUT sequence to consider the relevancy of, or "attend to" all other
positions in the same INPUT sequence when computing the representation of the sequence. 

It asessess and learns the relationships and dependencies between various parts of the INPUT itself, such as words in a sentence, or pixels in an emage (or later - contextual relationships between concepts as part of a larger conversation - TODO ! )

(Follows from Bahdanau attention mechanism that had RNN decoder (output) access all states of the encoder (input) sequence. RNNs focused on relationships between elements of two different sequences, such as in sequence-to-sequence models where the attention might be between an input sequence and and output sequence. Self-attention, and hence transformers, eliminates RNNs entirely and establishes a new order of encode/decode with self-attention)

'''
