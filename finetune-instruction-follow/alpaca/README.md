Notes:

As an option I can also evaluate instruction fine-tuning the
demosthenes-pretrained (un-finetuned) model to alpaca dataset

(https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json) 

The Alpaca dataset, by researchers at Stanford, is one of the earliest and most popular 
openly shared instruction datasets, consisting of 52,002 entries. As an alternative to 
the instruction-data.json file I can consider fine-tuning an LLM on this dataset.

This dataset contains 52,002 entries, which is approximately 50 times more than those I used
and most entries are longer. Thus I will use a GPU to conduct the training, which will accelerate 
the fine-tuning process. If I encounter out-of-memory errors, will reduce the batch_size from 8 
to 4, 2, or even 1. 

Lowering the allowed_max_length from 1,024 to 512 or 256 can also help manage memory problems.

