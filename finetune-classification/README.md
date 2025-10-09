
Fine tune my LLM now on specific tasks (here: classification)

My code for finetuning demosthenes llm (auto complete with GP2 arch trained by hand - see pretraining)
The finetuning is for classification (spam, sentiment recognition etc)

The idea is to understand and do modifications to an already trained LLM by modifying the endpoints and then applying
additional training to the weights to achieve classification !

In pretraining, I first trained the llm on a single story (the verdict), then I trained it on gutenberg book data 
(about 17G of text) to create xM params model, and finally I loaded open source weights (GPT2 from openAI) onto 
demosthenes to compare and contrast the performance. I also have inference / chatbots from each of these versions 
working. (All of this done on a mac)

Demosthenes llm is my llm from scratch.

In classification fine-tuning,(follownig classical machine learning) the model is trained to recognize a specific set 
of class labels, such as “spam” and “not spam.” Examples of classification tasks extend beyond LLMs and email filtering: 
they include identifying different species of plants from images; categorizing news articles into topics like sports, 
politics, and technology; and distinguishing between benign and malignant tumors in medical imaging.

The key point is that a classification fine-tuned model is restricted to predicting classes it has encountered during its 
training. For instance, it can determine whether something is “spam” or “not spam,” but it can’t say anything else about 
the input text.

Stages for classification finetuning:

Stage 1:
1. Download the dataset (the one used for training to recognize and classify labels)
2. Preprocess the dataset
3. Create data loaders

Stage 2:
4. Initialize the model
5. Load pretrained weights (or choose the gutenberg version, or any other)
6. Modify the model to enable fine tuning (the interesting part)
7. Implement evaluation utilities

Stage 3:
8. Fine-tune the model
9. Evaluate the fine-tuned model
10. Use model on new data (inference)

--> order of perusal

spam-dataset.py  (this generates a bunch of SMS data and ancillary files)
spam-dataloader.py (this creates data loaders specific for the SMS dataset, as a precursor to the classification based training)
spam-datasetclass.py (this defines the dataset class that I use to instantiate the loaders, used by spam-dataloader.py)
model.py (code for generating the model and modifying its final output layer to make it more suitable for classification fine tuning)



