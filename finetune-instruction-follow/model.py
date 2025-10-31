'''

This is where I prepare the model for instruction fine-tuning training
for demosthenes. I'll use all the data preparation from dataset_tuning.py
For the fine tuning process, I will reuse the loss calculation code that 
I wrote up in the "pretraining" folder. Heck, I will even use the same
train_model_simple training code that I used in pretraining demosthenes !

'''

import sys, os
sys.path.append( os.path.join( os.path.dirname(os.path.abspath(__file__)),  '../pretraining/') )

from training import train_model_simple
from utils_loss import calc_loss_loader

# note: "training.py" in ../pretraining has not been well written
# pulls in garbage code from ltv that pulls in verdict.txt
# needs either a re-write or a re-import of train_model_simple into its own container





