'''
We need to import modules from attention folder. Temporarily add to python's search path
to do so. (tokenizers and is module is in ../ )
'''

import sys 
import os

# get current dir
current_dir = os.path.dirname(os.path.abspath(__file__))
print("current dir: ", current_dir, "\n")

# add the actual path where import module is
module_path = os.path.join(current_dir, '../attention/')
print("module path to be used for import: ", module_path, "\n")

# add new path to sys.path (what python uses to search imported modules)
sys.path.append(module_path)

import MultiHeadAttention
