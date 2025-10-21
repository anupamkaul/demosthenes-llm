"""
Agent-specific model adaptation of Demosthenes LLM
Based on your existing classification fine-tuning approach
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../pretraining/preloaded_weights/openai/scripts'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../llm-infra/'))

import torch
import tiktoken
from torch.utils.data import DataLoader

from gpt_download import download_and_load_gpt2
from load_wts_to_gpt import load_weights_into_gpt
from GPTModel import GPTModel
from agent_datasetclass import AgentDataset

# Configuration
CHOOSE_MODEL = "gpt2-small (124M)"
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True
}
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
}
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

def setup_agent_model():
    """Setup model for agent function calling"""
    
    # Load pretrained weights
    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
    
    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval()
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace output head for 6 agent functions
    num_functions = 6  # get_weather, send_email, search_web, calculate, set_reminder, no_function
    model.out_head = torch.nn.Linear(
        in_features=BASE_CONFIG["emb_dim"],
        out_features=num_functions
    )
    
    # Unfreeze last transformer block and final norm
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True
    
    return model

def predict_function(model, text, tokenizer):
    """Predict which function to call for given text"""
    
    model.eval()
    inputs = tokenizer.encode(text)
    inputs = torch.tensor(inputs).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(inputs)[:, -1, :]  # Last token
    
    predicted_function_id = torch.argmax(logits, dim=-1).item()
    
    # Function mapping
    functions = ["get_weather", "send_email", "search_web", "calculate", "set_reminder", "no_function"]
    return functions[predicted_function_id], predicted_function_id

def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    """Calculate accuracy on data loader"""
    
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
                logits = model(input_batch)[:, -1, :]
            predicted_labels = torch.argmax(logits, dim=-1)
            
            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
            
    return correct_predictions / num_examples

if __name__ == "__main__":
    # Setup
    tokenizer = tiktoken.get_encoding("gpt2")
    model = setup_agent_model()
    
    # Test function prediction
    test_texts = [
        "What's the weather in Boston?",
        "Send email to Alice",
        "Hello there"
    ]
    
    for text in test_texts:
        func, func_id = predict_function(model, text, tokenizer)
        print(f"Text: '{text}' -> Function: {func} (ID: {func_id})")
