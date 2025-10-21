"""
Fine-tune Demosthenes LLM for agent function calling
Minimal training loop based on your classification approach
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tiktoken

from agent_datasetclass import AgentDataset
exec(open('agent_model.py').read())

def train_agent_model():
    """Fine-tune model for agent function calling"""
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Load datasets
    train_dataset = AgentDataset("agent_train.csv", tokenizer, max_length=256)
    val_dataset = AgentDataset("agent_validation.csv", tokenizer, max_length=256)
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    # Setup model
    model = setup_agent_model()
    model.to(device)
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    loss_fn = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    for epoch in range(5):  # Minimal epochs
        total_loss = 0
        
        for input_batch, target_batch in train_loader:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass - use last token for classification
            logits = model(input_batch)[:, -1, :]
            loss = loss_fn(logits, target_batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation accuracy
        val_accuracy = calc_accuracy_loader(val_loader, model, device)
        
        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Val Acc={val_accuracy*100:.2f}%")
    
    return model

if __name__ == "__main__":
    # Generate dataset first
    from agent_dataset import generate_agent_dataset
    generate_agent_dataset()
    
    # Train model
    trained_model = train_agent_model()
    
    # Save model
    torch.save(trained_model.state_dict(), "agent_model.pth")
    print("Agent model saved!")
