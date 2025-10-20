"""
Agent-specific dataset generation for fine-tuning Demosthenes LLM
Adapts the spam classification approach for agent function calling
"""

import pandas as pd
import json

def generate_agent_dataset():
    """Generate synthetic agent training data"""
    
    # Agent function definitions
    functions = {
        "get_weather": "Get current weather for a location",
        "send_email": "Send an email to a recipient", 
        "search_web": "Search the web for information",
        "calculate": "Perform mathematical calculations",
        "set_reminder": "Set a reminder for a specific time",
        "no_function": "No function call needed"
    }
    
    # Training examples: (user_input, function_name)
    examples = [
        ("What's the weather like in San Francisco?", "get_weather"),
        ("Send an email to John about the meeting", "send_email"),
        ("Search for Python tutorials", "search_web"),
        ("What is 25 * 47?", "calculate"),
        ("Remind me to call mom at 3pm", "set_reminder"),
        ("Hello, how are you?", "no_function"),
        ("Can you check the temperature in New York?", "get_weather"),
        ("Email the report to Sarah", "send_email"),
        ("Look up information about machine learning", "search_web"),
        ("Calculate 150 divided by 6", "calculate"),
        ("Set a reminder for my dentist appointment tomorrow", "set_reminder"),
        ("Thanks for your help", "no_function"),
    ]
    
    # Create function label mapping
    func_to_label = {func: idx for idx, func in enumerate(functions.keys())}
    
    # Generate dataset
    data = []
    for text, func_name in examples:
        data.append({
            "Text": text,
            "Function": func_name,
            "Label": func_to_label[func_name]
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    
    # Split into train/val/test
    train_size = int(0.7 * len(df))
    val_size = int(0.15 * len(df))
    
    train_df = df[:train_size]
    val_df = df[train_size:train_size + val_size]
    test_df = df[train_size + val_size:]
    
    # Save datasets
    train_df.to_csv("agent_train.csv", index=False)
    val_df.to_csv("agent_validation.csv", index=False)
    test_df.to_csv("agent_test.csv", index=False)
    
    # Save function mapping
    with open("agent_functions.json", "w") as f:
        json.dump(functions, f, indent=2)
    
    print(f"Generated {len(df)} agent training examples")
    print(f"Functions: {list(functions.keys())}")
    return df

if __name__ == "__main__":
    generate_agent_dataset()
