"""
Advanced agent dataset with multi-step reasoning and tool chaining
"""

import pandas as pd
import json

def generate_advanced_dataset():
    """Generate dataset for advanced agent capabilities"""
    
    # Expanded function set
    functions = {
        "get_weather": "Get weather information",
        "send_email": "Send email messages", 
        "search_web": "Search for information",
        "calculate": "Perform calculations",
        "set_reminder": "Set reminders",
        "read_file": "Read file contents",
        "write_file": "Write to files",
        "run_code": "Execute code",
        "multi_step": "Multi-step reasoning required",
        "no_function": "No function needed"
    }
    
    # Advanced examples with reasoning chains
    examples = [
        # Single function calls
        ("What's the weather in Tokyo?", "get_weather"),
        ("Send email to boss about project update", "send_email"),
        ("Search for machine learning papers", "search_web"),
        ("Calculate 156 * 23", "calculate"),
        ("Remind me about meeting at 2pm", "set_reminder"),
        ("Read the config.json file", "read_file"),
        ("Write results to output.txt", "write_file"),
        ("Run the data analysis script", "run_code"),
        
        # Multi-step reasoning
        ("Check weather and email the forecast to team", "multi_step"),
        ("Calculate budget and save to spreadsheet", "multi_step"),
        ("Search for tutorials and create study plan", "multi_step"),
        
        # Conversational
        ("Hello there", "no_function"),
        ("Thank you", "no_function"),
        ("How are you?", "no_function"),
    ]
    
    # Create label mapping
    func_to_label = {func: idx for idx, func in enumerate(functions.keys())}
    
    # Generate dataset
    data = []
    for text, func_name in examples:
        data.append({
            "Text": text,
            "Function": func_name,
            "Label": func_to_label[func_name]
        })
    
    df = pd.DataFrame(data)
    
    # Split datasets
    train_size = int(0.7 * len(df))
    val_size = int(0.15 * len(df))
    
    train_df = df[:train_size]
    val_df = df[train_size:train_size + val_size]
    test_df = df[train_size + val_size:]
    
    # Save
    train_df.to_csv("advanced_agent_train.csv", index=False)
    val_df.to_csv("advanced_agent_validation.csv", index=False)
    test_df.to_csv("advanced_agent_test.csv", index=False)
    
    with open("advanced_agent_functions.json", "w") as f:
        json.dump(functions, f, indent=2)
    
    print(f"Generated {len(df)} advanced agent examples")
    print(f"Functions: {len(functions)}")
    return df

if __name__ == "__main__":
    generate_advanced_dataset()
