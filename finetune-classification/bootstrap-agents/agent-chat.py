"""
Complete agent chat interface with function calling
"""

import torch
import tiktoken

from agent_executor import AgentExecutor
from agent_model import setup_agent_model

# Load the trained model and setup
def load_agent_model():
    model = setup_agent_model()
    
    # Load trained weights if available
    try:
        model.load_state_dict(torch.load("agent_model.pth"))
        print("Loaded trained agent model")
    except:
        print("Using untrained model - run agent-finetune.py first")
    
    return model

class AgentChat:
    
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.model = load_agent_model()
        self.executor = AgentExecutor()
        self.functions = ["get_weather", "send_email", "search_web", "calculate", "set_reminder", "no_function"]
    
    def predict_and_execute(self, user_input):
        """Predict function and execute it"""
        
        # Predict function
        inputs = self.tokenizer.encode(user_input)
        inputs = torch.tensor(inputs).unsqueeze(0)
        
        with torch.no_grad():
            logits = self.model(inputs)[:, -1, :]
        
        predicted_id = torch.argmax(logits, dim=-1).item()
        predicted_function = self.functions[predicted_id]

        # debug print
        print("predicted function: ", predicted_function)
        
        # Get confidence scores
        probabilities = torch.softmax(logits, dim=-1)
        confidence = probabilities[0][predicted_id].item()

        print("probabilities: ", probabilities, " confidence: ", confidence) 
        
        # Execute function
        result = self.executor.execute(predicted_function, user_input)
        
        return {
            "function": predicted_function,
            "confidence": confidence,
            "result": result
        }
    
    def chat(self):
        """Interactive chat loop"""
        print("Agent Chat Ready! Type 'quit' to exit.")
        
        while True:
            user_input = input("\nYou: ")
            
            if user_input.lower() == 'quit':
                break
            
            response = self.predict_and_execute(user_input)
            
            print(f"Function: {response['function']} ({response['confidence']:.2f})")
            print(f"Agent: {response['result']}")

if __name__ == "__main__":
    agent = AgentChat()
    agent.chat()
