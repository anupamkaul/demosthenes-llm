"""
Complete agent system with memory, context, and advanced reasoning
"""

import torch
import tiktoken
import json
from datetime import datetime

exec(open('agent-executor.py').read())
exec(open('multi-step-agent.py').read())

class CompleteAgent:
    
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.executor = AgentExecutor()
        self.multi_step = MultiStepAgent()
        self.memory = []  # Conversation memory
        self.context = {}  # Current context
        
    def add_to_memory(self, user_input, response):
        """Add interaction to memory"""
        self.memory.append({
            "timestamp": datetime.now().isoformat(),
            "user": user_input,
            "agent": response
        })
        
        # Keep only last 10 interactions
        if len(self.memory) > 10:
            self.memory = self.memory[-10:]
    
    def update_context(self, user_input, function_called):
        """Update current context based on interaction"""
        if "weather" in user_input.lower():
            self.context["last_weather_query"] = user_input
        elif "email" in user_input.lower():
            self.context["last_email_action"] = user_input
        elif "calculate" in user_input.lower():
            self.context["last_calculation"] = user_input
    
    def get_contextual_response(self, user_input):
        """Generate response considering context and memory"""
        
        # Check for follow-up questions
        if user_input.lower() in ["again", "repeat", "same"]:
            if self.memory:
                last_user_input = self.memory[-1]["user"]
                return self.process_input(last_user_input)
        
        # Check for context references
        if "that" in user_input.lower() or "it" in user_input.lower():
            if "last_weather_query" in self.context:
                user_input = user_input.replace("that", self.context["last_weather_query"])
                user_input = user_input.replace("it", "the weather")
        
        return self.process_input(user_input)
    
    def process_input(self, user_input):
        """Main processing function"""
        
        # Determine if multi-step reasoning is needed
        multi_step_keywords = ["and", "then", "after", "also"]
        needs_multi_step = any(keyword in user_input.lower() for keyword in multi_step_keywords)
        
        if needs_multi_step:
            response = self.multi_step.process(user_input)
            function_called = "multi_step"
        else:
            # Single function call
            response = self.executor.execute("no_function", user_input)
            function_called = "single_step"
        
        # Update context and memory
        self.update_context(user_input, function_called)
        self.add_to_memory(user_input, response)
        
        return response
    
    def chat(self):
        """Interactive chat with advanced capabilities"""
        print("🤖 Advanced Agent Ready!")
        print("Features: Multi-step reasoning, Memory, Context awareness")
        print("Type 'memory' to see conversation history")
        print("Type 'context' to see current context")
        print("Type 'quit' to exit\n")
        
        while True:
            user_input = input("You: ")
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'memory':
                print("\n📚 Conversation Memory:")
                for i, mem in enumerate(self.memory[-5:], 1):
                    print(f"{i}. User: {mem['user']}")
                    print(f"   Agent: {mem['agent']}")
                continue
            elif user_input.lower() == 'context':
                print(f"\n🧠 Current Context: {self.context}")
                continue
            
            response = self.get_contextual_response(user_input)
            print(f"Agent: {response}")
    
    def save_session(self, filename="agent_session.json"):
        """Save current session"""
        session_data = {
            "memory": self.memory,
            "context": self.context,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filename, "w") as f:
            json.dump(session_data, f, indent=2)
        
        print(f"Session saved to {filename}")

if __name__ == "__main__":
    agent = CompleteAgent()
    
    # Demo mode
    print("🚀 Demo Mode - Testing Advanced Agent\n")
    
    test_inputs = [
        "What's the weather in London?",
        "Send that to my team",
        "Calculate 45 * 67 and save it to results",
        "Search for AI papers and create a reading list",
        "again"
    ]
    
    for test_input in test_inputs:
        print(f"You: {test_input}")
        response = agent.get_contextual_response(test_input)
        print(f"Agent: {response}\n")
    
    print("Starting interactive mode...")
    agent.chat()
