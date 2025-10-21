"""
Multi-step agent with reasoning chains
"""

import torch
import tiktoken
exec(open('agent-executor.py').read())

class MultiStepAgent:
    
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.executor = AgentExecutor()
        self.functions = ["get_weather", "send_email", "search_web", "calculate", 
                         "set_reminder", "read_file", "write_file", "run_code", 
                         "multi_step", "no_function"]
    
    def decompose_task(self, user_input):
        """Break down complex tasks into steps"""
        
        # Simple rule-based decomposition
        steps = []
        
        if "weather" in user_input.lower() and "email" in user_input.lower():
            steps = [
                ("get_weather", "get weather information"),
                ("send_email", "email the weather forecast")
            ]
        elif "calculate" in user_input.lower() and ("save" in user_input.lower() or "write" in user_input.lower()):
            steps = [
                ("calculate", "perform calculation"),
                ("write_file", "save results to file")
            ]
        elif "search" in user_input.lower() and "plan" in user_input.lower():
            steps = [
                ("search_web", "search for information"),
                ("write_file", "create study plan")
            ]
        else:
            # Single step task
            steps = [("no_function", "single step task")]
        
        return steps
    
    def execute_steps(self, user_input, steps):
        """Execute a sequence of steps"""
        
        results = []
        
        for step_func, step_desc in steps:
            print(f"Executing: {step_desc}")
            
            if step_func == "get_weather":
                result = self.executor._get_weather(user_input)
            elif step_func == "send_email":
                result = self.executor._send_email(user_input)
            elif step_func == "calculate":
                result = self.executor._calculate(user_input)
            elif step_func == "write_file":
                result = "Results saved to file"
            elif step_func == "search_web":
                result = self.executor._search_web(user_input)
            else:
                result = "Step completed"
            
            results.append(result)
            print(f"Result: {result}")
        
        return results
    
    def process(self, user_input):
        """Process user input with multi-step reasoning"""
        
        print(f"Processing: {user_input}")
        
        # Decompose task
        steps = self.decompose_task(user_input)
        
        if len(steps) > 1:
            print(f"Multi-step task detected: {len(steps)} steps")
            results = self.execute_steps(user_input, steps)
            return f"Completed {len(steps)} steps: " + " -> ".join(results)
        else:
            # Single step
            result = self.executor.execute("no_function", user_input)
            return result

# Test the multi-step agent
if __name__ == "__main__":
    agent = MultiStepAgent()
    
    test_cases = [
        "Check weather in Paris and email it to the team",
        "Calculate 25 * 30 and save to results.txt",
        "Search for Python tutorials and create a study plan",
        "Hello, how are you?"
    ]
    
    for test in test_cases:
        print(f"\n{'='*50}")
        result = agent.process(test)
        print(f"Final result: {result}")
        print(f"{'='*50}")
