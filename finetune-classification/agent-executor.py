"""
Agent function executor - executes the predicted functions
"""

import json
import re
from datetime import datetime

class AgentExecutor:
    
    def __init__(self):
        self.functions = {
            "get_weather": self._get_weather,
            "send_email": self._send_email,
            "search_web": self._search_web,
            "calculate": self._calculate,
            "set_reminder": self._set_reminder,
            "no_function": self._no_function
        }
    
    def execute(self, function_name, user_input):
        """Execute the predicted function with user input"""
        if function_name in self.functions:
            return self.functions[function_name](user_input)
        return "Unknown function"
    
    def _get_weather(self, text):
        # Extract location from text
        location = self._extract_location(text)
        return f"Weather in {location}: 72°F, sunny"
    
    def _send_email(self, text):
        # Extract recipient from text
        recipient = self._extract_recipient(text)
        return f"Email sent to {recipient}"
    
    def _search_web(self, text):
        # Extract search query
        query = self._extract_search_query(text)
        return f"Search results for: {query}"
    
    def _calculate(self, text):
        # Extract and evaluate math expression
        result = self._extract_and_calculate(text)
        return f"Result: {result}"
    
    def _set_reminder(self, text):
        # Extract reminder details
        reminder = self._extract_reminder(text)
        return f"Reminder set: {reminder}"
    
    def _no_function(self, text):
        return "I understand. How can I help you?"
    
    # Helper methods for extraction
    def _extract_location(self, text):
        # Simple pattern matching for locations
        words = text.split()
        for i, word in enumerate(words):
            if word.lower() in ["in", "for"]:
                if i + 1 < len(words):
                    return words[i + 1].strip("?.,")
        return "your location"
    
    def _extract_recipient(self, text):
        # Extract name after "to"
        words = text.split()
        for i, word in enumerate(words):
            if word.lower() == "to":
                if i + 1 < len(words):
                    return words[i + 1].strip(".,")
        return "recipient"
    
    def _extract_search_query(self, text):
        # Extract search terms
        stop_words = ["search", "for", "look", "up", "find"]
        words = [w for w in text.split() if w.lower() not in stop_words]
        return " ".join(words[:3])  # First 3 relevant words
    
    def _extract_and_calculate(self, text):
        # Extract and evaluate simple math
        import re
        # Find numbers and operators
        pattern = r'(\d+)\s*([+\-*/])\s*(\d+)'
        match = re.search(pattern, text)
        if match:
            num1, op, num2 = match.groups()
            try:
                return eval(f"{num1}{op}{num2}")
            except:
                return "calculation error"
        return "no calculation found"
    
    def _extract_reminder(self, text):
        # Simple reminder extraction
        words = text.split()
        reminder_start = -1
        for i, word in enumerate(words):
            if word.lower() in ["remind", "reminder"]:
                reminder_start = i
                break
        
        if reminder_start >= 0:
            return " ".join(words[reminder_start:])
        return "reminder"
