"""
Function Calling / Tool Use
LLM that can call functions to perform actions.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import re
import math
from datetime import datetime

class FunctionCallingLLM:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """Initialize the function calling LLM"""
        print("Loading function calling model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        
        if not torch.cuda.is_available():
            self.model = self.model.to("cpu")
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Available functions
        self.functions = {
            "calculator": {
                "description": "Perform mathematical calculations",
                "parameters": {
                    "expression": {"type": "string", "description": "Mathematical expression to evaluate"}
                }
            },
            "get_time": {
                "description": "Get current date and time",
                "parameters": {}
            },
            "text_processor": {
                "description": "Process text (uppercase, lowercase, reverse, count words)",
                "parameters": {
                    "text": {"type": "string", "description": "Text to process"},
                    "operation": {"type": "string", "description": "Operation: uppercase, lowercase, reverse, count"}
                }
            },
            "web_search": {
                "description": "Search the web (simulated)",
                "parameters": {
                    "query": {"type": "string", "description": "Search query"}
                }
            }
        }
        
        print("Function calling system ready!")
    
    def execute_function(self, function_name: str, parameters: dict) -> str:
        """Execute a function and return result"""
        if function_name == "calculator":
            try:
                # Safe evaluation of mathematical expressions
                expression = parameters.get("expression", "")
                # Remove dangerous functions
                allowed_chars = set("0123456789+-*/()., ")
                if all(c in allowed_chars for c in expression):
                    result = eval(expression)
                    return f"Calculation result: {result}"
                else:
                    return "Error: Invalid expression"
            except Exception as e:
                return f"Error: {str(e)}"
        
        elif function_name == "get_time":
            now = datetime.now()
            return f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}"
        
        elif function_name == "text_processor":
            text = parameters.get("text", "")
            operation = parameters.get("operation", "").lower()
            
            if operation == "uppercase":
                return text.upper()
            elif operation == "lowercase":
                return text.lower()
            elif operation == "reverse":
                return text[::-1]
            elif operation == "count":
                return f"Word count: {len(text.split())}"
            else:
                return "Error: Invalid operation"
        
        elif function_name == "web_search":
            query = parameters.get("query", "")
            # Simulated web search
            return f"Search results for '{query}': [This is a simulated search. In production, integrate with a real search API.]"
        
        else:
            return f"Error: Unknown function '{function_name}'"
    
    def get_functions_description(self) -> str:
        """Get description of available functions"""
        desc = "Available functions:\n\n"
        for func_name, func_info in self.functions.items():
            desc += f"Function: {func_name}\n"
            desc += f"Description: {func_info['description']}\n"
            desc += "Parameters:\n"
            for param_name, param_info in func_info.get("parameters", {}).items():
                desc += f"  - {param_name}: {param_info['description']}\n"
            desc += "\n"
        return desc
    
    def detect_function_call(self, text: str) -> dict:
        """Detect if the text contains a function call"""
        # Simple pattern matching (in production, use structured outputs)
        patterns = [
            r'call\s+(\w+)\s+with\s+({.*?})',
            r'function:\s*(\w+)\s*params:\s*({.*?})',
            r'(\w+)\(({.*?})\)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                func_name = match.group(1)
                params_str = match.group(2)
                try:
                    params = json.loads(params_str)
                    return {"function": func_name, "parameters": params}
                except:
                    # Try to parse manually
                    params = self._parse_parameters(params_str)
                    return {"function": func_name, "parameters": params}
        
        return None
    
    def _parse_parameters(self, params_str: str) -> dict:
        """Parse parameters from string"""
        params = {}
        # Simple key-value parsing
        pairs = re.findall(r'(\w+):\s*"([^"]+)"', params_str)
        for key, value in pairs:
            params[key] = value
        return params
    
    def process_with_functions(self, user_input: str) -> str:
        """Process user input and potentially call functions"""
        # Build prompt with function descriptions
        functions_desc = self.get_functions_description()
        
        prompt = f"""You are a helpful assistant that can call functions to help users.

{functions_desc}

When the user asks for something that requires a function, respond in this format:
FUNCTION_CALL: function_name
PARAMETERS: {{"param1": "value1", "param2": "value2"}}

User: {user_input}
Assistant:"""
        
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + 100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "Assistant:" in response:
            assistant_response = response.split("Assistant:")[-1].strip()
        else:
            assistant_response = response[len(prompt):].strip()
        
        # Check for function calls
        if "FUNCTION_CALL:" in assistant_response:
            func_call = self._extract_function_call(assistant_response)
            if func_call:
                function_name = func_call["function"]
                parameters = func_call["parameters"]
                
                if function_name in self.functions:
                    # Execute function
                    func_result = self.execute_function(function_name, parameters)
                    return f"I called {function_name} and got: {func_result}"
                else:
                    return f"Error: Unknown function '{function_name}'"
        
        # Simple keyword-based function detection (fallback)
        user_lower = user_input.lower()
        
        # Calculator
        if any(word in user_lower for word in ["calculate", "compute", "math", "add", "multiply"]):
            # Extract numbers and operation
            numbers = re.findall(r'\d+', user_input)
            if len(numbers) >= 2:
                if "add" in user_lower or "+" in user_input:
                    result = sum(int(n) for n in numbers)
                    return f"Calculation result: {result}"
                elif "multiply" in user_lower or "*" in user_input or "times" in user_lower:
                    result = math.prod(int(n) for n in numbers)
                    return f"Calculation result: {result}"
        
        # Time
        if "time" in user_lower or "date" in user_lower:
            return self.execute_function("get_time", {})
        
        # Text processing
        if any(word in user_lower for word in ["uppercase", "lowercase", "reverse", "count words"]):
            # Extract text and operation
            operation = "uppercase" if "uppercase" in user_lower else "lowercase" if "lowercase" in user_lower else "reverse" if "reverse" in user_lower else "count"
            # For demo, use the whole input as text
            return self.execute_function("text_processor", {"text": user_input, "operation": operation})
        
        return assistant_response
    
    def _extract_function_call(self, text: str) -> dict:
        """Extract function call from text"""
        if "FUNCTION_CALL:" in text and "PARAMETERS:" in text:
            func_line = text.split("FUNCTION_CALL:")[1].split("\n")[0].strip()
            params_line = text.split("PARAMETERS:")[1].split("\n")[0].strip()
            
            try:
                params = json.loads(params_line)
                return {"function": func_line, "parameters": params}
            except:
                params = self._parse_parameters(params_line)
                return {"function": func_line, "parameters": params}
        
        return None

def main():
    llm = FunctionCallingLLM()
    
    print("\n" + "="*50)
    print("Function Calling LLM")
    print("="*50)
    print("\nAvailable functions:")
    for func_name in llm.functions.keys():
        print(f"  - {func_name}")
    print("\nExamples:")
    print("  - 'Calculate 25 * 37'")
    print("  - 'What time is it?'")
    print("  - 'Make this uppercase: hello world'")
    print("  - 'Search for Python tutorials'")
    print("\nType 'quit' to exit\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        print("Assistant: ", end="", flush=True)
        response = llm.process_with_functions(user_input)
        print(response + "\n")

if __name__ == "__main__":
    main()


