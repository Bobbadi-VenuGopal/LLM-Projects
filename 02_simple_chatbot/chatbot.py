"""
Simple Chatbot with Conversation History
Build a chatbot that maintains context across conversations.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class SimpleChatbot:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """Initialize the chatbot"""
        print("Loading chatbot model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        
        if not torch.cuda.is_available():
            self.model = self.model.to("cpu")
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Conversation history
        self.conversation_history = []
        self.max_history_length = 10  # Keep last 10 exchanges
        
        print("Chatbot ready!")
    
    def format_prompt(self, user_input, system_prompt="You are a helpful AI assistant."):
        """Format the conversation for the model"""
        # Build conversation context
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add recent history
        for exchange in self.conversation_history[-self.max_history_length:]:
            messages.append({"role": "user", "content": exchange["user"]})
            messages.append({"role": "assistant", "content": exchange["assistant"]})
        
        # Add current user input
        messages.append({"role": "user", "content": user_input})
        
        # Convert to prompt format
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        return prompt
    
    def generate_response(self, user_input, max_length=200, temperature=0.7):
        """Generate a response to user input"""
        # Format prompt with conversation history
        prompt = self.format_prompt(user_input)
        
        # Tokenize
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the new response (remove the prompt)
        response = full_response[len(prompt):].strip()
        
        # Clean up response
        if response.startswith("assistant"):
            response = response.replace("assistant", "").strip()
        
        # Update conversation history
        self.conversation_history.append({
            "user": user_input,
            "assistant": response
        })
        
        # Keep history manageable
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
        
        return response
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("Conversation history cleared!")

def main():
    # Initialize chatbot
    chatbot = SimpleChatbot()
    
    print("\n" + "="*50)
    print("Simple Chatbot")
    print("="*50)
    print("\nCommands:")
    print("  - Type 'clear' to clear conversation history")
    print("  - Type 'quit' to exit")
    print("\nStart chatting!\n")
    
    while True:
        # Get user input
        user_input = input("You: ")
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        if user_input.lower() == 'clear':
            chatbot.clear_history()
            continue
        
        if not user_input.strip():
            continue
        
        # Generate and display response
        print("Bot: ", end="", flush=True)
        response = chatbot.generate_response(user_input)
        print(response + "\n")

if __name__ == "__main__":
    main()


