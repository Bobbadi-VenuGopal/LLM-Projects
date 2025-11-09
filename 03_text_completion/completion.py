"""
Text Completion / Autocomplete System
Complete sentences and generate text suggestions.
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class TextCompleter:
    def __init__(self, model_name="gpt2"):
        """Initialize the text completer"""
        print(f"Loading {model_name} model...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
            print("Using GPU acceleration")
        else:
            print("Using CPU")
        
        # Set pad token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Model loaded!")
    
    def complete_text(self, prefix, num_completions=3, max_new_tokens=50, temperature=0.8):
        """
        Generate multiple completions for a given prefix
        
        Args:
            prefix: The starting text
            num_completions: Number of different completions to generate
            max_new_tokens: Maximum number of tokens to generate
            temperature: Controls randomness
        """
        # Tokenize prefix
        inputs = self.tokenizer.encode(prefix, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        completions = []
        
        # Generate multiple completions
        with torch.no_grad():
            for _ in range(num_completions):
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1,
                )
                
                # Decode the completion
                completion = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                completions.append(completion)
        
        return completions
    
    def complete_single(self, prefix, max_new_tokens=50, temperature=0.7):
        """Generate a single completion"""
        inputs = self.tokenizer.encode(prefix, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        completion = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return completion
    
    def get_next_words(self, prefix, num_words=5, temperature=0.8):
        """Get the most likely next words"""
        inputs = self.tokenizer.encode(prefix, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        with torch.no_grad():
            outputs = self.model(inputs)
            logits = outputs.logits[0, -1, :]
            
            # Get top predictions
            top_indices = torch.topk(logits, num_words).indices
            top_words = []
            
            for idx in top_indices:
                word = self.tokenizer.decode([idx.item()])
                probability = torch.softmax(logits, dim=0)[idx].item()
                top_words.append((word.strip(), probability))
        
        return top_words

def main():
    # Initialize completer
    completer = TextCompleter()
    
    print("\n" + "="*50)
    print("Text Completion System")
    print("="*50)
    print("\nCommands:")
    print("  - Type 'multi' for multiple completions")
    print("  - Type 'next' to see next word suggestions")
    print("  - Type 'quit' to exit")
    print("\n")
    
    while True:
        prefix = input("Enter text to complete: ")
        
        if prefix.lower() == 'quit':
            print("Goodbye!")
            break
        
        if not prefix.strip():
            continue
        
        mode = input("Mode (single/multi/next) [single]: ").strip().lower() or "single"
        
        if mode == "multi":
            print("\nGenerating multiple completions...")
            completions = completer.complete_text(prefix, num_completions=3)
            print("\nCompletions:")
            for i, completion in enumerate(completions, 1):
                print(f"\n{i}. {completion}")
        
        elif mode == "next":
            print("\nTop next words:")
            next_words = completer.get_next_words(prefix, num_words=5)
            for word, prob in next_words:
                print(f"  {word}: {prob:.2%}")
        
        else:
            print("\nGenerating completion...")
            completion = completer.complete_single(prefix)
            print(f"\nCompletion: {completion}")
        
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    main()


