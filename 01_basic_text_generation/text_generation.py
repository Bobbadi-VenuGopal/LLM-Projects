"""
Basic Text Generation with TinyLlama
Generate text from a prompt using a local LLM.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_model():
    """Load the TinyLlama model and tokenizer"""
    print("Loading TinyLlama model... (this may take a minute)")
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    
    if not torch.cuda.is_available():
        model = model.to("cpu")
    
    print("Model loaded successfully!")
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.7, top_p=0.9):
    """
    Generate text from a prompt
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input text prompt
        max_length: Maximum length of generated text
        temperature: Controls randomness (lower = more deterministic)
        top_p: Nucleus sampling parameter
    """
    # Tokenize the input
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode the output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def main():
    # Load model
    model, tokenizer = load_model()
    
    print("\n" + "="*50)
    print("Text Generation with TinyLlama")
    print("="*50)
    print("\nType 'quit' to exit\n")
    
    while True:
        # Get user input
        prompt = input("Enter your prompt: ")
        
        if prompt.lower() == 'quit':
            print("Goodbye!")
            break
        
        if not prompt.strip():
            print("Please enter a valid prompt.")
            continue
        
        print("\nGenerating text...")
        
        # Generate text
        generated = generate_text(model, tokenizer, prompt, max_length=150, temperature=0.7)
        
        # Display result
        print(f"\nGenerated text:\n{generated}\n")
        print("-"*50 + "\n")

if __name__ == "__main__":
    main()


