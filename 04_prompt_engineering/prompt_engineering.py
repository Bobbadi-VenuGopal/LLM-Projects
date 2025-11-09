"""
Prompt Engineering Techniques
Learn different ways to prompt LLMs for better results.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class PromptEngineer:
    def __init__(self, model_name="microsoft/Phi-2"):
        """Initialize the prompt engineer"""
        print(f"Loading {model_name} model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        if not torch.cuda.is_available():
            self.model = self.model.to("cpu")
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Model loaded!")
    
    def generate(self, prompt, max_length=200, temperature=0.7):
        """Generate text from a prompt"""
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        return response
    
    def zero_shot(self, question):
        """Zero-shot prompting - direct question"""
        prompt = f"Question: {question}\nAnswer:"
        return self.generate(prompt)
    
    def few_shot(self, examples, question):
        """Few-shot prompting - provide examples"""
        prompt = "Here are some examples:\n\n"
        for example in examples:
            prompt += f"{example}\n\n"
        prompt += f"Now answer this: {question}\nAnswer:"
        return self.generate(prompt)
    
    def chain_of_thought(self, problem):
        """Chain-of-thought prompting - step-by-step reasoning"""
        prompt = f"""Let's solve this problem step by step.

Problem: {problem}

Let's think through this:
1. First, I need to understand what's being asked.
2. Then, I'll break down the problem into smaller parts.
3. Finally, I'll solve each part and combine the results.

Solution:"""
        return self.generate(prompt, max_length=300)
    
    def role_based(self, role, task):
        """Role-based prompting - assign a role to the AI"""
        prompt = f"""You are an expert {role}. Your task is to {task}.

Please provide a detailed and professional response.

Response:"""
        return self.generate(prompt, max_length=250)
    
    def template_prompt(self, template_type, **kwargs):
        """Use predefined prompt templates"""
        templates = {
            "summary": f"""Summarize the following text in 2-3 sentences:

{kwargs.get('text', '')}

Summary:""",
            
            "translation": f"""Translate the following text from {kwargs.get('source_lang', 'English')} to {kwargs.get('target_lang', 'Spanish')}:

{kwargs.get('text', '')}

Translation:""",
            
            "classification": f"""Classify the following text into one of these categories: {', '.join(kwargs.get('categories', []))}

Text: {kwargs.get('text', '')}

Category:""",
            
            "explanation": f"""Explain the following concept in simple terms:

Concept: {kwargs.get('concept', '')}

Explanation:"""
        }
        
        if template_type in templates:
            prompt = templates[template_type]
            return self.generate(prompt)
        else:
            return "Unknown template type"

def main():
    engineer = PromptEngineer()
    
    print("\n" + "="*50)
    print("Prompt Engineering Playground")
    print("="*50)
    print("\nAvailable techniques:")
    print("1. Zero-shot")
    print("2. Few-shot")
    print("3. Chain-of-thought")
    print("4. Role-based")
    print("5. Template prompts")
    print("\nType 'quit' to exit\n")
    
    while True:
        print("\nSelect a technique:")
        print("1. Zero-shot | 2. Few-shot | 3. CoT | 4. Role-based | 5. Template")
        choice = input("Choice [1]: ").strip() or "1"
        
        if choice.lower() == 'quit':
            break
        
        if choice == "1":
            question = input("Enter your question: ")
            result = engineer.zero_shot(question)
            print(f"\nResult: {result}")
        
        elif choice == "2":
            print("Enter examples (press Enter twice to finish):")
            examples = []
            while True:
                ex = input()
                if not ex:
                    break
                examples.append(ex)
            question = input("Now enter your question: ")
            result = engineer.few_shot(examples, question)
            print(f"\nResult: {result}")
        
        elif choice == "3":
            problem = input("Enter the problem: ")
            result = engineer.chain_of_thought(problem)
            print(f"\nResult: {result}")
        
        elif choice == "4":
            role = input("Enter the role (e.g., 'doctor', 'teacher'): ")
            task = input("Enter the task: ")
            result = engineer.role_based(role, task)
            print(f"\nResult: {result}")
        
        elif choice == "5":
            print("Available templates: summary, translation, classification, explanation")
            template = input("Template type: ")
            if template == "summary":
                text = input("Enter text to summarize: ")
                result = engineer.template_prompt("summary", text=text)
            elif template == "explanation":
                concept = input("Enter concept to explain: ")
                result = engineer.template_prompt("explanation", concept=concept)
            else:
                result = "Template not fully implemented. Check code."
            print(f"\nResult: {result}")
        
        print("\n" + "-"*50)

if __name__ == "__main__":
    main()

