"""
Fine-tuning LLMs
Learn to fine-tune a small LLM on custom data.
"""

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import torch

class LLMFineTuner:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """Initialize the fine-tuner"""
        print(f"Loading base model: {model_name}")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Model loaded!")
    
    def prepare_dataset(self, texts, max_length=512):
        """Prepare dataset for training"""
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors="pt"
            )
        
        # Create dataset
        dataset = Dataset.from_dict({'text': texts})
        
        # Tokenize
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def create_sample_dataset(self):
        """Create a sample dataset for demonstration"""
        sample_texts = [
            "Python is a great programming language for beginners.",
            "Machine learning is transforming the tech industry.",
            "Artificial intelligence will change the future.",
            "Web development requires HTML, CSS, and JavaScript.",
            "Data science combines statistics and programming.",
            "Cloud computing enables scalable applications.",
            "Cybersecurity is crucial for protecting data.",
            "Mobile app development is in high demand.",
            "Blockchain technology offers decentralized solutions.",
            "DevOps practices improve software delivery."
        ]
        return self.prepare_dataset(sample_texts)
    
    def fine_tune(self, train_dataset, output_dir="./fine_tuned_model", 
                  num_epochs=3, batch_size=2, learning_rate=5e-5):
        """Fine-tune the model"""
        print("\nStarting fine-tuning...")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            logging_steps=10,
            save_steps=100,
            evaluation_strategy="no",
            save_total_limit=2,
            report_to="none",  # Disable wandb/tensorboard
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal LM, not masked LM
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        
        # Train
        print("\nTraining started...")
        trainer.train()
        
        # Save model
        print(f"\nSaving model to {output_dir}...")
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        print("Fine-tuning complete!")
        return trainer
    
    def generate_text(self, prompt, max_length=100, temperature=0.7):
        """Generate text with the fine-tuned model"""
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
            self.model = self.model.to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated

def main():
    print("\n" + "="*50)
    print("LLM Fine-tuning Demo")
    print("="*50)
    print("\nNote: This is a simplified example for learning.")
    print("For production, use proper training frameworks and techniques like LoRA.\n")
    
    # Initialize fine-tuner
    fine_tuner = LLMFineTuner()
    
    print("\nOptions:")
    print("1. Create sample dataset and fine-tune")
    print("2. Load custom dataset and fine-tune")
    print("3. Load fine-tuned model and test")
    print("4. Exit")
    
    choice = input("\nSelect option [1]: ").strip() or "1"
    
    if choice == "1":
        # Create sample dataset
        print("\nCreating sample dataset...")
        train_dataset = fine_tuner.create_sample_dataset()
        
        # Fine-tune
        print("\nWarning: Fine-tuning will take time and resources.")
        confirm = input("Continue? (yes/no) [no]: ").strip().lower()
        
        if confirm == "yes":
            fine_tuner.fine_tune(
                train_dataset,
                num_epochs=1,  # Reduced for demo
                batch_size=1,  # Small batch for limited resources
            )
        else:
            print("Fine-tuning cancelled.")
    
    elif choice == "2":
        print("\nEnter training texts (one per line, press Enter twice to finish):")
        texts = []
        while True:
            text = input()
            if not text:
                break
            texts.append(text)
        
        if texts:
            train_dataset = fine_tuner.prepare_dataset(texts)
            confirm = input("\nStart fine-tuning? (yes/no) [no]: ").strip().lower()
            
            if confirm == "yes":
                fine_tuner.fine_tune(train_dataset, num_epochs=1, batch_size=1)
            else:
                print("Fine-tuning cancelled.")
        else:
            print("No texts provided.")
    
    elif choice == "3":
        model_path = input("Enter path to fine-tuned model [./fine_tuned_model]: ").strip() or "./fine_tuned_model"
        
        try:
            print(f"\nLoading model from {model_path}...")
            fine_tuner.tokenizer = AutoTokenizer.from_pretrained(model_path)
            fine_tuner.model = AutoModelForCausalLM.from_pretrained(model_path)
            print("Model loaded!")
            
            while True:
                prompt = input("\nEnter prompt (or 'quit' to exit): ").strip()
                if prompt.lower() == 'quit':
                    break
                
                if prompt:
                    generated = fine_tuner.generate_text(prompt)
                    print(f"\nGenerated: {generated}\n")
        
        except Exception as e:
            print(f"Error loading model: {e}")
    
    else:
        print("Goodbye!")

if __name__ == "__main__":
    # Note: This requires the 'datasets' library
    # Install with: pip install datasets
    try:
        main()
    except ImportError:
        print("Error: 'datasets' library not found.")
        print("Install it with: pip install datasets")


