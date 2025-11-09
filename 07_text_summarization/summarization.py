"""
Text Summarization System
Summarize long documents into shorter versions.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

class TextSummarizer:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """Initialize the summarizer"""
        print("Loading summarization model...")
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
        
        print("Summarizer ready!")
    
    def summarize(self, text, max_length=100, summary_type="short"):
        """Generate a summary of the text"""
        length_instructions = {
            "short": "one or two sentences",
            "medium": "a brief paragraph (3-5 sentences)",
            "long": "a detailed summary (2-3 paragraphs)"
        }
        
        instruction = length_instructions.get(summary_type, "a brief summary")
        
        prompt = f"""Summarize the following text in {instruction}. Focus on the main points and key information.

Text:
{text}

Summary:"""
        
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        summary = response.split("Summary:")[-1].strip()
        
        return summary
    
    def extractive_summary(self, text, num_sentences=3):
        """Extract key sentences from the text"""
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Simple scoring based on length and keywords (can be improved)
        scored_sentences = []
        for sentence in sentences:
            score = len(sentence.split())  # Prefer longer sentences
            # Boost score for sentences with important words
            important_words = ['important', 'key', 'main', 'significant', 'critical', 'essential']
            score += sum(5 for word in important_words if word.lower() in sentence.lower())
            scored_sentences.append((score, sentence))
        
        # Sort by score and take top sentences
        scored_sentences.sort(reverse=True, key=lambda x: x[0])
        key_sentences = [sentence for _, sentence in scored_sentences[:num_sentences]]
        
        # Maintain original order
        summary_sentences = []
        for sentence in sentences:
            if sentence in key_sentences:
                summary_sentences.append(sentence)
                if len(summary_sentences) >= num_sentences:
                    break
        
        return ". ".join(summary_sentences) + "."
    
    def bullet_point_summary(self, text, num_points=5):
        """Generate a bullet-point summary"""
        prompt = f"""Summarize the following text as {num_points} bullet points. Each point should be concise and cover a key idea.

Text:
{text}

Summary:"""
        
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + 150,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        summary = response.split("Summary:")[-1].strip()
        
        return summary
    
    def summarize_file(self, file_path, summary_type="medium"):
        """Summarize a text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return self.summarize(text, summary_type=summary_type)
        except Exception as e:
            return f"Error reading file: {e}"

def main():
    summarizer = TextSummarizer()
    
    print("\n" + "="*50)
    print("Text Summarization System")
    print("="*50)
    print("\nCommands:")
    print("  - 'summarize <text>' - Generate summary")
    print("  - 'short <text>' - Short summary (1-2 sentences)")
    print("  - 'medium <text>' - Medium summary (paragraph)")
    print("  - 'long <text>' - Long summary (multi-paragraph)")
    print("  - 'extract <text>' - Extractive summary")
    print("  - 'bullets <text>' - Bullet point summary")
    print("  - 'file <path>' - Summarize a file")
    print("  - 'quit' - Exit")
    print("\n")
    
    while True:
        user_input = input("> ").strip()
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        if user_input.startswith('summarize '):
            text = user_input[10:].strip()
            summary = summarizer.summarize(text, summary_type="medium")
            print(f"\nSummary:\n{summary}\n")
        
        elif user_input.startswith('short '):
            text = user_input[6:].strip()
            summary = summarizer.summarize(text, max_length=50, summary_type="short")
            print(f"\nShort Summary:\n{summary}\n")
        
        elif user_input.startswith('medium '):
            text = user_input[7:].strip()
            summary = summarizer.summarize(text, max_length=100, summary_type="medium")
            print(f"\nMedium Summary:\n{summary}\n")
        
        elif user_input.startswith('long '):
            text = user_input[5:].strip()
            summary = summarizer.summarize(text, max_length=200, summary_type="long")
            print(f"\nLong Summary:\n{summary}\n")
        
        elif user_input.startswith('extract '):
            text = user_input[8:].strip()
            summary = summarizer.extractive_summary(text, num_sentences=3)
            print(f"\nExtractive Summary:\n{summary}\n")
        
        elif user_input.startswith('bullets '):
            text = user_input[8:].strip()
            summary = summarizer.bullet_point_summary(text, num_points=5)
            print(f"\nBullet Points:\n{summary}\n")
        
        elif user_input.startswith('file '):
            file_path = user_input[5:].strip()
            summary = summarizer.summarize_file(file_path)
            print(f"\nSummary:\n{summary}\n")
        
        else:
            print("Unknown command. Type 'quit' to exit.")

if __name__ == "__main__":
    main()


