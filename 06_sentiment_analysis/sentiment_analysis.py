"""
Sentiment Analysis with LLMs
Analyze emotions and sentiment in text.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

class SentimentAnalyzer:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """Initialize the sentiment analyzer"""
        print("Loading sentiment analysis model...")
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
        
        print("Sentiment analyzer ready!")
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of text (Positive, Negative, Neutral)"""
        prompt = f"""Analyze the sentiment of the following text and respond with only one word: Positive, Negative, or Neutral.

Text: "{text}"

Sentiment:"""
        
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + 10,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        sentiment = response.split("Sentiment:")[-1].strip().split()[0]
        
        # Clean up response
        sentiment = re.sub(r'[^a-zA-Z]', '', sentiment)
        if sentiment.lower() not in ['positive', 'negative', 'neutral']:
            # Fallback: simple keyword matching
            text_lower = text.lower()
            positive_words = ['good', 'great', 'excellent', 'love', 'happy', 'amazing', 'wonderful']
            negative_words = ['bad', 'terrible', 'hate', 'sad', 'awful', 'horrible', 'disappointed']
            
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            if pos_count > neg_count:
                sentiment = "Positive"
            elif neg_count > pos_count:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"
        
        return sentiment.capitalize()
    
    def detect_emotion(self, text):
        """Detect specific emotions in text"""
        emotions = ['Happy', 'Sad', 'Angry', 'Fearful', 'Surprised', 'Disgusted', 'Neutral']
        emotions_str = ', '.join(emotions)
        
        prompt = f"""Identify the primary emotion in the following text. Respond with only one word from this list: {emotions_str}

Text: "{text}"

Emotion:"""
        
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + 10,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        emotion = response.split("Emotion:")[-1].strip().split()[0]
        emotion = re.sub(r'[^a-zA-Z]', '', emotion).capitalize()
        
        # Validate emotion
        if emotion not in emotions:
            emotion = "Neutral"
        
        return emotion
    
    def detailed_analysis(self, text):
        """Get detailed sentiment and emotion analysis"""
        sentiment = self.analyze_sentiment(text)
        emotion = self.detect_emotion(text)
        
        # Generate explanation
        prompt = f"""Explain the sentiment and emotion of this text in one sentence: "{text}"

Explanation:"""
        
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + 50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        explanation = response.split("Explanation:")[-1].strip()
        
        return {
            'sentiment': sentiment,
            'emotion': emotion,
            'explanation': explanation
        }
    
    def analyze_batch(self, texts):
        """Analyze multiple texts at once"""
        results = []
        for text in texts:
            sentiment = self.analyze_sentiment(text)
            results.append({'text': text, 'sentiment': sentiment})
        return results

def main():
    analyzer = SentimentAnalyzer()
    
    print("\n" + "="*50)
    print("Sentiment Analysis System")
    print("="*50)
    print("\nCommands:")
    print("  - 'sentiment <text>' - Analyze sentiment")
    print("  - 'emotion <text>' - Detect emotion")
    print("  - 'detailed <text>' - Full analysis")
    print("  - 'batch' - Analyze multiple texts")
    print("  - 'quit' - Exit")
    print("\n")
    
    while True:
        user_input = input("> ").strip()
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        if user_input.startswith('sentiment '):
            text = user_input[10:].strip()
            sentiment = analyzer.analyze_sentiment(text)
            print(f"\nText: {text}")
            print(f"Sentiment: {sentiment}\n")
        
        elif user_input.startswith('emotion '):
            text = user_input[8:].strip()
            emotion = analyzer.detect_emotion(text)
            print(f"\nText: {text}")
            print(f"Emotion: {emotion}\n")
        
        elif user_input.startswith('detailed '):
            text = user_input[9:].strip()
            analysis = analyzer.detailed_analysis(text)
            print(f"\nText: {text}")
            print(f"Sentiment: {analysis['sentiment']}")
            print(f"Emotion: {analysis['emotion']}")
            print(f"Explanation: {analysis['explanation']}\n")
        
        elif user_input == 'batch':
            print("Enter texts to analyze (one per line, press Enter twice to finish):")
            texts = []
            while True:
                text = input()
                if not text:
                    break
                texts.append(text)
            
            results = analyzer.analyze_batch(texts)
            print("\nResults:")
            for result in results:
                print(f"\nText: {result['text']}")
                print(f"Sentiment: {result['sentiment']}")
        
        else:
            print("Unknown command. Try 'sentiment <text>', 'emotion <text>', 'detailed <text>', 'batch', or 'quit'")

if __name__ == "__main__":
    main()


