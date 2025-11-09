"""
Document Question & Answering System
Answer questions based on document content.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import re

class DocumentQA:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """Initialize the Q&A system"""
        print("Loading Q&A model...")
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
        
        # Document storage
        self.documents = []
        self.chunks = []
        
        print("Q&A system ready!")
    
    def load_document(self, file_path):
        """Load a document from a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.documents.append(content)
            print(f"Document loaded: {file_path}")
            return True
        except Exception as e:
            print(f"Error loading document: {e}")
            return False
    
    def add_document_text(self, text):
        """Add document text directly"""
        self.documents.append(text)
        print("Document text added")
    
    def chunk_document(self, chunk_size=500, overlap=50):
        """Split documents into chunks for processing"""
        self.chunks = []
        
        for doc in self.documents:
            # Simple sentence-based chunking
            sentences = re.split(r'[.!?]+', doc)
            current_chunk = ""
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                if len(current_chunk) + len(sentence) < chunk_size:
                    current_chunk += sentence + ". "
                else:
                    if current_chunk:
                        self.chunks.append(current_chunk.strip())
                    current_chunk = sentence + ". "
            
            if current_chunk:
                self.chunks.append(current_chunk.strip())
        
        print(f"Document split into {len(self.chunks)} chunks")
    
    def find_relevant_chunks(self, question, top_k=3):
        """Find the most relevant chunks for a question"""
        # Simple keyword-based retrieval (you can improve this with embeddings)
        question_lower = question.lower()
        question_words = set(question_lower.split())
        
        chunk_scores = []
        for i, chunk in enumerate(self.chunks):
            chunk_lower = chunk.lower()
            # Count matching words
            score = sum(1 for word in question_words if word in chunk_lower)
            chunk_scores.append((score, i, chunk))
        
        # Sort by score and return top_k
        chunk_scores.sort(reverse=True, key=lambda x: x[0])
        relevant_chunks = [chunk for _, _, chunk in chunk_scores[:top_k]]
        
        return relevant_chunks
    
    def answer_question(self, question, max_length=200):
        """Answer a question based on document content"""
        if not self.chunks:
            self.chunk_document()
        
        # Find relevant context
        relevant_chunks = self.find_relevant_chunks(question, top_k=2)
        context = "\n\n".join(relevant_chunks)
        
        # Build prompt
        prompt = f"""Use the following context to answer the question. If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}

Answer:"""
        
        # Generate answer
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer
        if "Answer:" in response:
            answer = response.split("Answer:")[-1].strip()
        else:
            answer = response[len(prompt):].strip()
        
        return answer, context

def main():
    qa_system = DocumentQA()
    
    print("\n" + "="*50)
    print("Document Question & Answering System")
    print("="*50)
    print("\nCommands:")
    print("  - 'load <file_path>' to load a document")
    print("  - 'add' to add text directly")
    print("  - 'ask <question>' to ask a question")
    print("  - 'quit' to exit")
    print("\n")
    
    # Sample document for testing
    sample_doc = """
    Python is a high-level programming language known for its simplicity and readability.
    It was created by Guido van Rossum and first released in 1991.
    Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.
    It has a large standard library and an active community.
    Python is widely used in web development, data science, artificial intelligence, and automation.
    """
    qa_system.add_document_text(sample_doc)
    qa_system.chunk_document()
    
    while True:
        user_input = input("> ").strip()
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        if user_input.startswith('load '):
            file_path = user_input[5:].strip()
            qa_system.load_document(file_path)
            qa_system.chunk_document()
        
        elif user_input == 'add':
            print("Enter document text (press Enter twice to finish):")
            lines = []
            while True:
                line = input()
                if not line:
                    break
                lines.append(line)
            text = "\n".join(lines)
            qa_system.add_document_text(text)
            qa_system.chunk_document()
        
        elif user_input.startswith('ask '):
            question = user_input[4:].strip()
            if question:
                print("\nSearching documents...")
                answer, context = qa_system.answer_question(question)
                print(f"\nQuestion: {question}")
                print(f"\nAnswer: {answer}")
                print(f"\nContext used:\n{context[:200]}...")
        
        else:
            print("Unknown command. Try 'load <file>', 'add', 'ask <question>', or 'quit'")
        
        print()

if __name__ == "__main__":
    main()


