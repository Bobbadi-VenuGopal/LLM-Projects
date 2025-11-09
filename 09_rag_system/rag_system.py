"""
RAG System (Retrieval Augmented Generation)
Use vector database to retrieve relevant context before generating answers.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import torch
import os

class RAGSystem:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                 embedding_model="all-MiniLM-L6-v2"):
        """Initialize the RAG system"""
        print("Initializing RAG system...")
        
        # Load generation model
        print("Loading generation model...")
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
        
        # Load embedding model
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB
        print("Initializing vector database...")
        chroma_client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./rag_db"
        ))
        
        # Get or create collection
        self.collection = chroma_client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        print("RAG system ready!")
    
    def add_document(self, document_id, text, metadata=None):
        """Add a document to the vector database"""
        # Split into chunks (simple sentence-based chunking)
        chunks = self._chunk_text(text, chunk_size=200)
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(chunks).tolist()
        
        # Create IDs for chunks
        chunk_ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
        
        # Prepare metadata
        chunk_metadata = []
        for i, chunk in enumerate(chunks):
            chunk_meta = {"document_id": document_id, "chunk_index": i, "text": chunk}
            if metadata:
                chunk_meta.update(metadata)
            chunk_metadata.append(chunk_meta)
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=chunks,
            ids=chunk_ids,
            metadatas=chunk_metadata
        )
        
        print(f"Added document '{document_id}' with {len(chunks)} chunks")
    
    def _chunk_text(self, text, chunk_size=200, overlap=50):
        """Split text into overlapping chunks"""
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def retrieve_context(self, query, top_k=3):
        """Retrieve relevant context for a query"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        
        # Search in vector database
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Extract documents
        contexts = results['documents'][0] if results['documents'] else []
        return contexts
    
    def generate_answer(self, question, max_length=200):
        """Generate an answer using RAG"""
        # Retrieve relevant context
        contexts = self.retrieve_context(question, top_k=2)
        
        if not contexts:
            return "I don't have enough information to answer this question."
        
        # Combine contexts
        context = "\n\n".join(contexts)
        
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
        
        return answer, contexts
    
    def query(self, question):
        """Query the RAG system"""
        answer, contexts = self.generate_answer(question)
        return {
            'question': question,
            'answer': answer,
            'contexts': contexts
        }

def main():
    rag = RAGSystem()
    
    print("\n" + "="*50)
    print("RAG System (Retrieval Augmented Generation)")
    print("="*50)
    print("\nCommands:")
    print("  - 'add <doc_id> <text>' - Add a document")
    print("  - 'query <question>' - Ask a question")
    print("  - 'load <file_path> <doc_id>' - Load document from file")
    print("  - 'quit' - Exit")
    print("\n")
    
    # Add sample documents
    sample_docs = [
        ("python_intro", "Python is a high-level programming language known for its simplicity. It was created by Guido van Rossum in 1991. Python supports multiple programming paradigms."),
        ("ai_basics", "Artificial Intelligence (AI) is the simulation of human intelligence by machines. Machine learning is a subset of AI that enables computers to learn from data."),
        ("web_dev", "Web development involves creating websites and web applications. Common technologies include HTML, CSS, and JavaScript. Backend development uses languages like Python, Node.js, and Java.")
    ]
    
    print("Adding sample documents...")
    for doc_id, text in sample_docs:
        rag.add_document(doc_id, text)
    print()
    
    while True:
        user_input = input("> ").strip()
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        if user_input.startswith('add '):
            parts = user_input[4:].strip().split(' ', 1)
            if len(parts) == 2:
                doc_id = parts[0]
                text = parts[1]
                rag.add_document(doc_id, text)
            else:
                print("Usage: add <doc_id> <text>")
        
        elif user_input.startswith('query '):
            question = user_input[6:].strip()
            if question:
                result = rag.query(question)
                print(f"\nQuestion: {result['question']}")
                print(f"\nAnswer: {result['answer']}")
                print(f"\nContext used:")
                for i, ctx in enumerate(result['contexts'], 1):
                    print(f"  {i}. {ctx[:100]}...")
                print()
        
        elif user_input.startswith('load '):
            parts = user_input[5:].strip().split(' ', 1)
            if len(parts) == 2:
                file_path = parts[0]
                doc_id = parts[1]
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    rag.add_document(doc_id, text)
                except Exception as e:
                    print(f"Error loading file: {e}")
            else:
                print("Usage: load <file_path> <doc_id>")
        
        else:
            print("Unknown command. Try 'add', 'query', 'load', or 'quit'")

if __name__ == "__main__":
    main()


