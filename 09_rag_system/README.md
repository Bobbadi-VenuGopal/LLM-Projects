# Project 9: RAG System (Retrieval Augmented Generation)

## Overview
Build a Retrieval Augmented Generation system that uses vector databases to retrieve relevant information before generating answers.

## What You'll Learn
- Vector embeddings
- Vector databases (ChromaDB)
- Semantic search
- RAG architecture
- Context retrieval and generation

## Model Used
- **TinyLlama-1.1B-Chat** - For generation
- **Sentence Transformers** - For embeddings

## Dependencies
```bash
pip install chromadb sentence-transformers
```

## How to Run

```bash
cd 09_rag_system
python rag_system.py
```

## Features
- Document embedding and storage
- Semantic search
- Context-aware generation
- Persistent vector database
- Multiple document support

## Example
```
Document: "Python is a programming language..."
Question: "What is Python?"
System: Retrieves relevant chunks -> Generates answer with context
```


