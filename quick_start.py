"""
Quick Start Script
Helps you get started with the LLM projects.
"""

import os
import sys

def check_dependencies():
    """Check if required dependencies are installed"""
    required = ['transformers', 'torch', 'numpy']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    else:
        print("✅ All core dependencies are installed!")
        return True

def list_projects():
    """List all available projects"""
    projects = [
        ("01_basic_text_generation", "Basic Text Generation"),
        ("02_simple_chatbot", "Simple Chatbot"),
        ("03_text_completion", "Text Completion"),
        ("04_prompt_engineering", "Prompt Engineering"),
        ("05_document_qa", "Document Q&A"),
        ("06_sentiment_analysis", "Sentiment Analysis"),
        ("07_text_summarization", "Text Summarization"),
        ("08_named_entity_recognition", "Named Entity Recognition"),
        ("09_rag_system", "RAG System"),
        ("10_fine_tuning", "Fine-tuning"),
        ("11_multi_agent_system", "Multi-Agent System"),
        ("12_function_calling", "Function Calling"),
    ]
    
    print("\n📚 Available Projects:")
    print("=" * 50)
    for folder, name in projects:
        if os.path.exists(folder):
            print(f"✅ {folder}: {name}")
        else:
            print(f"❌ {folder}: {name} (not found)")
    print()

def run_project(project_num):
    """Run a specific project"""
    projects = {
        "1": ("01_basic_text_generation", "text_generation.py"),
        "2": ("02_simple_chatbot", "chatbot.py"),
        "3": ("03_text_completion", "completion.py"),
        "4": ("04_prompt_engineering", "prompt_engineering.py"),
        "5": ("05_document_qa", "document_qa.py"),
        "6": ("06_sentiment_analysis", "sentiment_analysis.py"),
        "7": ("07_text_summarization", "summarization.py"),
        "8": ("08_named_entity_recognition", "ner.py"),
        "9": ("09_rag_system", "rag_system.py"),
        "10": ("10_fine_tuning", "fine_tune.py"),
        "11": ("11_multi_agent_system", "multi_agent.py"),
        "12": ("12_function_calling", "function_calling.py"),
    }
    
    if project_num not in projects:
        print(f"❌ Invalid project number: {project_num}")
        return
    
    folder, script = projects[project_num]
    
    if not os.path.exists(folder):
        print(f"❌ Project folder not found: {folder}")
        return
    
    script_path = os.path.join(folder, script)
    if not os.path.exists(script_path):
        print(f"❌ Script not found: {script_path}")
        return
    
    print(f"\n🚀 Running {folder}...")
    print("=" * 50)
    os.chdir(folder)
    os.system(f"{sys.executable} {script}")
    os.chdir("..")

def main():
    print("\n" + "=" * 50)
    print("🚀 LLM Projects - Quick Start")
    print("=" * 50)
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    deps_ok = check_dependencies()
    
    if not deps_ok:
        print("\n⚠️  Please install dependencies first:")
        print("   pip install -r requirements.txt")
        return
    
    # List projects
    list_projects()
    
    # Menu
    print("Options:")
    print("  1-12: Run a specific project")
    print("  list: List all projects")
    print("  quit: Exit")
    
    while True:
        choice = input("\nEnter your choice: ").strip().lower()
        
        if choice == 'quit':
            print("Goodbye! 👋")
            break
        elif choice == 'list':
            list_projects()
        elif choice.isdigit() and 1 <= int(choice) <= 12:
            run_project(choice)
        else:
            print("Invalid choice. Enter a number (1-12), 'list', or 'quit'")

if __name__ == "__main__":
    main()


