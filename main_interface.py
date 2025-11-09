"""
Unified Interface for All LLM Projects
A single interface to interact with all 12 projects.
"""

import os
import sys
import importlib.util

# Fix encoding for Windows terminals
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

# Add project directories to path
project_dirs = [
    "01_basic_text_generation",
    "02_simple_chatbot",
    "03_text_completion",
    "04_prompt_engineering",
    "05_document_qa",
    "06_sentiment_analysis",
    "07_text_summarization",
    "08_named_entity_recognition",
    "09_rag_system",
    "10_fine_tuning",
    "11_multi_agent_system",
    "12_function_calling"
]

class UnifiedInterface:
    def __init__(self):
        """Initialize the unified interface"""
        self.projects = {
            "1": {
                "name": "Basic Text Generation",
                "folder": "01_basic_text_generation",
                "module": "text_generation",
                "description": "Generate text from prompts using TinyLlama",
                "handler": self._handle_text_generation
            },
            "2": {
                "name": "Simple Chatbot",
                "folder": "02_simple_chatbot",
                "module": "chatbot",
                "description": "Chat with an AI that maintains conversation history",
                "handler": self._handle_chatbot
            },
            "3": {
                "name": "Text Completion",
                "folder": "03_text_completion",
                "module": "completion",
                "description": "Autocomplete and text completion system",
                "handler": self._handle_completion
            },
            "4": {
                "name": "Prompt Engineering",
                "folder": "04_prompt_engineering",
                "module": "prompt_engineering",
                "description": "Learn and practice prompt engineering techniques",
                "handler": self._handle_prompt_engineering
            },
            "5": {
                "name": "Document Q&A",
                "folder": "05_document_qa",
                "module": "document_qa",
                "description": "Answer questions based on documents",
                "handler": self._handle_document_qa
            },
            "6": {
                "name": "Sentiment Analysis",
                "folder": "06_sentiment_analysis",
                "module": "sentiment_analysis",
                "description": "Analyze sentiment and emotions in text",
                "handler": self._handle_sentiment_analysis
            },
            "7": {
                "name": "Text Summarization",
                "folder": "07_text_summarization",
                "module": "summarization",
                "description": "Summarize long documents",
                "handler": self._handle_summarization
            },
            "8": {
                "name": "Named Entity Recognition",
                "folder": "08_named_entity_recognition",
                "module": "ner",
                "description": "Extract named entities from text",
                "handler": self._handle_ner
            },
            "9": {
                "name": "RAG System",
                "folder": "09_rag_system",
                "module": "rag_system",
                "description": "Retrieval Augmented Generation with vector database",
                "handler": self._handle_rag
            },
            "10": {
                "name": "Fine-tuning",
                "folder": "10_fine_tuning",
                "module": "fine_tune",
                "description": "Fine-tune LLMs on custom data",
                "handler": self._handle_fine_tuning
            },
            "11": {
                "name": "Multi-Agent System",
                "folder": "11_multi_agent_system",
                "module": "multi_agent",
                "description": "Multiple AI agents working together",
                "handler": self._handle_multi_agent
            },
            "12": {
                "name": "Function Calling",
                "folder": "12_function_calling",
                "module": "function_calling",
                "description": "LLM that can call functions and tools",
                "handler": self._handle_function_calling
            }
        }
        
        self.loaded_modules = {}
        self.active_project = None
    
    def load_module(self, project_key):
        """Dynamically load a project module"""
        if project_key in self.loaded_modules:
            return self.loaded_modules[project_key]
        
        project = self.projects[project_key]
        folder = project["folder"]
        module_name = project["module"]
        
        if not os.path.exists(folder):
            return None
        
        module_path = os.path.join(folder, f"{module_name}.py")
        if not os.path.exists(module_path):
            return None
        
        # Load module
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.path.insert(0, folder)
        spec.loader.exec_module(module)
        sys.path.pop(0)
        
        self.loaded_modules[project_key] = module
        return module
    
    def show_menu(self):
        """Display the main menu"""
        print("\n" + "="*70)
        print("LLM Projects - Unified Interface")
        print("="*70)
        print("\nAvailable Projects:\n")
        
        for key, project in self.projects.items():
            status = "[OK]" if os.path.exists(project["folder"]) else "[X]"
            print(f"  {key}. {status} {project['name']}")
            print(f"     {project['description']}\n")
        
        print("Commands:")
        print("  - Enter project number (1-12) to use that project")
        print("  - 'list' - Show this menu again")
        print("  - 'help' - Show help for current project")
        print("  - 'back' - Return to main menu (when in a project)")
        print("  - 'quit' - Exit the program")
        print("="*70)
    
    def _handle_text_generation(self, module):
        """Handle text generation project"""
        print("\n" + "="*70)
        print("Basic Text Generation")
        print("="*70)
        print("\nLoading model... (this may take a minute on first run)")
        
        try:
            model, tokenizer = module.load_model()
            print("\n[OK] Model loaded! Start generating text.")
            print("Type 'back' to return to main menu\n")
            
            while True:
                prompt = input("Enter prompt: ").strip()
                
                if prompt.lower() == 'back':
                    break
                
                if not prompt:
                    continue
                
                print("\nGenerating...")
                result = module.generate_text(model, tokenizer, prompt, max_length=150)
                print(f"\nGenerated text:\n{result}\n")
                print("-"*70 + "\n")
        
        except Exception as e:
            print(f"\n[ERROR] Error: {e}\n")
    
    def _handle_chatbot(self, module):
        """Handle chatbot project"""
        print("\n" + "="*70)
        print("Simple Chatbot")
        print("="*70)
        print("\nLoading chatbot... (this may take a minute on first run)")
        
        try:
            chatbot = module.SimpleChatbot()
            print("\n[OK] Chatbot ready! Start chatting.")
            print("Type 'clear' to clear history, 'back' to return to main menu\n")
            
            while True:
                user_input = input("You: ").strip()
                
                if user_input.lower() == 'back':
                    break
                
                if user_input.lower() == 'clear':
                    chatbot.clear_history()
                    continue
                
                if not user_input:
                    continue
                
                print("Bot: ", end="", flush=True)
                response = chatbot.generate_response(user_input)
                print(response + "\n")
        
        except Exception as e:
            print(f"\n[ERROR] Error: {e}\n")
    
    def _handle_completion(self, module):
        """Handle text completion project"""
        print("\n" + "="*70)
        print("Text Completion")
        print("="*70)
        print("\nLoading model... (this may take a minute on first run)")
        
        try:
            completer = module.TextCompleter()
            print("\n[OK] Completer ready!")
            print("Mode options: single, multi, next")
            print("Type 'back' to return to main menu\n")
            
            while True:
                prefix = input("Enter text to complete: ").strip()
                
                if prefix.lower() == 'back':
                    break
                
                if not prefix:
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
                
                print("\n" + "-"*70 + "\n")
        
        except Exception as e:
            print(f"\n[ERROR] Error: {e}\n")
    
    def _handle_prompt_engineering(self, module):
        """Handle prompt engineering project"""
        print("\n" + "="*70)
        print("Prompt Engineering")
        print("="*70)
        print("\nLoading model... (this may take a minute on first run)")
        
        try:
            engineer = module.PromptEngineer()
            print("\n[OK] Prompt engineer ready!")
            print("\nTechniques: 1. Zero-shot | 2. Few-shot | 3. CoT | 4. Role-based | 5. Template")
            print("Type 'back' to return to main menu\n")
            
            while True:
                print("\nSelect technique:")
                choice = input("Choice [1]: ").strip() or "1"
                
                if choice.lower() == 'back':
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
                    print("Available templates: summary, explanation")
                    template = input("Template type: ")
                    if template == "summary":
                        text = input("Enter text to summarize: ")
                        result = engineer.template_prompt("summary", text=text)
                    elif template == "explanation":
                        concept = input("Enter concept to explain: ")
                        result = engineer.template_prompt("explanation", concept=concept)
                    else:
                        result = "Template not fully implemented."
                    print(f"\nResult: {result}")
                
                print("\n" + "-"*70)
        
        except Exception as e:
            print(f"\n[ERROR] Error: {e}\n")
    
    def _handle_document_qa(self, module):
        """Handle document Q&A project"""
        print("\n" + "="*70)
        print("Document Question & Answering")
        print("="*70)
        print("\nLoading Q&A system... (this may take a minute on first run)")
        
        try:
            qa_system = module.DocumentQA()
            print("\n[OK] Q&A system ready!")
            print("\nCommands: 'load <file>', 'add' (add text), 'ask <question>'")
            print("Type 'back' to return to main menu\n")
            
            # Add sample document
            sample_doc = """
            Python is a high-level programming language known for its simplicity and readability.
            It was created by Guido van Rossum and first released in 1991.
            Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.
            """
            qa_system.add_document_text(sample_doc)
            qa_system.chunk_document()
            print("[OK] Sample document added\n")
            
            while True:
                user_input = input("> ").strip()
                
                if user_input.lower() == 'back':
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
                        print(f"\nContext: {context[:200]}...")
                
                else:
                    print("Commands: 'load <file>', 'add', 'ask <question>', 'back'")
        
        except Exception as e:
            print(f"\n[ERROR] Error: {e}\n")
    
    def _handle_sentiment_analysis(self, module):
        """Handle sentiment analysis project"""
        print("\n" + "="*70)
        print("Sentiment Analysis")
        print("="*70)
        print("\nLoading sentiment analyzer... (this may take a minute on first run)")
        
        try:
            analyzer = module.SentimentAnalyzer()
            print("\n[OK] Sentiment analyzer ready!")
            print("\nCommands: 'sentiment <text>', 'emotion <text>', 'detailed <text>', 'batch'")
            print("Type 'back' to return to main menu\n")
            
            while True:
                user_input = input("> ").strip()
                
                if user_input.lower() == 'back':
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
                    print("Enter texts (one per line, press Enter twice to finish):")
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
                    print("Commands: 'sentiment <text>', 'emotion <text>', 'detailed <text>', 'batch', 'back'")
        
        except Exception as e:
            print(f"\n[ERROR] Error: {e}\n")
    
    def _handle_summarization(self, module):
        """Handle summarization project"""
        print("\n" + "="*70)
        print("Text Summarization")
        print("="*70)
        print("\nLoading summarizer... (this may take a minute on first run)")
        
        try:
            summarizer = module.TextSummarizer()
            print("\n[OK] Summarizer ready!")
            print("\nCommands: 'short <text>', 'medium <text>', 'long <text>', 'extract <text>', 'bullets <text>', 'file <path>'")
            print("Type 'back' to return to main menu\n")
            
            while True:
                user_input = input("> ").strip()
                
                if user_input.lower() == 'back':
                    break
                
                if user_input.startswith('short '):
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
                    print("Commands: 'short <text>', 'medium <text>', 'long <text>', 'extract <text>', 'bullets <text>', 'file <path>', 'back'")
        
        except Exception as e:
            print(f"\n[ERROR] Error: {e}\n")
    
    def _handle_ner(self, module):
        """Handle NER project"""
        print("\n" + "="*70)
        print("Named Entity Recognition")
        print("="*70)
        print("\nLoading NER system... (this may take a minute on first run)")
        
        try:
            ner = module.NamedEntityRecognizer()
            print("\n[OK] NER system ready!")
            print("\nCommands: 'extract <text>', 'json <text>', 'type <entity_type> <text>', 'structured <text>'")
            print("Type 'back' to return to main menu\n")
            
            while True:
                user_input = input("> ").strip()
                
                if user_input.lower() == 'back':
                    break
                
                if user_input.startswith('extract '):
                    text = user_input[8:].strip()
                    entities = ner.extract_entities(text)
                    print(f"\nText: {text}")
                    print("\nEntities:")
                    for entity_type, entity_list in entities.items():
                        if entity_list:
                            print(f"  {entity_type}: {', '.join(entity_list)}")
                    print()
                
                elif user_input.startswith('json '):
                    text = user_input[5:].strip()
                    json_output = ner.format_json(text)
                    print(f"\n{json_output}\n")
                
                elif user_input.startswith('type '):
                    parts = user_input[5:].strip().split(' ', 1)
                    if len(parts) == 2:
                        entity_type = parts[0]
                        text = parts[1]
                        entities = ner.extract_by_type(text, entity_type)
                        print(f"\n{entity_type.upper()} entities: {entities}\n")
                    else:
                        print("Usage: type <entity_type> <text>")
                
                elif user_input.startswith('structured '):
                    text = user_input[11:].strip()
                    structured = ner.extract_structured(text)
                    print(f"\nStructured entities:")
                    for entity in structured:
                        print(f"  {entity}\n")
                
                else:
                    print("Commands: 'extract <text>', 'json <text>', 'type <entity_type> <text>', 'structured <text>', 'back'")
        
        except Exception as e:
            print(f"\n[ERROR] Error: {e}\n")
    
    def _handle_rag(self, module):
        """Handle RAG system project"""
        print("\n" + "="*70)
        print("RAG System (Retrieval Augmented Generation)")
        print("="*70)
        print("\nLoading RAG system... (this may take a minute on first run)")
        print("Note: This requires chromadb and sentence-transformers")
        
        try:
            rag = module.RAGSystem()
            print("\n[OK] RAG system ready!")
            print("\nCommands: 'add <doc_id> <text>', 'query <question>', 'load <file_path> <doc_id>'")
            print("Type 'back' to return to main menu\n")
            
            # Add sample documents
            sample_docs = [
                ("python_intro", "Python is a high-level programming language known for its simplicity."),
                ("ai_basics", "Artificial Intelligence (AI) is the simulation of human intelligence by machines."),
            ]
            print("Adding sample documents...")
            for doc_id, text in sample_docs:
                rag.add_document(doc_id, text)
            print()
            
            while True:
                user_input = input("> ").strip()
                
                if user_input.lower() == 'back':
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
                    print("Commands: 'add <doc_id> <text>', 'query <question>', 'load <file_path> <doc_id>', 'back'")
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Make sure chromadb and sentence-transformers are installed:")
            print("  pip install chromadb sentence-transformers\n")
    
    def _handle_fine_tuning(self, module):
        """Handle fine-tuning project"""
        print("\n" + "="*70)
        print("Fine-tuning")
        print("="*70)
        print("\nNote: Fine-tuning requires significant computational resources")
        print("This is a simplified example for learning purposes.")
        print("\nType 'back' to return to main menu")
        print("Or run the project directly: cd 10_fine_tuning && python fine_tune.py\n")
        
        response = input("Continue? (yes/no) [no]: ").strip().lower()
        if response == 'yes':
            print("\nLoading fine-tuning module...")
            try:
                # This would require more complex setup
                print("For fine-tuning, please run the project directly:")
                print("  cd 10_fine_tuning")
                print("  python fine_tune.py")
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("Returning to main menu...")
    
    def _handle_multi_agent(self, module):
        """Handle multi-agent system project"""
        print("\n" + "="*70)
        print("Multi-Agent System")
        print("="*70)
        print("\nLoading multi-agent system... (this may take a minute on first run)")
        
        try:
            system = module.MultiAgentSystem()
            print("\nSetting up agents...")
            system.add_agent("alice", "researcher")
            system.add_agent("bob", "writer")
            system.add_agent("charlie", "analyzer")
            print("\n[OK] Multi-agent system ready!")
            print("\nCommands: 'task <description>', 'simple <agent_name> <task>', 'agents'")
            print("Type 'back' to return to main menu\n")
            
            while True:
                user_input = input("> ").strip()
                
                if user_input.lower() == 'back':
                    break
                
                elif user_input.lower() == 'agents':
                    print("\nAgents:")
                    for name, agent in system.agents.items():
                        print(f"  {name}: {agent.role}")
                    print()
                
                elif user_input.startswith('task '):
                    task = user_input[5:].strip()
                    if task:
                        results = system.coordinate_task(task, ["alice", "bob", "charlie"])
                        print("\n" + "="*70)
                        print("Results:")
                        print("="*70)
                        for key, value in results.items():
                            print(f"\n{key.upper()}:\n{value[:200]}...")
                        print()
                
                elif user_input.startswith('simple '):
                    parts = user_input[7:].strip().split(' ', 1)
                    if len(parts) == 2:
                        agent_name = parts[0]
                        task = parts[1]
                        result = system.simple_task(task, agent_name)
                        print(f"\nResult:\n{result}\n")
                    else:
                        print("Usage: simple <agent_name> <task>")
                
                else:
                    print("Commands: 'task <description>', 'simple <agent_name> <task>', 'agents', 'back'")
        
        except Exception as e:
            print(f"\n[ERROR] Error: {e}\n")
    
    def _handle_function_calling(self, module):
        """Handle function calling project"""
        print("\n" + "="*70)
        print("Function Calling")
        print("="*70)
        print("\nLoading function calling system... (this may take a minute on first run)")
        
        try:
            llm = module.FunctionCallingLLM()
            print("\n[OK] Function calling system ready!")
            print("\nAvailable functions: calculator, get_time, text_processor, web_search")
            print("Examples: 'Calculate 25 * 37', 'What time is it?', 'Make this uppercase: hello'")
            print("Type 'back' to return to main menu\n")
            
            while True:
                user_input = input("You: ").strip()
                
                if user_input.lower() == 'back':
                    break
                
                if not user_input:
                    continue
                
                print("Assistant: ", end="", flush=True)
                response = llm.process_with_functions(user_input)
                print(response + "\n")
        
        except Exception as e:
            print(f"\n[ERROR] Error: {e}\n")
    
    def run_project(self, project_key):
        """Run a specific project"""
        if project_key not in self.projects:
            print(f"[ERROR] Invalid project number: {project_key}")
            return
        
        project = self.projects[project_key]
        
        if not os.path.exists(project["folder"]):
            print(f"[ERROR] Project folder not found: {project['folder']}")
            return
        
        # Load module
        module = self.load_module(project_key)
        if module is None:
            print(f"[ERROR] Failed to load project module: {project['module']}")
            return
        
        # Set active project
        self.active_project = project_key
        
        # Run handler
        try:
            project["handler"](module)
        except KeyboardInterrupt:
            print("\n\n[WARNING] Interrupted by user")
        except Exception as e:
            print(f"\n[ERROR] Error running project: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.active_project = None
    
    def run(self):
        """Main run loop"""
        self.show_menu()
        
        while True:
            try:
                choice = input("\nSelect a project (1-12) or command: ").strip().lower()
                
                if choice == 'quit' or choice == 'exit':
                    print("\nGoodbye! Thanks for using LLM Projects!")
                    break
                
                elif choice == 'list':
                    self.show_menu()
                
                elif choice == 'help':
                    if self.active_project:
                        project = self.projects[self.active_project]
                        print(f"\nHelp for {project['name']}:")
                        print(f"{project['description']}")
                    else:
                        print("\nSelect a project first to get help.")
                
                elif choice.isdigit() and 1 <= int(choice) <= 12:
                    self.run_project(choice)
                    print("\n" + "="*70)
                    print("Returned to main menu")
                    print("="*70)
                
                else:
                    print("[ERROR] Invalid choice. Enter a number (1-12), 'list', 'help', or 'quit'")
            
            except KeyboardInterrupt:
                print("\n\nGoodbye! Thanks for using LLM Projects!")
                break
            except Exception as e:
                print(f"\n[ERROR] Error: {e}")

def main():
    """Main entry point"""
    interface = UnifiedInterface()
    interface.run()

if __name__ == "__main__":
    main()

