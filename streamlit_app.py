"""
Streamlit Unified Interface for All LLM Projects
A web-based interface to interact with all 12 projects.
"""

import streamlit as st
import os
import sys
import importlib.util

# Fix encoding for Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

# Page config
st.set_page_config(
    page_title="LLM Projects",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    .project-card {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ddd;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'loaded_modules' not in st.session_state:
    st.session_state.loaded_modules = {}
if 'active_project' not in st.session_state:
    st.session_state.active_project = None
if 'project_states' not in st.session_state:
    st.session_state.project_states = {}

# Project definitions
PROJECTS = {
    "1": {
        "name": "Basic Text Generation",
        "folder": "01_basic_text_generation",
        "module": "text_generation",
        "description": "Generate text from prompts using TinyLlama",
        "icon": "📝"
    },
    "2": {
        "name": "Simple Chatbot",
        "folder": "02_simple_chatbot",
        "module": "chatbot",
        "description": "Chat with an AI that maintains conversation history",
        "icon": "💬"
    },
    "3": {
        "name": "Text Completion",
        "folder": "03_text_completion",
        "module": "completion",
        "description": "Autocomplete and text completion system",
        "icon": "✏️"
    },
    "4": {
        "name": "Prompt Engineering",
        "folder": "04_prompt_engineering",
        "module": "prompt_engineering",
        "description": "Learn and practice prompt engineering techniques",
        "icon": "🎯"
    },
    "5": {
        "name": "Document Q&A",
        "folder": "05_document_qa",
        "module": "document_qa",
        "description": "Answer questions based on documents",
        "icon": "📚"
    },
    "6": {
        "name": "Sentiment Analysis",
        "folder": "06_sentiment_analysis",
        "module": "sentiment_analysis",
        "description": "Analyze sentiment and emotions in text",
        "icon": "😊"
    },
    "7": {
        "name": "Text Summarization",
        "folder": "07_text_summarization",
        "module": "summarization",
        "description": "Summarize long documents",
        "icon": "📄"
    },
    "8": {
        "name": "Named Entity Recognition",
        "folder": "08_named_entity_recognition",
        "module": "ner",
        "description": "Extract named entities from text",
        "icon": "🏷️"
    },
    "9": {
        "name": "RAG System",
        "folder": "09_rag_system",
        "module": "rag_system",
        "description": "Retrieval Augmented Generation with vector database",
        "icon": "🔍"
    },
    "10": {
        "name": "Fine-tuning",
        "folder": "10_fine_tuning",
        "module": "fine_tune",
        "description": "Fine-tune LLMs on custom data",
        "icon": "🎓"
    },
    "11": {
        "name": "Multi-Agent System",
        "folder": "11_multi_agent_system",
        "module": "multi_agent",
        "description": "Multiple AI agents working together",
        "icon": "🤖"
    },
    "12": {
        "name": "Function Calling",
        "folder": "12_function_calling",
        "module": "function_calling",
        "description": "LLM that can call functions and tools",
        "icon": "⚙️"
    }
}

def load_module(project_key):
    """Dynamically load a project module"""
    if project_key in st.session_state.loaded_modules:
        return st.session_state.loaded_modules[project_key]
    
    project = PROJECTS[project_key]
    folder = project["folder"]
    module_name = project["module"]
    
    # Check if folder exists
    if not os.path.exists(folder):
        error_msg = f"Project folder not found: {folder}"
        st.error(error_msg)
        st.info("Make sure you're running the app from the project root directory.")
        return None
    
    module_path = os.path.join(folder, f"{module_name}.py")
    if not os.path.exists(module_path):
        error_msg = f"Module file not found: {module_path}"
        st.error(error_msg)
        return None
    
    try:
        # Check for common dependencies first
        required_deps = {
            'transformers': 'transformers library for LLM models',
            'torch': 'PyTorch for model execution'
        }
        
        missing_deps = []
        for dep, desc in required_deps.items():
            try:
                __import__(dep)
            except ImportError:
                missing_deps.append(f"{dep} ({desc})")
        
        if missing_deps:
            error_msg = f"Missing required dependencies:\n\n"
            for dep in missing_deps:
                error_msg += f"  - {dep}\n"
            error_msg += f"\nPlease install them with:\n```bash\npip install -r requirements.txt\n```"
            st.error(error_msg)
            return None
        
        # Add folder to path before importing (keep it there for relative imports)
        original_path = sys.path.copy()
        if folder not in sys.path:
            sys.path.insert(0, os.path.abspath(folder))
        
        # Also add parent directory for absolute imports
        parent_dir = os.path.dirname(os.path.abspath(folder))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        try:
            # Load the module
            spec = importlib.util.spec_from_file_location(module_name, os.path.abspath(module_path))
            if spec is None or spec.loader is None:
                error_msg = f"Could not create spec for module: {module_name}"
                st.error(error_msg)
                return None
            
            module = importlib.util.module_from_spec(spec)
            
            # Execute the module
            spec.loader.exec_module(module)
            
            # Store in session state
            st.session_state.loaded_modules[project_key] = module
            return module
        finally:
            # Restore original path
            sys.path[:] = original_path
            
    except ImportError as e:
        error_msg = f"**Import Error:**\n\n{str(e)}\n\n"
        error_msg += "This is usually due to missing dependencies.\n\n"
        error_msg += "**Solution:**\n"
        error_msg += "```bash\npip install -r requirements.txt\n```"
        st.error(error_msg)
        with st.expander("🔍 Full Error Details"):
            import traceback
            st.code(traceback.format_exc(), language="python")
        return None
    except Exception as e:
        error_msg = f"**Error loading module:**\n\n{str(e)}"
        st.error(error_msg)
        with st.expander("🔍 Full Error Details"):
            import traceback
            st.code(traceback.format_exc(), language="python")
        return None

def render_text_generation(module):
    """Render text generation interface"""
    st.subheader("Basic Text Generation")
    
    if 'text_gen_model' not in st.session_state:
        with st.spinner("Loading model... (this may take a minute on first run)"):
            try:
                model, tokenizer = module.load_model()
                st.session_state.text_gen_model = model
                st.session_state.text_gen_tokenizer = tokenizer
                st.success("Model loaded!")
            except Exception as e:
                st.error(f"Error loading model: {e}")
                return
    
    prompt = st.text_area("Enter your prompt:", height=100)
    
    col1, col2 = st.columns(2)
    max_length = col1.slider("Max Length", 50, 300, 150)
    temperature = col2.slider("Temperature", 0.1, 1.0, 0.7, 0.1)
    
    if st.button("Generate Text", type="primary"):
        if prompt:
            with st.spinner("Generating text..."):
                try:
                    result = module.generate_text(
                        st.session_state.text_gen_model,
                        st.session_state.text_gen_tokenizer,
                        prompt,
                        max_length=max_length,
                        temperature=temperature
                    )
                    st.text_area("Generated Text:", result, height=200)
                except Exception as e:
                    st.error(f"Error generating text: {e}")
        else:
            st.warning("Please enter a prompt")

def render_chatbot(module):
    """Render chatbot interface"""
    st.subheader("Simple Chatbot")
    
    if 'chatbot' not in st.session_state:
        with st.spinner("Loading chatbot... (this may take a minute on first run)"):
            try:
                chatbot = module.SimpleChatbot()
                st.session_state.chatbot = chatbot
                st.success("Chatbot ready!")
            except Exception as e:
                st.error(f"Error loading chatbot: {e}")
                return
    
    # Display chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Show chat history
    for exchange in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(exchange["user"])
        with st.chat_message("assistant"):
            st.write(exchange["assistant"])
    
    # User input
    user_input = st.chat_input("Type your message here...")
    
    col1, col2 = st.columns(2)
    if col1.button("Clear History"):
        st.session_state.chat_history = []
        st.session_state.chatbot.clear_history()
        st.rerun()
    
    if user_input:
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.chatbot.generate_response(user_input)
                st.session_state.chat_history.append({
                    "user": user_input,
                    "assistant": response
                })
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

def render_completion(module):
    """Render text completion interface"""
    st.subheader("Text Completion")
    
    if 'completer' not in st.session_state:
        with st.spinner("Loading completer... (this may take a minute on first run)"):
            try:
                completer = module.TextCompleter()
                st.session_state.completer = completer
                st.success("Completer ready!")
            except Exception as e:
                st.error(f"Error loading completer: {e}")
                return
    
    prefix = st.text_area("Enter text to complete:", height=100)
    mode = st.radio("Mode:", ["Single", "Multiple", "Next Words"], horizontal=True)
    
    if st.button("Complete", type="primary"):
        if prefix:
            with st.spinner("Completing..."):
                try:
                    if mode == "Single":
                        result = st.session_state.completer.complete_single(prefix)
                        st.text_area("Completion:", result, height=200)
                    elif mode == "Multiple":
                        completions = st.session_state.completer.complete_text(prefix, num_completions=3)
                        for i, completion in enumerate(completions, 1):
                            st.text_area(f"Completion {i}:", completion, height=150)
                    else:  # Next Words
                        next_words = st.session_state.completer.get_next_words(prefix, num_words=5)
                        st.write("Top next words:")
                        for word, prob in next_words:
                            st.write(f"- **{word}**: {prob:.2%}")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please enter text to complete")

def render_prompt_engineering(module):
    """Render prompt engineering interface"""
    st.subheader("Prompt Engineering")
    
    if 'engineer' not in st.session_state:
        with st.spinner("Loading prompt engineer... (this may take a minute on first run)"):
            try:
                engineer = module.PromptEngineer()
                st.session_state.engineer = engineer
                st.success("Prompt engineer ready!")
            except Exception as e:
                st.error(f"Error loading engineer: {e}")
                return
    
    technique = st.selectbox(
        "Select technique:",
        ["Zero-shot", "Few-shot", "Chain-of-thought", "Role-based", "Template"]
    )
    
    if technique == "Zero-shot":
        question = st.text_input("Enter your question:")
        if st.button("Generate", type="primary") and question:
            with st.spinner("Generating..."):
                result = st.session_state.engineer.zero_shot(question)
                st.text_area("Result:", result, height=200)
    
    elif technique == "Few-shot":
        st.write("Enter examples (one per line):")
        examples_text = st.text_area("Examples:", height=150)
        question = st.text_input("Enter your question:")
        if st.button("Generate", type="primary") and question and examples_text:
            examples = [ex.strip() for ex in examples_text.split('\n') if ex.strip()]
            with st.spinner("Generating..."):
                result = st.session_state.engineer.few_shot(examples, question)
                st.text_area("Result:", result, height=200)
    
    elif technique == "Chain-of-thought":
        problem = st.text_area("Enter the problem:", height=100)
        if st.button("Generate", type="primary") and problem:
            with st.spinner("Generating..."):
                result = st.session_state.engineer.chain_of_thought(problem)
                st.text_area("Result:", result, height=200)
    
    elif technique == "Role-based":
        role = st.text_input("Enter the role (e.g., 'doctor', 'teacher'):")
        task = st.text_area("Enter the task:", height=100)
        if st.button("Generate", type="primary") and role and task:
            with st.spinner("Generating..."):
                result = st.session_state.engineer.role_based(role, task)
                st.text_area("Result:", result, height=200)
    
    elif technique == "Template":
        template_type = st.selectbox("Template type:", ["summary", "explanation"])
        if template_type == "summary":
            text = st.text_area("Enter text to summarize:", height=150)
            if st.button("Generate", type="primary") and text:
                with st.spinner("Generating..."):
                    result = st.session_state.engineer.template_prompt("summary", text=text)
                    st.text_area("Result:", result, height=200)
        else:
            concept = st.text_input("Enter concept to explain:")
            if st.button("Generate", type="primary") and concept:
                with st.spinner("Generating..."):
                    result = st.session_state.engineer.template_prompt("explanation", concept=concept)
                    st.text_area("Result:", result, height=200)

def render_document_qa(module):
    """Render document Q&A interface"""
    st.subheader("Document Question & Answering")
    
    if 'qa_system' not in st.session_state:
        with st.spinner("Loading Q&A system... (this may take a minute on first run)"):
            try:
                qa_system = module.DocumentQA()
                st.session_state.qa_system = qa_system
                # Add sample document
                sample_doc = """
                Python is a high-level programming language known for its simplicity and readability.
                It was created by Guido van Rossum and first released in 1991.
                Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.
                """
                qa_system.add_document_text(sample_doc)
                qa_system.chunk_document()
                st.success("Q&A system ready! Sample document added.")
            except Exception as e:
                st.error(f"Error loading Q&A system: {e}")
                return
    
    tab1, tab2, tab3 = st.tabs(["Ask Question", "Add Document", "Load from File"])
    
    with tab1:
        question = st.text_input("Enter your question:")
        if st.button("Ask", type="primary") and question:
            with st.spinner("Searching documents..."):
                try:
                    answer, context = st.session_state.qa_system.answer_question(question)
                    st.write("**Answer:**")
                    st.write(answer)
                    st.write("**Context used:**")
                    st.write(context[:300] + "..." if len(context) > 300 else context)
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with tab2:
        doc_text = st.text_area("Enter document text:", height=200)
        if st.button("Add Document", type="primary") and doc_text:
            st.session_state.qa_system.add_document_text(doc_text)
            st.session_state.qa_system.chunk_document()
            st.success("Document added!")
    
    with tab3:
        file_path = st.text_input("Enter file path:")
        if st.button("Load File", type="primary") and file_path:
            if st.session_state.qa_system.load_document(file_path):
                st.session_state.qa_system.chunk_document()
                st.success("File loaded!")

def render_sentiment_analysis(module):
    """Render sentiment analysis interface"""
    st.subheader("Sentiment Analysis")
    
    if 'analyzer' not in st.session_state:
        with st.spinner("Loading sentiment analyzer... (this may take a minute on first run)"):
            try:
                analyzer = module.SentimentAnalyzer()
                st.session_state.analyzer = analyzer
                st.success("Sentiment analyzer ready!")
            except Exception as e:
                st.error(f"Error loading analyzer: {e}")
                return
    
    analysis_type = st.radio(
        "Analysis Type:",
        ["Sentiment", "Emotion", "Detailed"],
        horizontal=True
    )
    
    text = st.text_area("Enter text to analyze:", height=150)
    
    if st.button("Analyze", type="primary") and text:
        with st.spinner("Analyzing..."):
            try:
                if analysis_type == "Sentiment":
                    sentiment = st.session_state.analyzer.analyze_sentiment(text)
                    st.metric("Sentiment", sentiment)
                elif analysis_type == "Emotion":
                    emotion = st.session_state.analyzer.detect_emotion(text)
                    st.metric("Emotion", emotion)
                else:  # Detailed
                    analysis = st.session_state.analyzer.detailed_analysis(text)
                    col1, col2 = st.columns(2)
                    col1.metric("Sentiment", analysis['sentiment'])
                    col2.metric("Emotion", analysis['emotion'])
                    st.write("**Explanation:**")
                    st.write(analysis['explanation'])
            except Exception as e:
                st.error(f"Error: {e}")

def render_summarization(module):
    """Render summarization interface"""
    st.subheader("Text Summarization")
    
    if 'summarizer' not in st.session_state:
        with st.spinner("Loading summarizer... (this may take a minute on first run)"):
            try:
                summarizer = module.TextSummarizer()
                st.session_state.summarizer = summarizer
                st.success("Summarizer ready!")
            except Exception as e:
                st.error(f"Error loading summarizer: {e}")
                return
    
    summary_type = st.selectbox(
        "Summary Type:",
        ["Short", "Medium", "Long", "Extractive", "Bullet Points"]
    )
    
    text = st.text_area("Enter text to summarize:", height=200)
    
    if st.button("Summarize", type="primary") and text:
        with st.spinner("Summarizing..."):
            try:
                if summary_type == "Short":
                    summary = st.session_state.summarizer.summarize(text, max_length=50, summary_type="short")
                elif summary_type == "Medium":
                    summary = st.session_state.summarizer.summarize(text, max_length=100, summary_type="medium")
                elif summary_type == "Long":
                    summary = st.session_state.summarizer.summarize(text, max_length=200, summary_type="long")
                elif summary_type == "Extractive":
                    summary = st.session_state.summarizer.extractive_summary(text, num_sentences=3)
                else:  # Bullet Points
                    summary = st.session_state.summarizer.bullet_point_summary(text, num_points=5)
                
                st.text_area("Summary:", summary, height=200)
            except Exception as e:
                st.error(f"Error: {e}")

def render_ner(module):
    """Render NER interface"""
    st.subheader("Named Entity Recognition")
    
    if 'ner' not in st.session_state:
        with st.spinner("Loading NER system... (this may take a minute on first run)"):
            try:
                ner = module.NamedEntityRecognizer()
                st.session_state.ner = ner
                st.success("NER system ready!")
            except Exception as e:
                st.error(f"Error loading NER system: {e}")
                return
    
    text = st.text_area("Enter text to extract entities from:", height=150)
    output_format = st.radio("Output Format:", ["List", "JSON", "Structured"], horizontal=True)
    
    if st.button("Extract Entities", type="primary") and text:
        with st.spinner("Extracting entities..."):
            try:
                if output_format == "List":
                    entities = st.session_state.ner.extract_entities(text)
                    for entity_type, entity_list in entities.items():
                        if entity_list:
                            st.write(f"**{entity_type}:**")
                            st.write(", ".join(entity_list))
                elif output_format == "JSON":
                    json_output = st.session_state.ner.format_json(text)
                    st.code(json_output, language="json")
                else:  # Structured
                    structured = st.session_state.ner.extract_structured(text)
                    for entity in structured:
                        st.json(entity)
            except Exception as e:
                st.error(f"Error: {e}")

def render_rag(module):
    """Render RAG system interface"""
    st.subheader("RAG System (Retrieval Augmented Generation)")
    
    if 'rag' not in st.session_state:
        with st.spinner("Loading RAG system... (this may take a minute on first run)"):
            try:
                rag = module.RAGSystem()
                st.session_state.rag = rag
                # Add sample documents
                sample_docs = [
                    ("python_intro", "Python is a high-level programming language known for its simplicity."),
                    ("ai_basics", "Artificial Intelligence (AI) is the simulation of human intelligence by machines."),
                ]
                for doc_id, text in sample_docs:
                    rag.add_document(doc_id, text)
                st.success("RAG system ready! Sample documents added.")
            except Exception as e:
                st.error(f"Error loading RAG system: {e}")
                st.info("Make sure chromadb and sentence-transformers are installed: pip install chromadb sentence-transformers")
                return
    
    tab1, tab2, tab3 = st.tabs(["Query", "Add Document", "Load from File"])
    
    with tab1:
        question = st.text_input("Enter your question:")
        if st.button("Query", type="primary") and question:
            with st.spinner("Searching and generating..."):
                try:
                    result = st.session_state.rag.query(question)
                    st.write("**Answer:**")
                    st.write(result['answer'])
                    st.write("**Context used:**")
                    for i, ctx in enumerate(result['contexts'], 1):
                        st.write(f"{i}. {ctx[:200]}...")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with tab2:
        doc_id = st.text_input("Document ID:")
        doc_text = st.text_area("Document text:", height=200)
        if st.button("Add Document", type="primary") and doc_id and doc_text:
            st.session_state.rag.add_document(doc_id, doc_text)
            st.success("Document added!")
    
    with tab3:
        file_path = st.text_input("File path:")
        doc_id = st.text_input("Document ID:")
        if st.button("Load File", type="primary") and file_path and doc_id:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                st.session_state.rag.add_document(doc_id, text)
                st.success("File loaded!")
            except Exception as e:
                st.error(f"Error loading file: {e}")

def render_fine_tuning(module):
    """Render fine-tuning interface"""
    st.subheader("Fine-tuning")
    st.info("Fine-tuning requires significant computational resources. Please run the project directly for full functionality.")
    st.code("cd 10_fine_tuning\npython fine_tune.py", language="bash")

def render_multi_agent(module):
    """Render multi-agent system interface"""
    st.subheader("Multi-Agent System")
    
    if 'multi_agent' not in st.session_state:
        with st.spinner("Loading multi-agent system... (this may take a minute on first run)"):
            try:
                system = module.MultiAgentSystem()
                system.add_agent("alice", "researcher")
                system.add_agent("bob", "writer")
                system.add_agent("charlie", "analyzer")
                st.session_state.multi_agent = system
                st.success("Multi-agent system ready!")
            except Exception as e:
                st.error(f"Error loading multi-agent system: {e}")
                return
    
    task_type = st.radio("Task Type:", ["Coordinated Task", "Simple Task"], horizontal=True)
    
    if task_type == "Coordinated Task":
        task = st.text_area("Enter task description:", height=100)
        if st.button("Execute Task", type="primary") and task:
            with st.spinner("Coordinating task..."):
                try:
                    results = st.session_state.multi_agent.coordinate_task(task, ["alice", "bob", "charlie"])
                    for key, value in results.items():
                        st.write(f"**{key.upper()}:**")
                        st.write(value[:300] + "..." if len(value) > 300 else value)
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        agent_name = st.selectbox("Select agent:", ["alice", "bob", "charlie"])
        task = st.text_area("Enter task:", height=100)
        if st.button("Execute Task", type="primary") and task:
            with st.spinner("Executing task..."):
                try:
                    result = st.session_state.multi_agent.simple_task(task, agent_name)
                    st.write("**Result:**")
                    st.write(result)
                except Exception as e:
                    st.error(f"Error: {e}")

def render_function_calling(module):
    """Render function calling interface"""
    st.subheader("Function Calling")
    
    if 'function_llm' not in st.session_state:
        with st.spinner("Loading function calling system... (this may take a minute on first run)"):
            try:
                llm = module.FunctionCallingLLM()
                st.session_state.function_llm = llm
                st.success("Function calling system ready!")
            except Exception as e:
                st.error(f"Error loading function calling system: {e}")
                return
    
    st.write("**Available functions:** calculator, get_time, text_processor, web_search")
    st.write("**Examples:** 'Calculate 25 * 37', 'What time is it?', 'Make this uppercase: hello'")
    
    user_input = st.text_input("Enter your request:")
    
    if st.button("Process", type="primary") and user_input:
        with st.spinner("Processing..."):
            try:
                response = st.session_state.function_llm.process_with_functions(user_input)
                st.write("**Response:**")
                st.write(response)
            except Exception as e:
                st.error(f"Error: {e}")

# Main app
def main():
    st.markdown('<h1 class="main-header">🚀 LLM Projects - Unified Interface</h1>', unsafe_allow_html=True)
    
    # Sidebar for project selection
    with st.sidebar:
        st.header("Projects")
        
        # Project selection
        selected_project = st.selectbox(
            "Select a project:",
            options=list(PROJECTS.keys()),
            format_func=lambda x: f"{PROJECTS[x]['icon']} {PROJECTS[x]['name']}"
        )
        
        st.markdown("---")
        st.write(f"**{PROJECTS[selected_project]['description']}**")
        
        # Show project info
        project = PROJECTS[selected_project]
        folder = project["folder"]
        module_file = f"{project['module']}.py"
        module_path = os.path.join(folder, module_file)
        
        # Diagnostic info
        with st.expander("🔍 Project Diagnostics", expanded=False):
            st.write(f"**Current Directory:** `{os.getcwd()}`")
            st.write(f"**Folder:** `{folder}`")
            st.write(f"**Module:** `{module_file}`")
            st.write(f"**Full Path:** `{os.path.abspath(module_path)}`")
            
            folder_exists = os.path.exists(folder)
            file_exists = os.path.exists(module_path) if folder_exists else False
            
            st.write(f"**Folder exists:** {'✅ Yes' if folder_exists else '❌ No'}")
            st.write(f"**File exists:** {'✅ Yes' if file_exists else '❌ No'}")
            
            # Check dependencies
            st.write("\n**Dependencies Check:**")
            deps = ['transformers', 'torch', 'numpy']
            for dep in deps:
                try:
                    __import__(dep)
                    st.write(f"  ✅ {dep}")
                except ImportError:
                    st.write(f"  ❌ {dep} (not installed)")
            
            if not folder_exists:
                st.warning(f"⚠️ Project folder '{folder}' not found. Make sure you're in the correct directory.")
            elif not file_exists:
                st.warning(f"⚠️ Module file '{module_file}' not found in '{folder}'.")
        
        # Action buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Load Project", type="primary", use_container_width=True):
                with st.spinner("Loading project... (this may take a moment)"):
                    try:
                        module = load_module(selected_project)
                        if module:
                            st.session_state.active_project = selected_project
                            st.success("✅ Project loaded successfully!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to load project. Check the error message above for details.")
                    except Exception as e:
                        st.error(f"❌ Unexpected error: {str(e)}")
                        import traceback
                        with st.expander("🔍 Error Details"):
                            st.code(traceback.format_exc(), language="python")
        
        with col2:
            if st.button("Clear Cache", use_container_width=True):
                if selected_project in st.session_state.loaded_modules:
                    del st.session_state.loaded_modules[selected_project]
                if st.session_state.active_project == selected_project:
                    st.session_state.active_project = None
                st.success("Cache cleared!")
                st.rerun()
        
        # Show current status
        if selected_project in st.session_state.loaded_modules:
            st.info("✅ This project is already loaded and ready to use!")
        elif st.session_state.active_project == selected_project:
            st.warning("⚠️ Project was selected but module loading failed. Check errors above.")
    
    # Main content area
    if st.session_state.active_project:
        project_key = st.session_state.active_project
        module = st.session_state.loaded_modules.get(project_key)
        
        if module:
            # Route to appropriate renderer
            renderers = {
                "1": render_text_generation,
                "2": render_chatbot,
                "3": render_completion,
                "4": render_prompt_engineering,
                "5": render_document_qa,
                "6": render_sentiment_analysis,
                "7": render_summarization,
                "8": render_ner,
                "9": render_rag,
                "10": render_fine_tuning,
                "11": render_multi_agent,
                "12": render_function_calling
            }
            
            renderer = renderers.get(project_key)
            if renderer:
                renderer(module)
            else:
                st.error("Renderer not found for this project")
        else:
            st.warning("Please load the project first using the sidebar button.")
    else:
        st.info("👈 Select a project from the sidebar and click 'Load Project' to get started!")
        
        # Show project cards
        st.header("Available Projects")
        cols = st.columns(3)
        for idx, (key, project) in enumerate(PROJECTS.items()):
            with cols[idx % 3]:
                st.markdown(f"""
                <div class="project-card">
                    <h3>{project['icon']} {project['name']}</h3>
                    <p>{project['description']}</p>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

