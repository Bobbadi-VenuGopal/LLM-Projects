# Unified Interface Guide

## Overview

The `main_interface.py` provides a single, easy-to-use interface to interact with all 12 LLM projects. No need to navigate between folders or remember different command patterns!

## How to Use

### Starting the Interface

```bash
python main_interface.py
```

### Main Menu

When you start the interface, you'll see:

```
🚀 LLM Projects - Unified Interface
======================================================================

Available Projects:

  1. ✅ Basic Text Generation
     Generate text from prompts using TinyLlama

  2. ✅ Simple Chatbot
     Chat with an AI that maintains conversation history

  ... (and 10 more projects)

Commands:
  - Enter project number (1-12) to use that project
  - 'list' - Show this menu again
  - 'help' - Show help for current project
  - 'back' - Return to main menu (when in a project)
  - 'quit' - Exit the program
======================================================================
```

### Using Projects

1. **Select a Project**: Enter the project number (1-12)
2. **Interact**: Use the project-specific commands
3. **Return to Menu**: Type `back` to return to the main menu
4. **Switch Projects**: Select a different project number from the menu

## Project-Specific Guides

### 1. Basic Text Generation
- **Usage**: Enter a prompt to generate text
- **Commands**: Just type your prompt
- **Example**: "Write a short story about space"

### 2. Simple Chatbot
- **Usage**: Chat with the AI
- **Commands**: 
  - Type your message to chat
  - `clear` - Clear conversation history
- **Example**: "Hello! What's your name?"

### 3. Text Completion
- **Usage**: Complete partial text
- **Commands**: 
  - Enter text to complete
  - Choose mode: `single`, `multi`, or `next`
- **Example**: "The weather today is" (mode: single)

### 4. Prompt Engineering
- **Usage**: Practice different prompting techniques
- **Commands**: 
  - Select technique (1-5)
  - Follow prompts for each technique
- **Example**: Zero-shot, Few-shot, Chain-of-thought

### 5. Document Q&A
- **Usage**: Ask questions about documents
- **Commands**: 
  - `load <file_path>` - Load a document
  - `add` - Add text directly
  - `ask <question>` - Ask a question
- **Example**: `ask What is Python?`

### 6. Sentiment Analysis
- **Usage**: Analyze sentiment and emotions
- **Commands**: 
  - `sentiment <text>` - Analyze sentiment
  - `emotion <text>` - Detect emotion
  - `detailed <text>` - Full analysis
- **Example**: `sentiment I love this product!`

### 7. Text Summarization
- **Usage**: Summarize text
- **Commands**: 
  - `short <text>` - Short summary
  - `medium <text>` - Medium summary
  - `long <text>` - Long summary
  - `extract <text>` - Extractive summary
  - `bullets <text>` - Bullet points
- **Example**: `short [long text here]`

### 8. Named Entity Recognition
- **Usage**: Extract entities from text
- **Commands**: 
  - `extract <text>` - Extract all entities
  - `json <text>` - Get JSON format
  - `type <entity_type> <text>` - Extract specific type
- **Example**: `extract Apple Inc. was founded by Steve Jobs`

### 9. RAG System
- **Usage**: Ask questions using RAG
- **Commands**: 
  - `add <doc_id> <text>` - Add document
  - `query <question>` - Ask a question
  - `load <file_path> <doc_id>` - Load from file
- **Example**: `query What is artificial intelligence?`

### 10. Fine-tuning
- **Usage**: Fine-tune models (advanced)
- **Note**: This project requires running directly
- **See**: Project 10 README for details

### 11. Multi-Agent System
- **Usage**: Coordinate multiple AI agents
- **Commands**: 
  - `task <description>` - Coordinate complex task
  - `simple <agent_name> <task>` - Simple task
  - `agents` - List agents
- **Example**: `task Research and write about AI`

### 12. Function Calling
- **Usage**: Use LLM with function calling
- **Commands**: Just ask naturally
- **Examples**: 
  - "Calculate 25 * 37"
  - "What time is it?"
  - "Make this uppercase: hello world"

## Tips & Tricks

### Navigation
- **Quick Switch**: Type `back` to return to menu, then select new project
- **Help**: Type `help` while in a project for assistance
- **List**: Type `list` to see all projects again

### First Run
- Models download automatically on first use
- First run may take a few minutes
- Subsequent runs are much faster

### Memory Management
- Models stay loaded in memory
- Switch projects to free up resources if needed
- Restart the interface to fully clear memory

### Error Handling
- If a project fails to load, check that dependencies are installed
- Some projects require additional packages (see project READMEs)
- Check error messages for specific issues

## Common Commands

| Command | Description |
|---------|-------------|
| `1-12` | Select a project |
| `list` | Show main menu |
| `help` | Show help |
| `back` | Return to main menu |
| `quit` | Exit program |

## Troubleshooting

### Project Not Loading
- Check that the project folder exists
- Verify Python version (3.8+)
- Install missing dependencies: `pip install -r requirements.txt`

### Model Download Issues
- Check internet connection
- Models download to `~/.cache/huggingface/`
- First download may take time

### Memory Issues
- Close other applications
- Use smaller models (default)
- Restart the interface periodically

### Import Errors
- Make sure you're in the project root directory
- Install all dependencies: `pip install -r requirements.txt`
- Check that project files exist

## Advanced Usage

### Customizing Projects
- Projects can be modified directly in their folders
- Changes will be reflected in the interface
- Restart interface after major changes

### Extending the Interface
- Add new projects by updating `main_interface.py`
- Follow the existing project handler pattern
- Add project info to the `projects` dictionary

## Support

For issues or questions:
1. Check the project-specific README
2. Review error messages
3. Verify dependencies are installed
4. Check Python version compatibility

## Enjoy!

The unified interface makes it easy to explore all LLM projects. Have fun learning! 🚀


