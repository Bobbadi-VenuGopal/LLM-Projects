# Streamlit Interface Guide

## Overview

The Streamlit interface provides a beautiful web-based UI for all 12 LLM projects. No command-line needed!

## Installation

Make sure Streamlit is installed:

```bash
pip install streamlit
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

## Running the Interface

```bash
streamlit run streamlit_app.py
```

This will:
1. Start a local web server
2. Open your browser automatically
3. Display the interface at `http://localhost:8501`

## Features

### 🎨 Beautiful Web Interface
- Clean, modern UI
- Interactive widgets
- Real-time updates
- No terminal needed!

### 📱 Easy Navigation
- Sidebar for project selection
- One-click project loading
- Clear project descriptions

### ⚡ Interactive Components
- Text areas for input
- Sliders for parameters
- Buttons for actions
- Tabs for organization
- Chat interface for chatbots

## Using the Interface

### Step 1: Select a Project
- Use the sidebar dropdown to select a project
- See the project description

### Step 2: Load the Project
- Click "Load Project" button
- Wait for the model to load (first time may take a minute)
- You'll see a success message

### Step 3: Use the Project
- Follow the on-screen instructions
- Enter your input
- Click the action button
- View results in real-time

## Project-Specific Features

### Text Generation
- Text area for prompts
- Sliders for max_length and temperature
- Real-time generation

### Chatbot
- Chat interface (like ChatGPT)
- Message history
- Clear history button

### Text Completion
- Multiple modes (Single, Multiple, Next Words)
- Radio buttons for selection

### Prompt Engineering
- Dropdown for technique selection
- Different inputs for each technique
- Template examples

### Document Q&A
- Tabs for different actions
- File upload support
- Context display

### Sentiment Analysis
- Radio buttons for analysis type
- Metrics display
- Detailed explanations

### Summarization
- Dropdown for summary type
- Large text areas
- Multiple output formats

### NER
- Output format selection
- JSON code blocks
- Structured display

### RAG System
- Tabbed interface
- Document management
- Query interface

### Multi-Agent
- Task type selection
- Agent selection dropdown
- Coordinated task execution

### Function Calling
- Natural language input
- Function examples
- Tool execution display

## Tips

1. **First Run**: Models download automatically - be patient!
2. **Memory**: Models stay loaded in session - refresh to clear
3. **Errors**: Check the error messages - they're usually helpful
4. **Parameters**: Adjust sliders to see different results
5. **History**: Chat history persists in the session

## Troubleshooting

### Port Already in Use
```bash
streamlit run streamlit_app.py --server.port 8502
```

### Model Loading Issues
- Check internet connection
- Ensure dependencies are installed
- Check error messages

### Memory Issues
- Refresh the page to clear models
- Close other applications
- Use smaller models

### Display Issues
- Clear browser cache
- Try a different browser
- Check Streamlit version

## Keyboard Shortcuts

- `R` - Rerun the app
- `C` - Clear cache
- `?` - Show keyboard shortcuts

## Advanced Usage

### Custom Port
```bash
streamlit run streamlit_app.py --server.port 8080
```

### Custom Theme
Create `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
```

### Sharing
- Use Streamlit Cloud for free hosting
- Or deploy to your own server
- Share the URL with others

## Comparison: CLI vs Streamlit

| Feature | CLI Interface | Streamlit Interface |
|---------|--------------|-------------------|
| Ease of Use | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Visual Appeal | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Interactivity | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Setup | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Resource Usage | ⭐⭐⭐⭐ | ⭐⭐⭐ |

## Next Steps

1. Try all 12 projects
2. Experiment with parameters
3. Build your own projects
4. Share with others!

Enjoy the beautiful interface! 🚀

