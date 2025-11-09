# Setup Guide

## Quick Start

### 1. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Start with Basic Projects

Begin with Project 1:
```bash
cd 01_basic_text_generation
python text_generation.py
```

## Model Downloads

Models will be automatically downloaded on first use:
- **TinyLlama-1.1B** (~600MB) - Used in most projects
- **GPT-2 Small** (~500MB) - Used in Project 3
- **Phi-2** (~1.6GB) - Used in Project 4
- **Sentence Transformers** (~90MB) - Used in Project 9

Total download size: ~3GB (models are cached after first download)

## System Requirements

### Minimum Requirements
- **RAM**: 8GB (16GB recommended)
- **Storage**: 10GB free space (for models and dependencies)
- **Python**: 3.8 or higher

### GPU (Optional but Recommended)
- NVIDIA GPU with CUDA support
- CUDA 11.8 or higher
- Models will run on CPU, but GPU is much faster

## Troubleshooting

### Out of Memory Errors
- Use smaller batch sizes
- Close other applications
- Use CPU mode (slower but uses less memory)

### Slow Performance
- Enable GPU if available
- Use smaller models (TinyLlama)
- Reduce max_length parameters

### Model Download Issues
- Check internet connection
- Models download to `~/.cache/huggingface/`
- You can manually download models from Hugging Face

## Project-Specific Setup

### Project 9 (RAG System)
Requires additional setup:
```bash
pip install chromadb sentence-transformers
```

### Project 10 (Fine-tuning)
Requires additional setup:
```bash
pip install datasets
```

Note: Fine-tuning requires significant computational resources. Consider using cloud GPUs for training.

## Getting Help

1. Check the README in each project folder
2. Review error messages carefully
3. Ensure all dependencies are installed
4. Verify Python version (3.8+)

## Next Steps

1. Complete projects in order (1-12)
2. Experiment with different parameters
3. Modify code to suit your needs
4. Try combining techniques from different projects

Happy learning! 🚀


