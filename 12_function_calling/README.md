# Project 12: Function Calling / Tool Use

## Overview
Build an LLM that can call functions/tools to perform actions beyond text generation.

## What You'll Learn
- Function calling with LLMs
- Tool integration
- Structured outputs
- Extending LLM capabilities
- API integrations

## Model Used
- **TinyLlama-1.1B-Chat** - Base model
- Custom function calling layer

## How to Run

```bash
cd 12_function_calling
python function_calling.py
```

## Features
- Define custom functions
- LLM decides when to call functions
- Execute functions and use results
- Chain multiple function calls
- Error handling

## Example
```
User: "What's the weather in New York?"
System: Calls weather API -> Returns result
User: "Calculate 25 * 37"
System: Calls calculator function -> Returns 925
```


