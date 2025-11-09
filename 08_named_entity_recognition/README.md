# Project 8: Named Entity Recognition (NER)

## Overview
Extract named entities (people, places, organizations, etc.) from text using LLMs.

## What You'll Learn
- Entity extraction
- Entity classification
- Structured data extraction
- Information extraction techniques

## Model Used
- **TinyLlama-1.1B-Chat** - For NER tasks

## How to Run

```bash
cd 08_named_entity_recognition
python ner.py
```

## Features
- Extract people, places, organizations
- Extract dates, locations, products
- Format output as JSON
- Batch processing

## Example
```
Text: "Apple Inc. was founded by Steve Jobs in Cupertino, California in 1976."
Entities:
- Organization: Apple Inc.
- Person: Steve Jobs
- Location: Cupertino, California
- Date: 1976
```


