"""
Named Entity Recognition (NER) System
Extract named entities from text.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import re

class NamedEntityRecognizer:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """Initialize the NER system"""
        print("Loading NER model...")
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
        
        self.entity_types = [
            "PERSON", "ORGANIZATION", "LOCATION", "DATE", 
            "MONEY", "PRODUCT", "EVENT", "OTHER"
        ]
        
        print("NER system ready!")
    
    def extract_entities(self, text):
        """Extract named entities from text"""
        entity_types_str = ", ".join(self.entity_types)
        
        prompt = f"""Extract all named entities from the following text. For each entity, identify its type from this list: {entity_types_str}

Format your response as: ENTITY_TYPE: entity_name (one per line)

Text: "{text}"

Entities:"""
        
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + 100,
                temperature=0.5,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        entities_text = response.split("Entities:")[-1].strip()
        
        # Parse entities
        entities = self._parse_entities(entities_text)
        
        # Also use simple pattern matching as fallback
        pattern_entities = self._pattern_based_ner(text)
        
        # Combine results
        combined = self._merge_entities(entities, pattern_entities)
        
        return combined
    
    def _parse_entities(self, entities_text):
        """Parse entities from model response"""
        entities = {}
        lines = entities_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    entity_type = parts[0].strip().upper()
                    entity_name = parts[1].strip()
                    
                    # Clean entity name
                    entity_name = re.sub(r'[^\w\s-]', '', entity_name)
                    
                    if entity_type in self.entity_types and entity_name:
                        if entity_type not in entities:
                            entities[entity_type] = []
                        if entity_name not in entities[entity_type]:
                            entities[entity_type].append(entity_name)
        
        return entities
    
    def _pattern_based_ner(self, text):
        """Simple pattern-based NER as fallback"""
        entities = {}
        
        # Dates (simple patterns)
        date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}\b'
        dates = re.findall(date_pattern, text)
        if dates:
            entities['DATE'] = dates
        
        # Money
        money_pattern = r'\$[\d,]+(?:\.\d{2})?|\d+\s*(?:dollars|USD|EUR|pounds)'
        money = re.findall(money_pattern, text, re.IGNORECASE)
        if money:
            entities['MONEY'] = money
        
        # Capitalized words (potential entities)
        # This is very basic and can be improved
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        if capitalized:
            # Filter out common words
            common_words = {'The', 'This', 'That', 'These', 'Those', 'A', 'An'}
            filtered = [w for w in capitalized if w not in common_words]
            if filtered:
                entities['OTHER'] = filtered[:10]  # Limit to 10
        
        return entities
    
    def _merge_entities(self, entities1, entities2):
        """Merge two entity dictionaries"""
        merged = entities1.copy()
        
        for entity_type, entity_list in entities2.items():
            if entity_type not in merged:
                merged[entity_type] = []
            for entity in entity_list:
                if entity not in merged[entity_type]:
                    merged[entity_type].append(entity)
        
        return merged
    
    def extract_structured(self, text):
        """Extract entities in structured format"""
        entities = self.extract_entities(text)
        
        structured = []
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                structured.append({
                    'entity': entity,
                    'type': entity_type,
                    'text': text
                })
        
        return structured
    
    def format_json(self, text):
        """Extract entities and format as JSON"""
        entities = self.extract_entities(text)
        return json.dumps(entities, indent=2)
    
    def extract_by_type(self, text, entity_type):
        """Extract only specific entity type"""
        entities = self.extract_entities(text)
        return entities.get(entity_type.upper(), [])

def main():
    ner = NamedEntityRecognizer()
    
    print("\n" + "="*50)
    print("Named Entity Recognition System")
    print("="*50)
    print("\nCommands:")
    print("  - 'extract <text>' - Extract all entities")
    print("  - 'json <text>' - Get entities as JSON")
    print("  - 'type <entity_type> <text>' - Extract specific type")
    print("  - 'structured <text>' - Get structured format")
    print("  - 'quit' - Exit")
    print(f"\nEntity types: {', '.join(ner.entity_types)}")
    print("\n")
    
    while True:
        user_input = input("> ").strip()
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
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
            print("Unknown command. Type 'quit' to exit.")

if __name__ == "__main__":
    main()


