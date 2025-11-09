"""
Multi-Agent System
Multiple AI agents working together to solve complex tasks.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Dict
import time

class Agent:
    def __init__(self, name: str, role: str, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """Initialize an agent"""
        self.name = name
        self.role = role
        self.model_name = model_name
        
        # Load model (in a real system, you might share models)
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
        
        self.memory = []  # Agent's memory/context
    
    def get_system_prompt(self):
        """Get system prompt based on role"""
        prompts = {
            "researcher": "You are a research agent. Your role is to gather information, analyze data, and provide research findings. Be thorough and accurate.",
            "writer": "You are a writing agent. Your role is to create well-written, engaging content based on information provided. Be creative and clear.",
            "analyzer": "You are an analysis agent. Your role is to review content, identify issues, and provide improvements. Be critical and constructive.",
            "coordinator": "You are a coordination agent. Your role is to manage tasks, delegate work, and ensure everything runs smoothly.",
        }
        return prompts.get(self.role.lower(), "You are a helpful AI agent.")
    
    def process(self, task: str, context: str = "") -> str:
        """Process a task and return result"""
        system_prompt = self.get_system_prompt()
        
        if context:
            prompt = f"""{system_prompt}

Context from other agents:
{context}

Your task: {task}

Response:"""
        else:
            prompt = f"""{system_prompt}

Your task: {task}

Response:"""
        
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + 150,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "Response:" in response:
            result = response.split("Response:")[-1].strip()
        else:
            result = response[len(prompt):].strip()
        
        # Store in memory
        self.memory.append({"task": task, "result": result})
        
        return result
    
    def communicate(self, message: str, to_agent=None):
        """Communicate with other agents"""
        return {
            "from": self.name,
            "to": to_agent,
            "message": message,
            "timestamp": time.time()
        }

class MultiAgentSystem:
    def __init__(self):
        """Initialize the multi-agent system"""
        print("Initializing multi-agent system...")
        self.agents = {}
        self.message_queue = []
        print("System ready!")
    
    def add_agent(self, name: str, role: str):
        """Add an agent to the system"""
        print(f"Adding agent: {name} ({role})...")
        agent = Agent(name, role)
        self.agents[name] = agent
        print(f"Agent {name} added!")
    
    def send_message(self, from_agent: str, to_agent: str, message: str):
        """Send a message between agents"""
        if from_agent in self.agents and to_agent in self.agents:
            msg = self.agents[from_agent].communicate(message, to_agent)
            self.message_queue.append(msg)
            return True
        return False
    
    def get_messages(self, agent_name: str) -> List[Dict]:
        """Get messages for an agent"""
        return [msg for msg in self.message_queue if msg.get("to") == agent_name]
    
    def coordinate_task(self, task: str, agents: List[str]) -> Dict:
        """Coordinate a task across multiple agents"""
        print(f"\nCoordinating task: {task}")
        print(f"Agents involved: {', '.join(agents)}")
        
        results = {}
        
        # Step 1: Researcher gathers information
        if "researcher" in [self.agents[a].role for a in agents if a in self.agents]:
            researcher_name = [a for a in agents if a in self.agents and self.agents[a].role == "researcher"][0]
            print(f"\n[{researcher_name}] Researching...")
            research_result = self.agents[researcher_name].process(f"Research information about: {task}")
            results["research"] = research_result
            print(f"[{researcher_name}] Research complete")
        
        # Step 2: Writer creates content
        if "writer" in [self.agents[a].role for a in agents if a in self.agents]:
            writer_name = [a for a in agents if a in self.agents and self.agents[a].role == "writer"][0]
            context = results.get("research", "")
            print(f"\n[{writer_name}] Writing...")
            writing_result = self.agents[writer_name].process(f"Write about: {task}", context=context)
            results["writing"] = writing_result
            print(f"[{writer_name}] Writing complete")
        
        # Step 3: Analyzer reviews
        if "analyzer" in [self.agents[a].role for a in agents if a in self.agents]:
            analyzer_name = [a for a in agents if a in self.agents and self.agents[a].role == "analyzer"][0]
            context = results.get("writing", results.get("research", ""))
            print(f"\n[{analyzer_name}] Analyzing...")
            analysis_result = self.agents[analyzer_name].process(f"Analyze and improve: {task}", context=context)
            results["analysis"] = analysis_result
            print(f"[{analyzer_name}] Analysis complete")
        
        return results
    
    def simple_task(self, task: str, agent_name: str) -> str:
        """Execute a simple task with one agent"""
        if agent_name in self.agents:
            print(f"\n[{agent_name}] Processing task...")
            result = self.agents[agent_name].process(task)
            print(f"[{agent_name}] Task complete")
            return result
        return "Agent not found"

def main():
    system = MultiAgentSystem()
    
    print("\n" + "="*50)
    print("Multi-Agent System")
    print("="*50)
    
    # Add agents
    print("\nSetting up agents...")
    system.add_agent("alice", "researcher")
    system.add_agent("bob", "writer")
    system.add_agent("charlie", "analyzer")
    
    print("\nCommands:")
    print("  - 'task <task_description>' - Coordinate a complex task")
    print("  - 'simple <agent_name> <task>' - Simple task with one agent")
    print("  - 'agents' - List all agents")
    print("  - 'quit' - Exit")
    print("\n")
    
    while True:
        user_input = input("> ").strip()
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
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
                print("\n" + "="*50)
                print("Results:")
                print("="*50)
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
            print("Unknown command. Try 'task', 'simple', 'agents', or 'quit'")

if __name__ == "__main__":
    main()


