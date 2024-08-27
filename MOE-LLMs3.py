import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Model configuration
MODEL_CONFIG = {
    "director": {
        "name": "Agnuxo/Qwen2-1.5B-Instruct_MOE_Director_16bit",
        "task": "text-generation",
    },
    "programming": {
        "name": "Qwen/Qwen2-1.5B-Instruct",
        "task": "text-generation",
    },
    "biology": {
        "name": "Agnuxo/Qwen2-1.5B-Instruct_MOE_BIOLOGY_assistant_16bit",
        "task": "text-generation",
    },
    "mathematics": {
        "name": "Qwen/Qwen2-Math-1.5B-Instruct",
        "task": "text-generation",
    }
}

# Keywords for each subject
KEYWORDS = {
    "biology": ["cell", "DNA", "protein", "evolution", "genetics", "ecosystem", "organism", "metabolism", "photosynthesis", "microbiology", "célula", "ADN", "proteína", "evolución", "genética", "ecosistema", "organismo", "metabolismo", "fotosíntesis", "microbiología"],
    "mathematics": ["Math" "mathematics", "equation", "integral", "derivative", "function", "geometry", "algebra", "statistics", "probability", "ecuación", "integral", "derivada", "función", "geometría", "álgebra", "estadística", "probabilidad"],
    "programming": ["python", "java", "C++", "HTML", "scrip", "code", "Dataset", "API", "framework", "debugging", "algorithm", "compiler", "database", "CSS", "JSON", "XML", "encryption", "IDE", "repository", "Git", "version control", "front-end", "back-end", "API", "stack trace", "REST", "machine learning"]
}

class MOELLM:
    def __init__(self):
        self.current_expert = None
        self.current_model = None
        self.current_tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.load_director_model()

    def load_director_model(self):
        """Loads the director model."""
        print("Loading director model...")
        model_name = MODEL_CONFIG["director"]["name"]
        self.director_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.director_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(self.device)
        self.director_pipeline = pipeline(
            MODEL_CONFIG["director"]["task"],
            model=self.director_model,
            tokenizer=self.director_tokenizer,
            device=self.device
        )
        print("Director model loaded.")

    def load_expert_model(self, expert):
        """Dynamically loads an expert model, releasing memory from the previous model."""
        if expert not in MODEL_CONFIG:
            raise ValueError(f"Unknown expert: {expert}")

        if self.current_expert != expert:
            print(f"Loading expert model: {expert}...")
            
            # Free memory from the current model if it exists
            if self.current_model:
                del self.current_model
                del self.current_tokenizer
                torch.cuda.empty_cache()
            
            model_config = MODEL_CONFIG[expert]
            self.current_tokenizer = AutoTokenizer.from_pretrained(model_config["name"])
            self.current_model = AutoModelForCausalLM.from_pretrained(model_config["name"], torch_dtype=torch.float16).to(self.device)
            self.current_expert = expert
            
            print(f"{expert.capitalize()} model loaded.")
        
        return pipeline(
            MODEL_CONFIG[expert]["task"],
            model=self.current_model,
            tokenizer=self.current_tokenizer,
            device=self.device
        )

    def determine_expert_by_keywords(self, question):
        """Determines the expert based on keywords in the question."""
        question_lower = question.lower()
        for expert, keywords in KEYWORDS.items():
            if any(keyword in question_lower for keyword in keywords):
                return expert
        return None

    def determine_expert(self, question):
        """Determines which expert should answer the question."""
        expert = self.determine_expert_by_keywords(question)
        if expert:
            print(f"Expert determined by keyword: {expert}")
            return expert

        prompt = f"Classify the following question into one of these categories: programming, biology, mathematics. Question: {question}\nCategory:"
        response = self.director_pipeline(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
        expert = response.split(":")[-1].strip().lower()
        if expert not in MODEL_CONFIG:
            expert = "director"
        print(f"Redirecting question to: {expert}")
        return expert

    def generate_response(self, question, expert):
        """Generates a response using the appropriate model."""
        try:
            model = self.load_expert_model(expert)
            prompt = f"Answer the following question as an expert in {expert}: {question}\nAnswer:"
            response = model(prompt, max_length=200, num_return_sequences=1)[0]['generated_text']
            return response.split("Answer:")[-1].strip()
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "Sorry, there was an error processing your request. Please try again."

    def chat_interface(self):
        """Simple chat interface."""
        print("Welcome to the MOE-LLM chat. Type 'exit' to quit.")
        while True:
            question = input("\nYou: ")
            if question.lower() in ['exit', 'quit']:
                break
            
            try:
                expert = self.determine_expert(question)
                response = self.generate_response(question, expert)
                print(f"\n{expert.capitalize()}: {response}")
            except Exception as e:
                print(f"Error in chat: {str(e)}")
                print("Please try asking another question.")

if __name__ == "__main__":
    moe_llm = MOELLM()
    moe_llm.chat_interface()










