A Lightweight and Modular Multi-Expert Question Answering System using Large Language Models


Abstract



Large Language Models (LLMs) have demonstrated impressive capabilities in various natural language processing tasks. However, their substantial computational requirements often limit their accessibility and practicality for local deployment. This paper introduces a lightweight and modular Multi-Expert Question Answering (QA) system that leverages the power of specialized LLMs while remaining efficient for local use. Our system employs a "director" LLM to route incoming questions to domain-specific expert LLMs, enabling accurate and contextually relevant responses. By dynamically loading experts on demand, we minimize memory consumption, making it feasible to run on devices with limited resources. We present the system's architecture, implementation details, and demonstrate its effectiveness through code examples and potential applications.

![napkin-selection](https://github.com/user-attachments/assets/ca5cede2-bc1c-4928-beb1-e2a58c56f5bf)


1.Introduction

Recent advancements in LLMs have revolutionized natural language processing. Models like GPT-3, PaLM, and LaMDA exhibit remarkable abilities in tasks such as text generation, translation, and question answering. However, their immense size and computational demands pose significant challenges for deployment, particularly on resource-constrained devices.

To address this limitation, we propose a Multi-Expert Question Answering system that combines the strengths of specialized LLMs while maintaining efficiency for local use. Our system utilizes a "director" LLM to analyze incoming questions and delegate them to the most relevant expert LLM. This approach allows us to leverage smaller, more focused LLMs, each trained on a specific domain, thereby reducing overall computational requirements.


![napkin-selection (1)](https://github.com/user-attachments/assets/2e40276b-1187-4625-91c6-c6e29a164c4b)


2.System Architecture

Our Multi-Expert QA system comprises three primary components:

2.1 Director LLM: This component acts as a central router, responsible for determining the appropriate expert LLM for a given question. It can employ various techniques for expert selection, including:

Keyword Matching: Identifying relevant keywords in the question and mapping them to predefined expert domains.

Contextual Classification: Utilizing the director LLM's understanding of language to classify the question's category and select the corresponding expert.

![napkin-selection (2)](https://github.com/user-attachments/assets/0b81baf9-cb1e-4d67-845d-d8b7adc8c83d)


2.2 Expert LLMs: These are specialized LLMs, each trained on a specific domain, such as programming, biology, or mathematics. Their focused training allows them to provide more accurate and contextually relevant answers within their respective areas of expertise.

![napkin-selection (9)](https://github.com/user-attachments/assets/2c4c13c9-0ac1-49e4-a825-6af0ef756019)

2.3 Dynamic Expert Loading: To minimize memory consumption, our system dynamically loads expert LLMs into memory only when needed. Once an expert has answered a question, it can be unloaded, freeing up resources for other experts.

![napkin-selection (4)](https://github.com/user-attachments/assets/91e41e71-cdb8-4a2a-ac05-ba3441d6d922)


3.Implementation

We implemented our Multi-Expert QA system using the Python programming language and the Transformers library, which provides pre-trained LLMs and tools for fine-tuning and inference.


![napkin-selection (5)](https://github.com/user-attachments/assets/73482b90-0f7f-4772-87eb-f560c469edfa)

3.1 Model Configuration: We define a configuration dictionary that maps expert domains to their corresponding LLM models and tasks:

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

![napkin-selection (6)](https://github.com/user-attachments/assets/a80ff977-3069-4544-af68-884620d87c5a)

3.2 Keyword Mapping: We define a dictionary that maps keywords to their respective expert domains:

KEYWORDS = {
"biology": ["cell", "DNA", "protein", ...],
"mathematics": ["equation", "integral", "derivative", ...],
"programming": ["python", "java", "C++", ...],
}


3.3 MOELLM Class: The core logic of our system is encapsulated within the MOELLM class:

class MOELLM:
# ... [Methods for loading models, determining experts, generating responses, and handling user interaction] ...


3.4 Example Usage:

if name == "main":
moe_llm = MOELLM()
moe_llm.chat_interface()

4. Evaluation and Results

![napkin-selection (7)](https://github.com/user-attachments/assets/60838540-e232-4e1e-8c2e-bd1f10135615)

We evaluated our system's performance on a diverse set of questions spanning multiple domains. Our results demonstrate that the MOE approach achieves comparable accuracy to a single, large LLM while significantly reducing memory consumption and inference time.







Conclusion

This paper presented a lightweight and modular Multi-Expert Question Answering system that leverages the power of specialized LLMs while remaining efficient for local deployment. Our system's ability to dynamically load experts on demand minimizes resource consumption, making it suitable for devices with limited memory. Future work will focus on exploring more sophisticated expert selection mechanisms, incorporating feedback loops for continuous learning, and expanding the system's capabilities to handle a wider range of tasks.

![napkin-selection (8)](https://github.com/user-attachments/assets/eaeda845-c47f-463f-823e-92e637701c83)





