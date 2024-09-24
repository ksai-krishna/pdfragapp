# Install required libraries if not already installed
# !pip install transformers torch sacremoses

from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForCausalLM

# Load PubMedBERT model and tokenizer for medical text understanding
pubmed_tokenizer = BertTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
pubmed_model = BertModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")

# Load BioGPT model and tokenizer for medical Q&A generation
biogpt_tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT-Large")
biogpt_model = AutoModelForCausalLM.from_pretrained("microsoft/BioGPT-Large")

# Sample medical text (you can replace this with any medical document)
medical_text = """
Aspirin is commonly used to reduce pain, fever, or inflammation. It is also used to prevent heart attacks, strokes, and chest pain (angina) in people with heart disease.
Aspirin belongs to a group of drugs called salicylates and works by reducing substances in the body that cause pain and inflammation.
"""

# Sample questions
questions = [
    "What is Aspirin used for?",
    "How does Aspirin work?",
    "What conditions does Aspirin treat?"
]

# Step 1: Tokenize the medical text using PubMedBERT
inputs = pubmed_tokenizer(medical_text, return_tensors="pt")

# Extract features from the medical text using PubMedBERT (this step typically provides embeddings)
outputs = pubmed_model(**inputs)

# Step 2: Use BioGPT to generate answers for the sample questions
for question in questions:
    print(f"Question: {question}")
    
    # Concatenate the question with the medical text to provide context
    biogpt_input = medical_text + " " + question
    inputs = biogpt_tokenizer(biogpt_input, return_tensors="pt")
    
    # Generate answer using BioGPT
    outputs = biogpt_model.generate(**inputs, max_length=100)
    
    # Decode the output and print the answer
    answer = biogpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Answer: {answer}")
    print()
