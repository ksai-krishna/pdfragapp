import pdfplumber
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# Step 1: Extract Text from PDF
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        full_text = ""
        for page in pdf.pages:
            full_text += page.extract_text()
    return full_text

# Step 2: Preprocess and Split Text into Documents
def split_text_into_chunks(text, chunk_size=500):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Load the PDF and split into documents
pdf_file = "input.pdf"
pdf_text = extract_text_from_pdf(pdf_file)
documents = split_text_into_chunks(pdf_text)

# Step 3: Convert Documents to Embeddings using Sentence Transformerss
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
document_vectors = embedding_model.encode(documents)

# Step 4: Create FAISS Index for Document Retrieval
index = faiss.IndexFlatL2(document_vectors.shape[1])  # L2 distance index
index.add(document_vectors)

# Step 5: Define a Retrieval Function
def retrieve_relevant_documents(query, k=3):
    query_vector = embedding_model.encode([query])
    distances, indices = index.search(query_vector, k)
    return [documents[i] for i in indices[0]]

# Step 6: Load Pre-Trained Language Model for Generation (GPT-Neo)
model_name = "EleutherAI/gpt-neo-1.3B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs['input_ids'], max_length=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Step 7: Combine Retrieval and Generation in RAG Pipeline
def rag_pipeline(user_query):
    relevant_docs = retrieve_relevant_documents(user_query)
    context = " ".join(relevant_docs)
    combined_input = f"Query: {user_query} \n Context: {context}"
    answer = generate_answer(combined_input)
    return answer

# Example Usage
if __name__ == '__main__':
    user_query = input("Enter your query: ")
    response = rag_pipeline(user_query)
    print("Answer:", response)
