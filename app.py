from flask import Flask, render_template, request, jsonify
import PyPDF2
import subprocess
import numpy as np
import faiss

app = Flask(__name__)

# List to hold the extracted document texts
documents = []

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    text = ''
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + ' '  # Adding space between pages
    return text

# Function to load PDFs into the documents list
def load_pdfs(pdf_paths):
    for pdf_path in pdf_paths:
        text = extract_text_from_pdf(pdf_path)
        documents.append(text)

# Load your PDF files here (adjust paths accordingly)
pdf_paths = ['input.pdf']
load_pdfs(pdf_paths)

# Create FAISS index (this assumes you have embeddings prepared)
# You will need to create or replace this with actual embeddings from your documents
embeddings_np = np.random.rand(len(documents), 768).astype(np.float32)  # Replace with actual embeddings
index = faiss.IndexFlatL2(embeddings_np.shape[1])
index.add(embeddings_np)

# Define the main route to render the index page
@app.route('/')
def index_view():
    return render_template('index.html')

# Function to query the Ollama model
def query_ollama_model(prompt):
    """Query the Ollama model using subprocess."""
    result = subprocess.run(['ollama', 'run', 'llama2', prompt], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running model: {result.stderr}")
        return "An error occurred while querying the model."
    return result.stdout.strip()

# Route to handle user queries
@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        user_query = data['query']

        # Perform FAISS search here to get relevant documents
        query_embedding = np.random.rand(1, 768).astype(np.float32)  # Replace with actual embedding for the query
        distances, indices = index.search(query_embedding, 5)  # Retrieve top 5 documents
        retrieved_docs = [documents[i] for i in indices[0]]
        context = " ".join(retrieved_docs)

        # Query the Ollama model with context and user query
        prompt = f"Context: {context}\nQuestion: {user_query}\nAnswer:"
        answer = query_ollama_model(prompt)

        return jsonify({'response': answer})

    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({'response': "An error occurred while processing your request."}), 500

if __name__ == '__main__':
    app.run(debug=True)
