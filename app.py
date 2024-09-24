from flask import Flask, render_template, request, jsonify
import PyPDF2
import subprocess
import numpy as np
import faiss
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

documents = []

def extract_text_from_pdf(pdf_path):
    try:
        text = ''
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + ' '
        return text
    except Exception as e:
        logging.error(f"Error extracting text from {pdf_path}: {e}")
        return ""

def load_pdfs(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    if text:  # Only add non-empty documents
        documents.append(text)
    else:
        logging.warning(f"No text extracted from {pdf_path}")

# Use only input.pdf as the document
pdf_path = 'input.pdf'
load_pdfs(pdf_path)

# Initialize FAISS index with dummy embeddings
embeddings_np = np.random.rand(len(documents), 768).astype(np.float32)  # Replace with actual embeddings
index = faiss.IndexFlatL2(embeddings_np.shape[1])
index.add(embeddings_np)

@app.route('/')
def index_view():
    return render_template('index.html')

def query_ollama_model(prompt):
    try:
        result = subprocess.run(['ollama', 'run', 'llama2', prompt], capture_output=True, text=True)
        if result.returncode != 0:
            logging.error(f"Error running model: {result.stderr}")
            return "An error occurred while querying the model."
        return result.stdout.strip()
    except Exception as e:
        logging.error(f"Exception when querying model: {e}")
        return "An error occurred while querying the model."

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        user_query = data['query']
        logging.debug(f"Received query: {user_query}")

        query_embedding = np.random.rand(1, 768).astype(np.float32)  # Replace with actual embedding for the query
        distances, indices = index.search(query_embedding, 5)

        # Check if we received valid indices
        if indices.size == 0:
            return jsonify({'response': "No relevant documents found."}), 404

        retrieved_docs = [documents[i] for i in indices[0]]
        context = " ".join(retrieved_docs)

        prompt = f"Context: {context}\nQuestion: {user_query}\nAnswer:"
        answer = query_ollama_model(prompt)
        logging.debug(f"Model response: {answer}")

        return jsonify({'response': answer})

    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({'response': "An error occurred while processing your request."}), 500

if __name__ == '__main__':
    app.run(debug=True)
