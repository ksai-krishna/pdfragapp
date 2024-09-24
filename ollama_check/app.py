from flask import Flask, request, jsonify, render_template
import subprocess
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def query_ollama_model(prompt):
    try:
        # Log the query being sent to the model
        logging.debug(f"Running command: ['ollama', 'run', 'llama2', '{prompt}']")
        
        # Execute the command and capture output
        result = subprocess.run(['ollama', 'run', 'llama2', prompt], capture_output=True, text=True)
        
        # Check if the model returned an error
        if result.returncode != 0:
            logging.error(f"Error running model: {result.stderr.strip()}")
            return "An error occurred while querying the model."
        
        # Return the model's output
        return result.stdout.strip()
    except Exception as e:
        logging.error(f"Exception when querying model: {e}")
        return "An error occurred while querying the model."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        user_query = data['query']
        logging.debug(f"Received query: {user_query}")

        # Query the Ollama model
        answer = query_ollama_model(user_query)
        logging.debug(f"Model response: {answer}")

        return jsonify({'response': answer})
    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({'response': f"An error occurred while processing your request: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
