from flask import Flask, request, render_template_string
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Load the Pythia conversational model
try:
    logging.info("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/pythia-6.9b")  # Use appropriate Pythia model
    model = AutoModelForCausalLM.from_pretrained("facebook/pythia-6.9b")
    logging.info("Model and tokenizer loaded successfully.")
except Exception as e:
    logging.error("Error loading model: %s", e)
    raise

@app.route("/", methods=["GET"])
def home():
    return render_template_string("""
    <html>
        <head>
            <title>Chatbot</title>
            <style>
                body { font-family: Arial, sans-serif; }
                input { width: 300px; }
                button { padding: 10px; }
                h1 { font-size: 24px; }
            </style>
        </head>
        <body>
            <h1>Chatbot</h1>
            <form action="/chat" method="post">
                <input type="text" name="user_input" placeholder="Type your question here..." required>
                <button type="submit">Send</button>
            </form>
        </body>
    </html>
    """)

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_input = request.form['user_input']
        
        # Prepare the input for the model
        logging.info("User input: %s", user_input)
        inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

        # Generate a response
        reply_ids = model.generate(inputs, max_length=200, pad_token_id=tokenizer.eos_token_id)
        response_message = tokenizer.decode(reply_ids[0], skip_special_tokens=True)

        logging.info("Generated response: %s", response_message)

        return render_template_string(f"""
        <html>
            <head>
                <title>Chatbot</title>
                <style>
                    body {{ font-family: Arial, sans-serif; }}
                    h1 {{ font-size: 24px; }}
                    a {{ margin-top: 20px; display: inline-block; }}
                </style>
            </head>
            <body>
                <h1>Response: {response_message}</h1>
                <a href="/">Back</a>
            </body>
        </html>
        """)
    except Exception as e:
        # Log the error traceback
        logging.error("Error during request handling: %s", traceback.format_exc())
        return "<h1>Internal Server Error</h1><p>Something went wrong!</p>", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
