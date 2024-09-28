from flask import Flask, render_template, request
import ollama

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def chat():
    response_message = ""
    if request.method == 'POST':
        user_input = request.form['user_input']
        response = ollama.chat(model='llama3.2', messages=[
            {
                'role': 'user',
                'content': user_input,
            },
        ])
        response_message = response['message']['content']
        print(response_message)
    return render_template('index1.html', response=response_message)

if __name__ == '__main__':
    app.run(debug=True)
