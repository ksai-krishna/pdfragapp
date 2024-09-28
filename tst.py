import ollama
response = ollama.chat(model='llama3.2', messages=[
  {
    'role': 'user',
    'content': 'dony say hi',
  },
])
print(response['message']['content'])