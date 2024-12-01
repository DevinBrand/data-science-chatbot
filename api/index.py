from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
import openai
from pinecone import Pinecone

load_dotenv()

app = Flask(__name__)

# Initialize OpenAI and Pinecone
openai.api_key = os.getenv("OPENAI_API_KEY")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("INDEX_NAME"))

# Store chat history as a global variable
global_chat_history = []


def get_embedding(text):
    response = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding


def query_pinecone(embedding, top_k=3):
    results = index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True
    )
    return results


def generate_response(query, context, history):
    messages = [
        {"role": "system",
         "content": "You are a helpful assistant. Use the following context to answer the question. If the context doesn't contain relevant information, say so: " + context},
    ]

    # Add chat history
    for past_query, past_response in history[-3:]:
        messages.extend([
            {"role": "user", "content": past_query},
            {"role": "assistant", "content": past_response}
        ])

    messages.append({"role": "user", "content": query})

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0
    )

    return response.choices[0].message.content


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.json
        user_message = data['message']
        client_chat_history = data.get('chatHistory', [])

        # Get embedding for the question
        embedding = get_embedding(user_message)

        # Query Pinecone
        results = query_pinecone(embedding)

        # Combine context from results
        context = " ".join([
            match.metadata.get('text', '')
            for match in results.matches
            if hasattr(match, 'metadata')
        ])

        # Generate response using OpenAI
        answer = generate_response(user_message, context, global_chat_history)

        # Update global chat history
        global_chat_history.append((user_message, answer))

        return jsonify({
            'status': 'success',
            'answer': answer
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


if __name__ == '__main__':
    app.run(debug=True)
