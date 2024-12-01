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

chat_history = []


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


def generate_response(query, context):
    messages = [
        {"role": "system",
         "content": "You are a helpful assistant. Use the following context to answer the question: " + context},
        {"role": "user", "content": query}
    ]

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

        # Get embedding for the question
        embedding = get_embedding(user_message)

        # Query Pinecone
        results = query_pinecone(embedding)

        # Combine context from results
        context = " ".join([r.metadata.get('text', '') for r in results.matches])

        # Generate response using OpenAI
        answer = generate_response(user_message, context)

        # Update chat history
        chat_history.append((user_message, answer))

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
