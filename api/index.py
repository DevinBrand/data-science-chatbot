from flask import Flask, render_template, request, jsonify
import os
import warnings
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain_pinecone import PineconeVectorStore

warnings.filterwarnings("ignore")
load_dotenv()

app = Flask(__name__)


# Initialize the chat components
def init_chat():
    embeddings = OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"))
    vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"],
        embedding=embeddings
    )
    chat = ChatOpenAI(verbose=True, temperature=0, model_name="gpt-3.5-turbo")
    return ConversationalRetrievalChain.from_llm(
        llm=chat,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )


# Initialize chat outside of routes
qa = init_chat()
chat_history = []


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/ask', methods=['POST'])
def ask():
    try:
        user_message = request.json['message']

        input_dict = {
            "question": user_message,
            "chat_history": chat_history
        }

        res = qa.invoke(input_dict)

        chat_history.append((user_message, res["answer"]))

        return jsonify({
            'status': 'success',
            'answer': res["answer"]
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
