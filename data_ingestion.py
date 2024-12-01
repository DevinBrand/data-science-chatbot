import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()

if __name__ == "__main__":
    print("ingesting data...")

    # Initialize Pinecone client
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    # Define the directory containing PDF files
    pdf_directory = "data/"

    # Get all PDF files in the directory
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

    all_texts = []

    # Process each PDF file
    for pdf_file in pdf_files:
        print(f"Processing {pdf_file}...")

        # Load PDF document
        loader = PyPDFLoader(os.path.join(pdf_directory, pdf_file))
        document = loader.load()
        print(f"Loaded {pdf_file}")

        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(document)
        all_texts.extend(texts)

        print(f"Created {len(texts)} chunks from {pdf_file}")

    print(f"Total chunks created: {len(all_texts)}")

    # Create vector embeddings and save them in Pinecone database
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    PineconeVectorStore.from_documents(all_texts, embeddings, index_name=os.getenv("INDEX_NAME"))
