import os
import ollama
from langchain_ollama import OllamaEmbeddings
from os.path import curdir
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from ollama import embeddings

# Get the current directory of the script
current_dir = os.path.dirname(os.path.realpath(__file__))
# Define the path to the text file
file_path = os.path.join(current_dir, 'data', 'odyssey.txt')
# Define the path to the persistent directory for Chroma
persistent_directory = os.path.join(current_dir, 'db', 'chroma')

# Check if the persistent directory exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Building Chroma db from scratch")

    # Check if the text file exists
    if not os.path.exists(file_path):
        print("Text file does not exist. Please download the text file and place it in the data directory")
        exit(1)

    # Load the text file
    loader = TextLoader(file_path)
    document = loader.load()
    print("Loaded document from file")

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    chunks = text_splitter.split_documents(document)
    print("Document split into chunks")

    # Create embeddings for the chunks
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    print("Created embeddings for the document chunks")

    # Create the vector store and persist it automatically
    print("\n--- Creating vector store ---")
    db = Chroma.from_documents(document, embeddings, persist_directory=persistent_directory)
    print("\n--- Finished creating vector store ---")

    # Uncomment the following line if you want to use chunks instead of the whole document
    # db = Chroma.from_documents(chunks, embeddings, persistent_directory=persistent_directory)
else:
    print("Vector store already exists. No need to initialize.")