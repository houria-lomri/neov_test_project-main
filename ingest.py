import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Définition des chemins vers les données et vers la base de données FAISS
DATA_PATH="data/"
DB_FAISS_PATH="vectorstores/db_faiss"

def create_vector_db():
    # Chargement des documents depuis le répertoire spécifié
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    documents =loader.load()
    # Division du texte en petits morceaux (smaller chunks ) pour le traitement 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Initialisation du modèle d'embeddings HuggingFace
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs = {'device': 'cpu'})

    # Création FAISS vector store à partir des textes traités
    db = FAISS.from_documents(texts, embeddings)
    # Save the FAISS vector store locally
    db.save_local(DB_FAISS_PATH)

if __name__ == "__main__":
    # Execute the function to create the vector database
    create_vector_db()
    
    