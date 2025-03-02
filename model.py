import asyncio  
import os
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

# 📌 Définition des chemins
DB_FAISS_PATH = 'vectorestores/db_faiss'
DATA_FOLDER = "data"  
os.makedirs(DATA_FOLDER, exist_ok=True)  # ✅ Création du dossier s'il n'existe pas

# 📌 Modèle de prompt personnalisé
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """Créer un modèle de prompt personnalisé."""
    return PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])

def retrieval_qa_chain(llm, prompt, db):
    """Créer une chaîne de QA avec récupération des données."""
    return RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )

def load_llm():
    """Charger le modèle de langage."""
    return CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )

async def qa_bot():
    """Initialiser la chaîne de QA avec FAISS et le modèle LLM."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    return retrieval_qa_chain(llm, qa_prompt, db)

async def final_result(query):
    """Obtenir une réponse basée sur la requête utilisateur."""
    qa_result = await qa_bot()
    response = await qa_result({'query': query})
    return response

def load_document(file_path):
    """Charger un fichier et extraire son contenu."""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
    return loader.load()

def store_in_faiss(documents):
    """Stocker les embeddings des documents dans FAISS."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(documents, embeddings)
    db.save_local(DB_FAISS_PATH)  # ✅ Sauvegarde FAISS

@cl.on_chat_start
async def start():
    """Démarrer la session Chainlit et proposer un bouton d'upload."""
    chain = await qa_bot()
    cl.user_session.set("chain", chain)

    msg = cl.Message(content="Hi, Welcome to the chatbot. Upload a file to get started!")
    await msg.send()

    uploaded_files = await cl.AskFileMessage(content="Upload a PDF or text file.", accept=["application/pdf", "text/plain"]).send()
    
    if uploaded_files:
        uploaded_file = uploaded_files[0]  # ✅ Prendre le premier fichier de la liste
        
        if hasattr(uploaded_file, "content"):  # ✅ Vérifie que l'objet a bien un contenu
            file_path = os.path.join(DATA_FOLDER, uploaded_file.name)  # ✅ Stocke le fichier dans `data/`
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.content)  # ✅ Écrire le fichier correctement

            # Charger et indexer le document
            documents = load_document(file_path)
            store_in_faiss(documents)

            await cl.Message(content=f"📄 File '{uploaded_file.name}' has been indexed! You can now ask questions.").send()
        else:
            await cl.Message(content="🚨 Error: Uploaded file format is incorrect. Please try again.").send()

@cl.on_message
async def main(message): 
    """Gérer les messages utilisateur et répondre avec le chatbot."""
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"])
    cb.answer_reached = True
    res = await chain.acall(message, callbacks=[cb])
    answer = res["result"]
    await cl.Message(content=answer).send()

if __name__ == "__main__":
    asyncio.run(cl.main())
