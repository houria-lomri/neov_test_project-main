# ChatBot - AI-Powered Document Q&A

## 📌 Description du projet  
Ce projet vise à développer un chatbot intelligent capable de répondre aux questions des utilisateurs en utilisant des documents indexés (PDF ou texte). L'objectif est d'assurer une récupération rapide et précise des informations.

## ✅ Fonctionnalités  
- 📂 **Upload** de fichiers PDF ou texte pour l'analyse  
- 🔍 **Récupération** des réponses en fonction du contenu des documents  
- ⚡ **Indexation efficace** des documents avec FAISS  
- 🖥️ **Exécution locale** grâce à `CTransformers` pour l'inférence LLM  
- 💬 **Conversations multi-tours** supportées avec `Chainlit`  

## 🛠️ Technologies utilisées  
- **Python** (Backend)  
- **FAISS** (Stockage des vecteurs)  
- **HuggingFace Embeddings** (Traitement du texte)  
- **CTransformers & Llama 2** (Modèle de langage)  
- **Chainlit** (Interface interactive du chatbot)  

## 📁 Structure du projet  


📁 neov_test_project
 ┣ 📁 data/                   # Folder for uploaded documents  
 ┣ 📁 vectorestores/          # FAISS vector storage  
 ┣ 📁 images/                 # Screenshots & UI images  
 ┣ 📄 model.py                # Main script for the chatbot  
 ┣ 📄 README.md               # Project documentation  
 ┗ 📄 req.txt                 # Dependencies  



##  Installation & Setup

2️⃣ Create a virtual environment

python -m venv llama_env
source llama_env/bin/activate  # (Linux/macOS)
llama_env\Scripts\activate     # (Windows)

3️⃣ Install dependencies
pip install -r req.txt


4️⃣ Run the chatbot
chainlit run model.py --port 8501


5️⃣ Upload a document and start asking questions!

![Uploading interface1.PNG…]()



![Chatbot Interface](images/interface.png)
