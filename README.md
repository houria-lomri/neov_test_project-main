Medical Bot - AI-Powered Document Q&A


📌 Project Overview
This project aims to develop an intelligent chatbot capable of answering users' questions using indexed documents (PDF or text). The goal is to enable efficient and accurate information retrieval from a document corpus.

🚀 Features
✅ Upload PDF or text files for analysis.
✅ Retrieve answers based on document content.
✅ Uses FAISS for efficient document indexing.
✅ Runs locally with CTransformers for LLM inference.
✅ Supports multi-turn conversations with Chainlit.

🛠 Technology Stack
Python (Backend)
FAISS (Vector Storage)
HuggingFace Embeddings (Text Processing)
CTransformers & Llama 2 (Language Model)
Chainlit (Interactive Chat Interface)
📂 Project Structure

📁 neov_test_project
 ┣ 📁 data/                   # Folder for uploaded documents  
 ┣ 📁 vectorestores/          # FAISS vector storage  
 ┣ 📁 images/                 # Screenshots & UI images  
 ┣ 📄 model.py                # Main script for the chatbot  
 ┣ 📄 README.md               # Project documentation  
 ┗ 📄 requirements.txt        # Dependencies  
🔧 Installation & Setup
1️⃣ Clone the repository

bash
Copier le code
git clone https://github.com/your-repo/medical-bot.git
cd medical-bot
2️⃣ Create a virtual environment

python -m venv llama_env
source llama_env/bin/activate  # (Linux/macOS)
llama_env\Scripts\activate     # (Windows)

3️⃣ Install dependencies
pip install -r req.txt


4️⃣ Run the chatbot
chainlit run model.py --port 8501


5️⃣ Upload a document and start asking questions!

![Chatbot Interface](images/interface1.png)

![Chatbot Interface](images/interface.png)
