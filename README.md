# 🧠 PDF Q&A App (RAG with Groq + LLaMA3)

This app lets you upload a PDF, ask questions, and get answers via Groq-hosted LLaMA3.

---

## 🚀 Quick Start (Local or Colab)

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/rag-rag-app.git
cd rag-rag-app
### 2. Install Dependencies

pip install -r requirements.txt


### 3. Set Up Groq API Key
Create a file named .env based on .env.example

Obtain your API key from Groq and set it:

dotenv

GROQ_API_KEY=sk-your_actual_key_here
### 4. Run the App
streamlit run rag_app.py

Uploaded PDFs will be tokenized, embedded, and indexed via FAISS. Ask questions in the UI — answers will be generated using LLaMA3 through the Groq API.
