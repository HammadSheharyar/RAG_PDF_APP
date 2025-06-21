import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer
import requests
import numpy as np

# === CONFIG ===
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# === FUNCTION 1: Load PDF ===
def load_pdf(file) -> str:
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# === FUNCTION 2: Chunk Text ===
def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokens = tokenizer.tokenize(text)
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokens[i:i + chunk_size]
        chunk_text = tokenizer.convert_tokens_to_string(chunk)
        chunks.append(chunk_text)
    return chunks

# === FUNCTION 3: Embed Chunks ===
def embed_chunks(chunks):
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = model.encode(chunks)
    return embeddings, model

# === FUNCTION 4: Build FAISS Index ===
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index

# === FUNCTION 5: Query & Retrieve ===
def query_index(query_text, chunks, index, embed_model):
    query_vec = embed_model.encode([query_text])
    D, I = index.search(np.array(query_vec), k=3)
    results = [chunks[i] for i in I[0]]
    return "\n\n".join(results)

# === FUNCTION 6: Generate with Groq + LLaMA3 ===
def generate_with_groq(context, query):
    API_KEY = "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"  # 🔐 Hardcoded Groq Key
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
        ]
    }

    response = requests.post(url, headers=headers, json=payload)
    return response.json()['choices'][0]['message']['content']

# === STREAMLIT UI ===
def main():
    st.title("🧠 RAG App with LLaMA3 (Groq)")

    uploaded_file = st.file_uploader("📄 Upload your PDF", type=["pdf"])

    if uploaded_file:
        with st.spinner("📚 Reading and processing PDF..."):
            text = load_pdf(uploaded_file)
            chunks = chunk_text(text)
            embeddings, embed_model = embed_chunks(chunks)
            index = build_faiss_index(embeddings)

        st.success("✅ PDF processed and indexed!")

        user_query = st.text_input("🔍 Ask a question about the document:")
        if user_query:
            with st.spinner("💬 Thinking..."):
                context = query_index(user_query, chunks, index, embed_model)
                response = generate_with_groq(context, user_query)
            st.markdown("### 💡 Answer")
            st.write(response)

if __name__ == "__main__":
    main()
