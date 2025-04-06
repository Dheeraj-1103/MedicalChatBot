---

## 🧠 Medical RAG Chatbot using LangChain + Llama + PubMedBERT

This project is a **Retrieval-Augmented Generation (RAG)** pipeline that allows users to query medical PDFs and get **context-aware, accurate answers** using a local LLM. It uses LangChain, SentenceTransformers, Chroma vector store, and the `BioMistral-7B` model.

---

### 🚀 Features

- 📄 Upload and parse **PDF documents**
- 🔍 Perform **semantic search** over embedded medical content
- 🤖 Generate medically-aligned answers using **Llama-based LLM**
- 🧠 Uses **PubMedBERT** for biomedical embeddings

---

### 🛠️ Installation

Install required libraries (recommended to run in Google Colab):

```bash
pip install langchain sentence-transformers chromadb llama-cpp-python langchain_community pypdf
```

---

### 📁 File Structure

- `/content/drive/MyDrive/Colab Notebooks/` → Your PDF files
- `/content/drive/MyDrive/BioMistral-7B.Q2_K.gguf` → Your LLM file

---

### ⚙️ Setup Steps

1. **Mount Google Drive**:
   ```python
   from google.colab import drive
   drive.mount("/content/drive")
   ```

2. **Load PDFs and Split Text**:
   ```python
   from langchain_community.document_loaders import PyPDFDirectoryLoader
   ...
   docs = loader.load()
   ```

3. **Generate Embeddings** using PubMedBERT:
   ```python
   from langchain_community.embeddings import SentenceTransformerEmbeddings
   embedding = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
   ```

4. **Build Vector Store with Chroma**:
   ```python
   vectorstore = Chroma.from_documents(chunks, embedding)
   ```

5. **Load the Language Model**:
   ```python
   from langchain_community.llms import LlamaCpp
   llm = LlamaCpp(model_path=".../BioMistral-7B.Q2_K.gguf", ...)
   ```

6. **Start Chat Interface**:
   ```python
   while True:
       user_input = input("Input query: ")
       ...
   ```

---

### 🧬 Example Query

```text
Input query: Who is at risk of heart disease?
Answer: [contextual answer based on the uploaded PDFs]
```

---

### 📦 LLM Model Download

You can download the **BioMistral 7B GGUF** model (compatible with `llama-cpp-python`) here:

- 🔗 [BioMistral-7B.Q2_K.gguf (via Hugging Face)](https://huggingface.co/TheBloke/BioMistral-7B-GGUF)

> **Recommended model file**: `BioMistral-7B.Q2_K.gguf`

Place the model in your Google Drive:  
`/content/drive/MyDrive/BioMistral-7B.Q2_K.gguf`

---

### 🔐 Notes

- Don't forget to **add your Hugging Face token** if using private models or APIs:
  ```python
  os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_token_here"
  ```
- Never upload private tokens to public repositories.

---

### 💡 Inspiration

This project combines:
- Biomedical language understanding (via **PubMedBERT**)
- Efficient document-based QA (via **LangChain RAG pipeline**)
- Local inference with LLMs (**Llama.cpp**)

---
