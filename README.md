# 🧠 RAG-based QA System for Medical Documents

This project is a **Retrieval-Augmented Generation (RAG)** pipeline designed to answer medical questions based on the contents of a given PDF document. It uses:

- **PyMuPDF** to extract text from the PDF
- **SentenceTransformers** to embed text chunks
- **Qdrant** (vector database) to store and search embeddings
- **Flan-T5** (local) to generate natural language answers


---

## 📂 Project Structure

- `extract_text_from_pdf(path)` – Extracts raw text from the PDF
- `chunk_text_by_sentences(text)` – Chunks the text for better context coherence
- `store_in_qdrant(chunks, model)` – Embeds and stores chunks in a Qdrant collection
- `search_query(query, model, client)` – Retrieves top-k most relevant chunks
- `generate_answer_t5(context, question)` – Uses a local Flan-T5 model to answer questions based on the context

---

## 🔧 Dependencies

Install required libraries:

```bash
pip install sentence-transformers qdrant-client pymupdf transformers
