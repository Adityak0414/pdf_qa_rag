
import fitz  # PyMuPDF
import qdrant_client
from qdrant_client.http.models import PointStruct, Distance, VectorParams
from sentence_transformers import SentenceTransformer
import uuid
import requests
import os
import time
import json

# === 1. Load and Chunk the PDF ===
def extract_text_from_pdf(path):
    """Extract text from PDF using PyMuPDF"""
    try:
        doc = fitz.open(path)
        text = "\n".join([page.get_text() for page in doc])
        doc.close()
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks based on word count"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk)
    return chunks

def chunk_text_by_sentences(text, max_tokens=400):
    """Alternative chunking strategy: split by sentences with token limit"""
    sentences = text.replace('\n', ' ').split('.')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Rough token estimation (1 token ≈ 4 characters)
        if len(current_chunk) + len(sentence) < max_tokens * 4:
            current_chunk += sentence + ". "
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

# === 2. Embed and Store in Qdrant ===
def store_in_qdrant(chunks, model, collection_name="medical_chunks"):
    """Store text chunks as embeddings in Qdrant vector database"""
    client = qdrant_client.QdrantClient(":memory:")  # Use in-memory Qdrant
    
    # Get embedding dimension
    embedding_dim = model.get_sentence_embedding_dimension()
    
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
    )
    
    points = []
    print(f"Storing {len(chunks)} chunks in vector database...")
    
    for i, chunk in enumerate(chunks):
        embedding = model.encode(chunk).tolist()
        points.append(PointStruct(
            id=i, 
            vector=embedding, 
            payload={"text": chunk, "chunk_id": i}
        ))
    
    client.upsert(collection_name=collection_name, points=points)
    print(f"Successfully stored {len(points)} chunks")
    return client

# === 3. Query Interface ===
def search_query(query, model, client, collection_name="medical_chunks", top_k=3):
    """Retrieve most relevant chunks for a given query"""
    query_vec = model.encode(query).tolist()
    hits = client.search(
        collection_name=collection_name, 
        query_vector=query_vec, 
        limit=top_k
    )
    
    results = []
    for hit in hits:
        results.append({
            "text": hit.payload["text"],
            "score": hit.score,
            "chunk_id": hit.payload["chunk_id"]
        })
    
    return results

# === 4. LLM Answer Generation ===
def generate_answer_t5(context, question):
    """Generate answer using local T5 model"""
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    
    print("Loading T5 model...")
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
    
    # Format input for T5 - T5 works better with direct task instruction
    input_text = f"Question: {question}\nContext: {context}\nAnswer:"
    
    # Tokenize input
    input_ids = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).input_ids
    
    # Generate answer
    outputs = model.generate(
        input_ids,
        max_length=150,
        num_beams=5,
        early_stopping=True,
        temperature=0.5,
        do_sample=True
    )
    
    # Decode and return answer
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.strip()

# === MAIN PIPELINE ===
def main():
    # Check if PDF exists
    pdf_path = "9241544228_eng.pdf"
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file '{pdf_path}' not found.")
        print("Please ensure the PDF file is in the same directory as this script.")
        return
    
    print("=== RAG QA System for Medical Documents ===")
    print("1. Loading and processing PDF...")
    
    # Load and process
    raw_text = extract_text_from_pdf(pdf_path)
    if not raw_text:
        print("Error: Could not extract text from PDF")
        return
    
    print(f"Extracted {len(raw_text)} characters from PDF")
    
    # Chunk the text (using sentence-based chunking for better coherence)
    chunks = chunk_text_by_sentences(raw_text, max_tokens=400)
    print(f"Created {len(chunks)} chunks")
    
    # Show sample chunk
    if chunks:
        print(f"Sample chunk: {chunks[0][:200]}...")
    
    print("\n2. Creating embeddings and storing in vector database...")
    
    # Embed and store
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    qdrant = store_in_qdrant(chunks, embed_model)
    
    print("\n3. Running test queries...")
    
    # Questions to test
    test_questions = [
        "Give me the correct coded classification for the following diagnosis: Recurrent depressive disorder, currently in remission",
        "What are the diagnostic criteria for Obsessive-Compulsive Disorder (OCD)?",
    ]
    
    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print('='*60)
        
        # Retrieve relevant chunks
        search_results = search_query(question, embed_model, qdrant, top_k=3)
        
        print(f"Found {len(search_results)} relevant chunks:")
        for i, result in enumerate(search_results):
            print(f"  Chunk {i+1} (score: {result['score']:.3f}): {result['text'][:100]}...")
        
        # Combine context
        combined_context = "\n---\n".join([r["text"] for r in search_results])
        
        # Generate answer
        print("\nGenerating answer with T5 model...")
        
        # Use local T5 model for answer generation
        answer = generate_answer_t5(combined_context, question)
        
        print(f"\nAnswer: {answer}")

if __name__ == "__main__":
    main()