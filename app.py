import streamlit as st
from pdf_reader import read_pdf
from text_chunker import chunk_text
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from answer_generator import get_answer

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# FAISS index
dimension = 384
faiss_index = faiss.IndexFlatL2(dimension)
chunk_list = []

st.title("Chandrika RAG Chatbot - Full Chatbot")

# PDF upload
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    st.write("Uploaded:", uploaded_file.name)
    
    # Save PDF
    temp_pdf_path = "temp_uploaded.pdf"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    # Read PDF
    pdf_text = read_pdf(temp_pdf_path)
    chunks = chunk_text(pdf_text, chunk_size=500, overlap=50)

    # Create embeddings
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
    faiss_index.add(embeddings)
    chunk_list = chunks

    # Question input
    query = st.text_input("Ask a question about the PDF:")

    if query:
        # Encode query
        query_vector = embedding_model.encode([query], convert_to_numpy=True)
        k = 3  # top chunks
        distances, indices = faiss_index.search(query_vector, k)

        # Get top chunks
        top_chunks = [chunk_list[idx] for idx in indices[0]]

        # Get answer
        answer = get_answer(query, top_chunks)

        # Display answer
        st.subheader("Answer:")
        st.write(answer)

        # Show top chunks
        st.subheader("Relevant Chunks:")
        for idx, distance in zip(indices[0], distances[0]):
            st.write(f"**Score:** {distance:.4f}")
            st.write(chunk_list[idx])
            st.write("---")

    # Clean up
    if os.path.exists(temp_pdf_path):
        os.remove(temp_pdf_path)
