import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import tempfile
import os

def initialize_groq(api_key):
    return Groq(api_key=api_key)


def get_groq_response(client, context, question, model_name="llama-3.1-8b-instant"):
    prompt = f"""
Based on the following context, please answer the question in a concise manner:
Context: {context}
Question: {question}
Answer: Provide a clear, accurate answer based only on the context provided.
"""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error getting response: {str(e)}"


def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


class LocalVectorStore:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.chunks = []
        self.embeddings = None
        self.index = None
        
    def add_documents(self, documents):
        self.chunks = [doc.page_content for doc in documents]
        embeddings = self.embedding_model.encode(self.chunks)
        self.embeddings = np.array(embeddings, dtype=np.float32)

        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)
    
    def similarity_search(self, query, k=4):
        if self.index is None:
            return []
        
        query_embedding = self.embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype(np.float32)

        distances, indices = self.index.search(query_embedding, k)
        results = []

        for i in indices[0]:
            if i < len(self.chunks):
                results.append(self.chunks[i])
        
        return results


def load_and_split_pdf(uploaded_file):
    try:
        # First, check if we have a valid file
        if uploaded_file is None:
            st.error("No file was uploaded")
            return None

        # Create temp file with proper error handling
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name

        try:
            # Try to load the PDF
            loader = PyPDFLoader(temp_file_path)
            documents = loader.load()

            if not documents:
                st.error("The PDF appears to be empty")
                return None

            # Split the documents
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            split_docs = text_splitter.split_documents(documents)

            if not split_docs:
                st.error("Could not extract any text from the PDF")
                return None

            return split_docs

        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return None
        
        finally:
            # Clean up temp file in finally block to ensure it always happens
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    except Exception as e:
        st.error(f"Error handling file upload: {str(e)}")
        return None


def process_document(uploaded_file, groq_client, embedding_model):
    st.write("Processing document...")

    with st.spinner("Reading PDF ..."):
        chunks = load_and_split_pdf(uploaded_file)
    
    if not chunks:
        st.error("Failed to read or split the PDF document.")
        return 
    
    st.success(f"Loaded {len(chunks)} chunks from the document.")

    with st.spinner("Creating vector store ..."):
        vector_store = LocalVectorStore(embedding_model)
        vector_store.add_documents(chunks)
    
    st.success("Vector store created successfully.")

    st.session_state.vector_store = vector_store
    st.session_state.groq_client = groq_client
    st.session_state.ready = True

    if st.session_state.get("ready", False):
        st.header("Ask your Questions")

        st.write("**Try asking questions about the document you just uploaded.**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📝 What is this document about?"):
                st.session_state.question = "What is this document about?"
            if st.button("✍️ Who are the main authors or people mentioned?"):
                st.session_state.question = "Who are the main authors or people mentioned?"
        with col2:
            if st.button("🔍 What are the key findings or conclusions?"):
                st.session_state.question = "What are the key findings or conclusions?"
            if st.button("📊 Can you summarize the main points?"):
                st.session_state.question = "Can you summarize the main points?"
        
        question = st.text_input("Ask a question about the document:", value=st.session_state.get("question", ""))

        if question:
            try:
                with st.spinner("Searching for relevant information..."):
                    relevant_chunks = st.session_state.vector_store.similarity_search(question, k=4)
                    if not relevant_chunks:
                        st.warning("No relevant information found in the document.")
                        return 
                    
                    context = "\n\n".join(relevant_chunks)

                    answer = get_groq_response(
                        st.session_state.groq_client, context, question, st.session_state.selected_model
                    )

                st.write("**Answer:**")
                st.write(answer)

                st.success("Answer generated successfully.")
                with st.expander("View source chunks"):
                    for i, chunk in enumerate(relevant_chunks):
                        st.write(f"**Chunk {i+1}:**")
                        display_chunk = chunk[:400] + "..." if len(chunk) > 400 else chunk
                        st.write(display_chunk)
                        st.write("---")

            except Exception as e:
                #Handle different types of errors gracefully
                if "rate limit" in str(e).lower():
                    st.error("🕛 Rate limit exceeded. Please try again later.")
                    st.info(" 💡Free tier limits are generous but not unlimited")
                
                else:
                    st.error(f"🛑 Error: {str(e)}")
                    st.info(" 💡 Try simplifying your question or check your API key.")


def main():
    st.set_page_config(page_title="Document Q&A App (AISOC Raw & Stupid)",
    page_icon=":📚:",
    layout="wide")

    st.title("Document Q&A App (AISOC Raw & Stupid)")
    st.write("Upload a PDF and ask questions about it!")

    st.sidebar.header("Settings")
    st.sidebar.write("Configure your API settings and model preferences. You can get a free api at groq.com")

    model_options = {
        "llama-3.1-8b-instant": "Llama 3.1 8B (Fast & Smart)",
        "llama-3.3-70b-versatile": "Llama 3.3 70B (Most Accurate)",
        "gemma2-9b-it": "Gemma 2 9B (Balanced)"
    }

    selected_model = st.sidebar.selectbox(
        "Choose a model",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x],
        index=0
    )

    groq_api_key = st.sidebar.text_input("Groq API Key", type="password", help="Get your free API key at [groq.com](https://groq.com/).")

    if not groq_api_key:
        st.warning("Please enter your Groq API key to proceed.")
        return
    
    groq_client = initialize_groq(groq_api_key)
    embedding_model = load_embedding_model()

    st.session_state.selected_model = selected_model

    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"], help="Upload a PDF file to ask questions about it.")

    if uploaded_file is not None:
        process_document(uploaded_file, groq_client, embedding_model)

if __name__ == "__main__":
    main()