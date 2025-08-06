# Document Q&A Application

A Streamlit-based application that allows users to upload PDF documents and ask questions about their content using Groq's LLM API and efficient document retrieval.

## Features

- PDF document upload and processing
- Interactive Q&A interface
- Multiple LLM model options (Llama 3.1, Llama 3.3, Gemma 2)
- Efficient document chunking and retrieval
- Vector-based semantic search
- Real-time answers with source context

## Technology Stack

### Core Components

1. **Frontend (Streamlit)**
   - Handles all UI components
   - Manages state and user interactions
   - Provides real-time feedback and loading states

2. **Document Processing**
   - Uses `PyPDFLoader` for PDF parsing
   - Implements `RecursiveCharacterTextSplitter` for smart text chunking
   - Handles temporary file management

3. **Vector Store System**
   - Uses `sentence-transformers` for document embedding
   - Implements FAISS for efficient similarity search
   - Maintains document chunks and embeddings in memory

4. **LLM Integration**
   - Connects to Groq API for model inference
   - Supports multiple model options
   - Handles context-aware question answering

## Dependencies

- `streamlit`: Web application framework
- `groq`: Groq API client for LLM access
- `sentence-transformers`: For text embeddings
- `langchain` & `langchain-community`: Document loading and processing
- `faiss-cpu`: Vector similarity search
- `numpy`: Numerical operations
- `PyPDF2`: PDF processing

## Setup and Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Get a Groq API key from [groq.com](https://groq.com)

## Running the Application

1. Activate your virtual environment
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Open your browser at `http://localhost:8501`

## Usage

1. Enter your Groq API key in the sidebar
2. Select your preferred model
3. Upload a PDF document
4. Ask questions using:
   - Predefined question buttons
   - Custom questions in the text input
5. View answers and source context in the expandable sections

## How It Works

1. **Document Processing**
   - PDF is loaded and split into manageable chunks
   - Each chunk is processed and embedded using SentenceTransformer

2. **Vector Store**
   - Creates and maintains a FAISS index for similarity search
   - Stores document chunks and their embeddings
   - Enables efficient retrieval of relevant content

3. **Question Answering**
   - User question is processed and relevant chunks are retrieved
   - Context is sent to Groq API along with the question
   - Response is formatted and displayed with source context

## Error Handling

- Graceful handling of API rate limits
- PDF processing error management
- Invalid input detection
- Clear error messages and user guidance
