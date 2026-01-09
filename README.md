# AI Study Partner – Context-Aware Question Answering from PDFs

A production-ready NLP system that answers questions from uploaded PDF documents using Retrieval-Augmented Generation (RAG). The system **only** answers from uploaded content and explicitly says "I don't know" when information is not available.

## 🎯 Features

- **PDF Processing**: Extract and process text from multi-page PDFs
- **Semantic Search**: Find relevant content using embeddings (not just keywords)
- **Context-Aware Answers**: Generate answers grounded in retrieved documents
- **Anti-Hallucination**: Strict prompting prevents made-up answers
- **Document Summarization**: Generate summaries of uploaded notes
- **Web Interface**: User-friendly Gradio UI

## 🏗️ Architecture

This system implements a **Retrieval-Augmented Generation (RAG)** pipeline:

```
PDF → Text Extraction → Chunking → Embeddings → Vector Store → Retrieval → Answer Generation
```

### Components

1. **PDF Processor** (`pdf_processor.py`): Extracts text from PDFs using PyPDF2
2. **Text Chunker** (`text_chunker.py`): Splits text into sentence-aware chunks
3. **Embeddings** (`embeddings.py`): Converts text to semantic vectors
4. **Vector Store** (`vector_store.py`): FAISS-based similarity search
5. **Retriever** (`retriever.py`): Finds relevant chunks for queries
6. **Answer Generator** (`answer_generator.py`): Generates answers from context
7. **Summarizer** (`summarizer.py`): Summarizes documents
8. **RAG Pipeline** (`rag_pipeline.py`): Orchestrates all components
9. **UI** (`app.py`): Gradio web interface

## 📦 Installation & Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone or download this repository**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

**Note**: First-time installation will download pre-trained models (~2-3 GB total):
- `all-MiniLM-L6-v2` (embedding model, ~90 MB)
- `google/flan-t5-base` (answer model, ~990 MB)
- `facebook/bart-large-cnn` (summarization model, ~1.6 GB)

3. **Run the application**:
```bash
python app.py
```

4. **Open your browser** to `http://localhost:7860`

### Quick Test

1. Upload a PDF file (e.g., lecture notes, textbook chapter)
2. Click "Process PDF" and wait for indexing
3. Go to "Ask Questions" tab
4. Ask: "What is this document about?"
5. Get an answer grounded in your PDF!

### Programmatic Usage

See `example_usage.py` for how to use the pipeline without the UI:
```python
from rag_pipeline import RAGPipeline

pipeline = RAGPipeline()
chunks, text = pipeline.process_pdf("notes.pdf")
pipeline.index_documents(chunks)
answer, chunks = pipeline.answer_question("Your question here")
```

## 📚 Key Concepts Explained

### 1. Text Chunking

**Why chunk?**
- Language models have token limits (e.g., 512-1024 tokens)
- Large documents can't fit in a single prompt
- Chunking allows processing long documents

**Our approach:**
- **Chunk size**: ~500 tokens (balance between context and model limits)
- **Overlap**: ~100 tokens (preserves context at chunk boundaries)
- **Sentence-aware**: Never cuts sentences in half (better semantic coherence)

**Example:**
```
Original text: "Machine learning is... [500 words] ...deep learning."

Chunk 1: "Machine learning is... [500 tokens]"
Chunk 2: "[100 tokens overlap] ...deep learning." [400 new tokens]
```

### 2. Embeddings

**What are embeddings?**
- Numerical representations of text that capture semantic meaning
- Similar texts have similar embeddings (close in vector space)
- Enables semantic search beyond keyword matching

**Why embeddings instead of keyword search?**

| Keyword Search | Embedding-Based Search |
|---------------|----------------------|
| "car" ≠ "automobile" | "car" ≈ "automobile" (similar vectors) |
| No context understanding | Understands context ("bank" = financial vs river) |
| Misses synonyms | Handles synonyms and paraphrasing |
| Exact match required | Semantic similarity |

**Our model: `all-MiniLM-L6-v2`**
- 384-dimensional embeddings
- Fast and efficient
- Good quality for semantic search
- Normalized for cosine similarity

**Example:**
```
Query: "What is neural network?"
Document: "A neural network is a computational model..."
→ High similarity (semantically related)

Query: "What is neural network?"
Document: "The weather today is sunny..."
→ Low similarity (unrelated)
```

### 3. Semantic Search

**How it works:**
1. Convert query to embedding vector
2. Compare with all document chunk embeddings
3. Return top-k most similar chunks (by distance/similarity)

**FAISS (Facebook AI Similarity Search):**
- Fast library for similarity search
- IndexFlatL2: Exact search using L2 (Euclidean) distance
- For normalized embeddings, L2 distance ≈ cosine distance
- Supports millions of vectors efficiently

**Retrieval process:**
```
User Question → Embedding → Search in FAISS → Top-k Chunks
```

### 4. Retrieval-Augmented Generation (RAG)

**What is RAG?**
- Combines retrieval (finding relevant info) with generation (creating answers)
- Prevents hallucination by grounding answers in retrieved context
- Better than pure generation for factual questions

**RAG Pipeline:**
```
1. User asks: "What is machine learning?"
2. System retrieves relevant chunks from PDF
3. System generates answer using ONLY retrieved chunks
4. If chunks don't contain answer → "I don't know"
```

**Why RAG?**
- **Without RAG**: Model might make up answers from training data
- **With RAG**: Model only uses provided context (your PDFs)

**Our implementation:**
- Uses `google/flan-t5-base` for answer generation
- Strict prompting: "Answer using ONLY the context below"
- Validation to catch hallucinations

### 5. Answer Generation

**Model: `google/flan-t5-base`**
- T5 (Text-to-Text Transfer Transformer) architecture
- Trained for various NLP tasks
- Good for question answering
- Relatively small and fast

**Prompt engineering:**
```
"Answer the question using ONLY the information provided in the context below. 
If the context does not contain enough information to answer the question, 
respond with 'I don't know.'

Context: [retrieved chunks]

Question: [user question]

Answer (using only the context above):"
```

**Anti-hallucination measures:**
1. Explicit instruction to use only context
2. Post-processing validation
3. Returns "I don't know" if context is insufficient

### 6. Summarization

**Model: `facebook/bart-large-cnn`**
- BART (Bidirectional and Auto-Regressive Transformer)
- Trained specifically for summarization
- CNN/DailyMail dataset (news articles)
- Good for extractive-abstractive summaries

**Use case:**
- Quick overview of uploaded lecture notes
- Helps understand document structure
- Useful before asking specific questions

## 🔧 Configuration

You can customize the pipeline in `app.py`:

```python
pipeline = RAGPipeline(
    embedding_model="all-MiniLM-L6-v2",      # Embedding model
    answer_model="google/flan-t5-base",       # Answer generation model
    summarizer_model="facebook/bart-large-cnn", # Summarization model
    chunk_size=500,                           # Chunk size in tokens
    chunk_overlap=100                         # Overlap in tokens
)
```

## 📊 Limitations

1. **Model Limitations**:
   - `flan-t5-base` is relatively small (250M parameters)
   - May struggle with very complex reasoning
   - Answers are concise (not long-form)

2. **PDF Quality**:
   - Works best with text-based PDFs
   - Scanned PDFs (images) require OCR (not included)
   - Complex layouts may extract poorly

3. **Context Window**:
   - Limited by model's max input length (512 tokens for flan-t5-base)
   - Very long chunks may be truncated

4. **Retrieval Quality**:
   - Depends on embedding model quality
   - May miss relevant chunks if query phrasing differs significantly
   - Top-k retrieval may not always get the best chunks

5. **No Multi-Document Support**:
   - Currently processes one PDF at a time
   - Can be extended to support multiple PDFs

6. **No Conversation Memory**:
   - Each question is independent
   - No follow-up question handling

## 🚀 Usage Examples

### Example 1: Upload and Ask

1. Upload `lecture_notes.pdf`
2. Click "Process PDF"
3. Ask: "What are the main topics covered?"
4. Get answer grounded in your PDF

### Example 2: Summarize

1. Upload `lecture_notes.pdf`
2. Go to "Summarize" tab
3. Click "Generate Summary"
4. Get a concise overview

## 🔍 Technical Details

### Token Counting
- Uses T5 tokenizer for chunking (consistent with answer model)
- Approximate: 1 token ≈ 0.75 words (English)

### Similarity Threshold
- Default: 0.3 (30% similarity minimum)
- Lower = more results (may include irrelevant)
- Higher = fewer results (may miss relevant)

### Retrieval Strategy
- Top-k = 5 chunks (default)
- Can be adjusted based on document length
- More chunks = more context but slower generation

## 📝 Code Structure

```
StudyBuddy/
├── app.py                 # Gradio UI
├── rag_pipeline.py       # Main orchestrator
├── pdf_processor.py      # PDF extraction
├── text_chunker.py       # Text chunking
├── embeddings.py         # Embedding generation
├── vector_store.py       # FAISS vector DB
├── retriever.py          # Semantic retrieval
├── answer_generator.py   # Answer generation
├── summarizer.py         # Document summarization
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## 🛠️ Extending the System

### Add Multi-PDF Support
- Modify `rag_pipeline.py` to handle multiple PDFs
- Combine chunks from all PDFs in vector store

### Improve Answer Quality
- Use larger models (e.g., `flan-t5-large`)
- Implement re-ranking of retrieved chunks
- Add answer confidence scores

### Add Conversation Memory
- Store conversation history
- Use previous context in retrieval
- Implement follow-up question handling

### Support Other File Types
- Add DOCX support (python-docx)
- Add TXT support
- Add Markdown support

## 📄 License

This project is provided as-is for educational and research purposes.

## 🙏 Acknowledgments

- HuggingFace for transformer models
- Facebook AI Research for FAISS
- Gradio team for the UI framework

---

**Built with ❤️ using PyTorch, HuggingFace Transformers, and FAISS**
