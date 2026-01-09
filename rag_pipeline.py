"""
RAG Pipeline Module

This module orchestrates the complete RAG (Retrieval-Augmented Generation) pipeline:
1. PDF processing
2. Text chunking
3. Embedding generation
4. Vector store indexing
5. Retrieval
6. Answer generation

This is the main orchestrator that ties all components together.
"""

from typing import List, Tuple, Optional
from pathlib import Path
import numpy as np

from pdf_processor import PDFProcessor
from text_chunker import TextChunker
from embeddings import EmbeddingGenerator
from vector_store import VectorStore
from retriever import Retriever
from answer_generator import AnswerGenerator
from summarizer import Summarizer


class RAGPipeline:
    """
    Complete RAG pipeline for question answering from PDFs.
    
    Attributes:
        pdf_processor: PDFProcessor instance
        chunker: TextChunker instance
        embedding_generator: EmbeddingGenerator instance
        vector_store: VectorStore instance
        retriever: Retriever instance
        answer_generator: AnswerGenerator instance
        summarizer: Summarizer instance
        is_indexed: Whether documents have been indexed
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        answer_model: str = "google/flan-t5-base",
        summarizer_model: str = "facebook/bart-large-cnn",
        chunk_size: int = 500,
        chunk_overlap: int = 100
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            embedding_model: Model for embeddings (default: all-MiniLM-L6-v2)
            answer_model: Model for answer generation (default: google/flan-t5-base)
            summarizer_model: Model for summarization (default: facebook/bart-large-cnn)
            chunk_size: Chunk size in tokens (default: 500)
            chunk_overlap: Chunk overlap in tokens (default: 100)
        """
        print("Initializing RAG Pipeline...")
        
        # Initialize components
        self.pdf_processor = PDFProcessor()
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedding_generator = EmbeddingGenerator(model_name=embedding_model)
        self.answer_generator = AnswerGenerator(model_name=answer_model)
        self.summarizer = Summarizer(model_name=summarizer_model)
        
        # Initialize vector store with embedding dimension
        embedding_dim = self.embedding_generator.get_embedding_dimension()
        self.vector_store = VectorStore(dimension=embedding_dim)
        self.retriever = Retriever(self.embedding_generator, self.vector_store)
        
        self.is_indexed = False
        
        print("RAG Pipeline initialized successfully")
    
    def process_pdf(self, pdf_path: str) -> Tuple[List[str], str]:
        """
        Process a PDF file: extract text and chunk it.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Tuple of (chunks, full_text)
        """
        print(f"Processing PDF: {pdf_path}")
        
        # Extract text
        text = self.pdf_processor.extract_text(pdf_path)
        print(f"Extracted {len(text)} characters from PDF")
        
        # Chunk text
        chunks = self.chunker.chunk_text(text)
        print(f"Created {len(chunks)} chunks")
        
        return chunks, text
    
    def index_documents(self, chunks: List[str]) -> None:
        """
        Index document chunks in the vector store.
        
        Args:
            chunks: List of text chunks to index
        """
        if not chunks:
            raise ValueError("No chunks provided for indexing")
        
        print(f"Indexing {len(chunks)} chunks...")
        
        # Generate embeddings
        embeddings = self.embedding_generator.generate_embeddings(chunks)
        print(f"Generated embeddings of shape {embeddings.shape}")
        
        # Add to vector store
        self.vector_store.add_vectors(embeddings, chunks)
        self.is_indexed = True
        
        print(f"Indexed {self.vector_store.get_size()} chunks successfully")
    
    def answer_question(self, question: str, k: int = 5, min_similarity: float = 0.3) -> Tuple[str, List[Tuple[str, float]]]:
        """
        Answer a question using the RAG pipeline.
        
        Args:
            question: User question
            k: Number of chunks to retrieve
            min_similarity: Minimum similarity threshold
            
        Returns:
            Tuple of (answer, retrieved_chunks_with_scores)
        """
        if not self.is_indexed:
            return (
                "No documents have been indexed. Please upload and process a PDF first.",
                []
            )
        
        # Retrieve relevant chunks
        retrieved_chunks = self.retriever.retrieve(question, k=k, min_similarity=min_similarity)
        
        if not retrieved_chunks:
            return (
                "I don't know. The uploaded documents don't contain relevant information to answer this question.",
                []
            )
        
        # Generate answer
        answer = self.answer_generator.generate_answer(question, retrieved_chunks)
        
        return answer, retrieved_chunks
    
    def summarize_documents(self, text: str) -> str:
        """
        Summarize the uploaded documents.
        
        Args:
            text: Full text to summarize
            
        Returns:
            Summary string
        """
        return self.summarizer.summarize(text)
    
    def save_index(self, index_path: str, texts_path: str) -> None:
        """
        Save the vector index to disk.
        
        Args:
            index_path: Path to save FAISS index
            texts_path: Path to save texts
        """
        self.vector_store.save(index_path, texts_path)
        print(f"Index saved to {index_path}")
    
    def load_index(self, index_path: str, texts_path: str) -> None:
        """
        Load the vector index from disk.
        
        Args:
            index_path: Path to FAISS index file
            texts_path: Path to texts file
        """
        self.vector_store.load(index_path, texts_path)
        self.is_indexed = True
        print(f"Index loaded from {index_path}")
