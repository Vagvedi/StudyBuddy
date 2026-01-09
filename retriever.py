"""
Retrieval Module

This module implements semantic retrieval using embeddings and vector search.
Retrieves top-k most relevant chunks for a given query.
"""

from typing import List, Tuple
from embeddings import EmbeddingGenerator
from vector_store import VectorStore


class Retriever:
    """
    Handles semantic retrieval of relevant text chunks.
    
    Attributes:
        embedding_generator: EmbeddingGenerator instance
        vector_store: VectorStore instance
    """
    
    def __init__(self, embedding_generator: EmbeddingGenerator, vector_store: VectorStore):
        """
        Initialize the retriever.
        
        Args:
            embedding_generator: EmbeddingGenerator for query embeddings
            vector_store: VectorStore containing document chunks
        """
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
    
    def retrieve(self, query: str, k: int = 5, min_similarity: float = 0.0) -> List[Tuple[str, float]]:
        """
        Retrieve top-k most relevant chunks for a query.
        
        Args:
            query: User query string
            k: Number of chunks to retrieve
            min_similarity: Minimum similarity threshold (0.0 to 1.0)
            
        Returns:
            List of tuples (chunk_text, similarity_score) sorted by relevance
        """
        if not query or not query.strip():
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embeddings(query)
        
        # Search in vector store
        results = self.vector_store.search(query_embedding, k=k)
        
        # Filter by minimum similarity
        filtered_results = [
            (text, score) for text, score in results
            if score >= min_similarity
        ]
        
        return filtered_results
