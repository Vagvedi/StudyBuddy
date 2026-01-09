"""
Vector Store Module

This module implements a FAISS-based vector database for semantic search.
Uses IndexFlatL2 for exact similarity search (L2 distance = Euclidean distance).
For normalized embeddings, L2 distance is equivalent to cosine distance.

Why FAISS?
- Fast similarity search on large vector collections
- Efficient indexing and retrieval
- Supports saving/loading indices
- Industry-standard for production RAG systems
"""

import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Optional


class VectorStore:
    """
    FAISS-based vector store for semantic search.
    
    Attributes:
        index: FAISS index instance
        texts: List of text chunks corresponding to vectors
        dimension: Dimension of stored vectors
    """
    
    def __init__(self, dimension: int):
        """
        Initialize the vector store.
        
        Args:
            dimension: Dimension of vectors to store (e.g., 384 for all-MiniLM-L6-v2)
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance for normalized embeddings
        self.texts: List[str] = []
    
    def add_vectors(self, vectors: np.ndarray, texts: List[str]) -> None:
        """
        Add vectors and corresponding texts to the store.
        
        Args:
            vectors: Numpy array of shape (n_vectors, dimension)
            texts: List of text strings corresponding to vectors
            
        Raises:
            ValueError: If dimensions don't match or lengths don't match
        """
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension {vectors.shape[1]} doesn't match store dimension {self.dimension}"
            )
        
        if len(vectors) != len(texts):
            raise ValueError(
                f"Number of vectors {len(vectors)} doesn't match number of texts {len(texts)}"
            )
        
        # Ensure vectors are float32 (FAISS requirement)
        vectors = vectors.astype('float32')
        
        # Add to FAISS index
        self.index.add(vectors)
        
        # Store corresponding texts
        self.texts.extend(texts)
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for most similar vectors.
        
        Args:
            query_vector: Query vector of shape (dimension,) or (1, dimension)
            k: Number of results to return
            
        Returns:
            List of tuples (text, distance) sorted by similarity (lower distance = more similar)
            
        Raises:
            ValueError: If query dimension doesn't match store dimension
        """
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        if query_vector.shape[1] != self.dimension:
            raise ValueError(
                f"Query dimension {query_vector.shape[1]} doesn't match store dimension {self.dimension}"
            )
        
        # Ensure float32
        query_vector = query_vector.astype('float32')
        
        # Search in FAISS
        k = min(k, self.index.ntotal)  # Don't request more than available
        distances, indices = self.index.search(query_vector, k)
        
        # Convert to list of (text, distance) tuples
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.texts):
                # Convert L2 distance to similarity score (lower distance = higher similarity)
                # For normalized vectors, distance ranges from 0 to 2
                similarity = 1.0 / (1.0 + dist)  # Convert distance to similarity
                results.append((self.texts[idx], float(similarity)))
        
        return results
    
    def save(self, index_path: str, texts_path: str) -> None:
        """
        Save the FAISS index and texts to disk.
        
        Args:
            index_path: Path to save FAISS index
            texts_path: Path to save texts (as pickle)
        """
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save texts
        with open(texts_path, 'wb') as f:
            pickle.dump(self.texts, f)
    
    def load(self, index_path: str, texts_path: str) -> None:
        """
        Load the FAISS index and texts from disk.
        
        Args:
            index_path: Path to FAISS index file
            texts_path: Path to texts pickle file
            
        Raises:
            FileNotFoundError: If files don't exist
        """
        if not Path(index_path).exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        if not Path(texts_path).exists():
            raise FileNotFoundError(f"Texts file not found: {texts_path}")
        
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        self.dimension = self.index.d
        
        # Load texts
        with open(texts_path, 'rb') as f:
            self.texts = pickle.load(f)
    
    def get_size(self) -> int:
        """
        Get the number of vectors stored.
        
        Returns:
            Number of stored vectors
        """
        return self.index.ntotal
