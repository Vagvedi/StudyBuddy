"""
Embeddings Module

This module handles text embeddings using SentenceTransformers.
Uses 'all-MiniLM-L6-v2' model for efficient semantic representations.

Why Embeddings Instead of Keyword Search?
- Semantic understanding: "car" and "automobile" are similar
- Context awareness: "bank" (financial) vs "bank" (river)
- Handles synonyms and paraphrasing
- Better for conceptual queries
"""

import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    """
    Generates embeddings for text chunks using SentenceTransformers.
    
    Attributes:
        model: SentenceTransformer model instance
        model_name: Name of the model being used
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the SentenceTransformer model (default: all-MiniLM-L6-v2)
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        print("Embedding model loaded successfully")
    
    def generate_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for input text(s).
        
        Args:
            texts: Single text string or list of text strings
            
        Returns:
            Numpy array of embeddings (normalized)
            Shape: (n_texts, embedding_dim) for list, (embedding_dim,) for single text
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            raise ValueError("Empty text list provided")
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Normalize for cosine similarity
            show_progress_bar=len(texts) > 10
        )
        
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.
        
        Returns:
            Embedding dimension
        """
        # all-MiniLM-L6-v2 produces 384-dimensional embeddings
        test_embedding = self.generate_embeddings("test")
        return test_embedding.shape[-1]
