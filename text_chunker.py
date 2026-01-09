"""
Text Chunking Module

This module implements sentence-aware text chunking for RAG pipelines.
Chunks are created with ~500 tokens and ~100 token overlap to preserve context.
"""

import re
from typing import List, Tuple
from transformers import AutoTokenizer


class TextChunker:
    """
    Handles text chunking with sentence awareness and token-based sizing.
    
    Attributes:
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens
        tokenizer: Tokenizer for counting tokens
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100, model_name: str = "google/flan-t5-base"):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Target size of each chunk in tokens (default: 500)
            chunk_overlap: Overlap between chunks in tokens (default: 100)
            model_name: Model name for tokenizer (default: flan-t5-base)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks with sentence awareness.
        
        Strategy:
        1. Split text into sentences
        2. Group sentences into chunks that don't exceed chunk_size
        3. Add overlap between chunks to preserve context
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []
        
        # Split into sentences (preserve sentence boundaries)
        sentences = self._split_sentences(text)
        
        if not sentences:
            return [text]
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = len(self.tokenizer.encode(sentence, add_special_tokens=False))
            
            # If adding this sentence would exceed chunk size, finalize current chunk
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)
                
                # Start new chunk with overlap
                current_chunk, current_tokens = self._create_overlap_chunk(chunk_text)
            
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        
        # Add final chunk if it exists
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)
        
        # Ensure we have at least one chunk
        if not chunks:
            chunks = [text]
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Pattern to match sentence endings (., !, ?) followed by space or newline
        # Handles common abbreviations and decimal numbers
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        
        sentences = re.split(sentence_pattern, text)
        
        # Clean up sentences
        cleaned_sentences = []
        for sent in sentences:
            sent = sent.strip()
            if sent:
                cleaned_sentences.append(sent)
        
        return cleaned_sentences if cleaned_sentences else [text]
    
    def _create_overlap_chunk(self, chunk_text: str) -> Tuple[List[str], int]:
        """
        Create overlap for next chunk by taking last sentences from current chunk.
        
        Args:
            chunk_text: Current chunk text
            
        Returns:
            Tuple of (overlap sentences list, token count)
        """
        sentences = self._split_sentences(chunk_text)
        
        # Take last sentences that fit within overlap size
        overlap_sentences = []
        overlap_tokens = 0
        
        for sentence in reversed(sentences):
            sentence_tokens = len(self.tokenizer.encode(sentence, add_special_tokens=False))
            
            if overlap_tokens + sentence_tokens <= self.chunk_overlap:
                overlap_sentences.insert(0, sentence)
                overlap_tokens += sentence_tokens
            else:
                break
        
        return overlap_sentences, overlap_tokens
