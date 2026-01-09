"""
Summarization Module

This module provides text summarization using BART-large-CNN.
Summarizes uploaded lecture notes to provide quick overviews.
"""

from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class Summarizer:
    """
    Generates summaries of text using BART-large-CNN model.
    
    Attributes:
        tokenizer: Tokenizer for the model
        model: BART model for summarization
        device: Device (CPU or CUDA)
    """
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        """
        Initialize the summarizer.
        
        Args:
            model_name: Name of the model to use (default: facebook/bart-large-cnn)
        """
        print(f"Loading summarization model: {model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"Summarization model loaded on {self.device}")
    
    def summarize(self, text: str, max_length: int = 142, min_length: int = 56) -> str:
        """
        Generate a summary of the input text.
        
        Args:
            text: Input text to summarize
            max_length: Maximum length of summary
            min_length: Minimum length of summary
            
        Returns:
            Generated summary string
        """
        if not text or not text.strip():
            return "No text provided for summarization."
        
        # Handle long texts by chunking if necessary
        # BART has a max input length of 1024 tokens
        max_input_length = 1024
        
        # Tokenize to check length
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
            padding=True
        ).to(self.device)
        
        # Generate summary
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                early_stopping=True,
                length_penalty=2.0,
                do_sample=False
            )
        
        # Decode summary
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        return summary
    
    def summarize_chunks(self, chunks: List[str]) -> str:
        """
        Summarize multiple text chunks by combining them.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            Combined summary
        """
        if not chunks:
            return "No content to summarize."
        
        # Combine chunks with separators
        combined_text = "\n\n".join(chunks)
        
        # Summarize the combined text
        return self.summarize(combined_text)
