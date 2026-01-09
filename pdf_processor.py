"""
PDF Text Extraction Module

This module handles extraction of text from PDF files using PyPDF2.
It processes multi-page PDFs and cleans malformed text.
"""

import re
from typing import List, Optional
from pathlib import Path
import PyPDF2


class PDFProcessor:
    """
    Handles PDF text extraction and cleaning.
    
    Attributes:
        None
    """
    
    def __init__(self):
        """Initialize the PDF processor."""
        pass
    
    def extract_text(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as a single string
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            ValueError: If PDF cannot be read or is empty
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                if len(pdf_reader.pages) == 0:
                    raise ValueError("PDF file has no pages")
                
                text_parts = []
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                    except Exception as e:
                        print(f"Warning: Could not extract text from page {page_num + 1}: {e}")
                        continue
                
                if not text_parts:
                    raise ValueError("No text could be extracted from PDF")
                
                full_text = "\n\n".join(text_parts)
                cleaned_text = self._clean_text(full_text)
                
                return cleaned_text
                
        except PyPDF2.errors.PdfReadError as e:
            raise ValueError(f"Error reading PDF: {e}")
        except Exception as e:
            raise ValueError(f"Unexpected error processing PDF: {e}")
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing excessive whitespace and fixing common issues.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace (more than 2 consecutive spaces)
        text = re.sub(r' {3,}', ' ', text)
        
        # Remove excessive newlines (more than 2 consecutive)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Fix common PDF extraction issues (hyphenated words across lines)
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        # Remove special characters that might cause issues
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
