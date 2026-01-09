"""
Answer Generation Module

This module generates answers using a language model with strict prompting
to ensure answers are grounded only in retrieved context.
Uses google/flan-t5-base for answer generation.
"""

from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class AnswerGenerator:
    """
    Generates answers from retrieved context using a language model.
    
    Attributes:
        tokenizer: Tokenizer for the model
        model: Language model for generation
        device: Device (CPU or CUDA)
    """
    
    def __init__(self, model_name: str = "google/flan-t5-base"):
        """
        Initialize the answer generator.
        
        Args:
            model_name: Name of the model to use (default: google/flan-t5-base)
        """
        print(f"Loading answer generation model: {model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"Answer generation model loaded on {self.device}")
    
    def generate_answer(
        self,
        question: str,
        context_chunks: List[Tuple[str, float]],
        max_length: int = 512
    ) -> str:
        """
        Generate an answer from retrieved context.
        
        Uses strict prompting to ensure the model only answers from context.
        If context is insufficient, returns "I don't know."
        
        Args:
            question: User question
            context_chunks: List of (chunk_text, similarity_score) tuples
            max_length: Maximum generation length
            
        Returns:
            Generated answer string
        """
        if not context_chunks:
            return "I don't know. The uploaded documents don't contain relevant information to answer this question."
        
        # Combine context chunks
        context_text = "\n\n".join([chunk for chunk, _ in context_chunks])
        
        # Create strict prompt to prevent hallucination
        prompt = self._create_prompt(question, context_text)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        # Generate answer
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                do_sample=False
            )
        
        # Decode answer
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Post-process to ensure it's not hallucinating
        answer = self._validate_answer(answer, question, context_text)
        
        return answer
    
    def _create_prompt(self, question: str, context: str) -> str:
        """
        Create a strict prompt that prevents hallucination.
        
        Args:
            question: User question
            context: Retrieved context text
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""Answer the question using ONLY the information provided in the context below. 
If the context does not contain enough information to answer the question, respond with "I don't know."

Context:
{context}

Question: {question}

Answer (using only the context above):"""
        
        return prompt
    
    def _validate_answer(self, answer: str, question: str, context: str) -> str:
        """
        Validate that the answer is reasonable and not hallucinated.
        
        Args:
            answer: Generated answer
            question: Original question
            context: Context used for generation
            
        Returns:
            Validated answer (or "I don't know" if validation fails)
        """
        answer_lower = answer.lower().strip()
        
        # Check for explicit "don't know" responses
        dont_know_phrases = [
            "i don't know",
            "i do not know",
            "cannot answer",
            "not in the context",
            "not provided",
            "not mentioned"
        ]
        
        if any(phrase in answer_lower for phrase in dont_know_phrases):
            return "I don't know. The uploaded documents don't contain relevant information to answer this question."
        
        # If answer is too short or seems incomplete, be cautious
        if len(answer.strip()) < 5:
            return "I don't know. The uploaded documents don't contain relevant information to answer this question."
        
        # If answer seems to be repeating the question, it's likely not a real answer
        if answer_lower == question.lower() or answer_lower in question.lower():
            return "I don't know. The uploaded documents don't contain relevant information to answer this question."
        
        return answer
