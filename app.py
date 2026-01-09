"""
Gradio UI Application

This module provides a web interface for the AI Study Partner application.
Users can upload PDFs, ask questions, and get answers grounded in the uploaded content.
"""

import gradio as gr
from pathlib import Path
from rag_pipeline import RAGPipeline


# Global pipeline instance
pipeline = None


def initialize_pipeline():
    """Initialize the RAG pipeline."""
    global pipeline
    if pipeline is None:
        pipeline = RAGPipeline()
    return pipeline


def process_pdf_and_index(pdf_file) -> str:
    """
    Process uploaded PDF and index it.
    
    Args:
        pdf_file: Gradio file object (can be filepath string or file object)
        
    Returns:
        Status message
    """
    if pdf_file is None:
        return "Please upload a PDF file."
    
    try:
        pipeline = initialize_pipeline()
        
        # Handle Gradio file object (can be string path or object)
        if isinstance(pdf_file, str):
            pdf_path = pdf_file
        else:
            pdf_path = pdf_file.name if hasattr(pdf_file, 'name') else str(pdf_file)
        
        # Process PDF directly (Gradio handles temp file)
        chunks, full_text = pipeline.process_pdf(pdf_path)
        
        # Index documents
        pipeline.index_documents(chunks)
        
        return f"✓ PDF processed successfully!\n\n- Extracted {len(full_text)} characters\n- Created {len(chunks)} chunks\n- Indexed and ready for questions"
    
    except Exception as e:
        return f"Error processing PDF: {str(e)}"


def answer_question(question: str) -> tuple:
    """
    Answer a question using the RAG pipeline.
    
    Args:
        question: User question
        
    Returns:
        Tuple of (answer, retrieved_context)
    """
    if not question or not question.strip():
        return "Please enter a question.", ""
    
    try:
        pipeline = initialize_pipeline()
        
        # Get answer and retrieved chunks
        answer, retrieved_chunks = pipeline.answer_question(question, k=5, min_similarity=0.3)
        
        # Format retrieved context for display
        context_display = ""
        if retrieved_chunks:
            context_display = "Retrieved Context:\n\n"
            for i, (chunk, score) in enumerate(retrieved_chunks, 1):
                context_display += f"[Chunk {i}, Similarity: {score:.3f}]\n{chunk}\n\n"
        else:
            context_display = "No relevant context retrieved."
        
        return answer, context_display
    
    except Exception as e:
        return f"Error answering question: {str(e)}", ""


def summarize_documents(pdf_file) -> str:
    """
    Summarize uploaded PDF.
    
    Args:
        pdf_file: Gradio file object (can be filepath string or file object)
        
    Returns:
        Summary text
    """
    if pdf_file is None:
        return "Please upload a PDF file first."
    
    try:
        pipeline = initialize_pipeline()
        
        # Handle Gradio file object (can be string path or object)
        if isinstance(pdf_file, str):
            pdf_path = pdf_file
        else:
            pdf_path = pdf_file.name if hasattr(pdf_file, 'name') else str(pdf_file)
        
        # Process PDF directly (Gradio handles temp file)
        chunks, full_text = pipeline.process_pdf(pdf_path)
        
        # Generate summary
        summary = pipeline.summarize_documents(full_text)
        
        return summary
    
    except Exception as e:
        return f"Error generating summary: {str(e)}"


def create_interface():
    """Create and launch the Gradio interface."""
    
    with gr.Blocks(title="AI Study Partner", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            """
            # 📚 AI Study Partner – Context-Aware Question Answering from PDFs
            
            Upload your lecture notes (PDFs) and ask questions. The system will answer **ONLY** using the uploaded content.
            If the answer is not present in the documents, it will say "I don't know."
            
            **How it works:**
            1. Upload a PDF file
            2. The system processes and indexes the content
            3. Ask questions about the content
            4. Get answers grounded in your uploaded documents
            """
        )
        
        with gr.Tab("📄 Upload & Process PDF"):
            gr.Markdown("### Step 1: Upload your PDF lecture notes")
            pdf_upload = gr.File(
                label="Upload PDF",
                file_types=[".pdf"],
                type="filepath"
            )
            process_btn = gr.Button("Process PDF", variant="primary")
            process_status = gr.Textbox(
                label="Status",
                lines=5,
                interactive=False
            )
            
            process_btn.click(
                fn=process_pdf_and_index,
                inputs=pdf_upload,
                outputs=process_status
            )
        
        with gr.Tab("❓ Ask Questions"):
            gr.Markdown("### Step 2: Ask questions about your uploaded content")
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="e.g., What is the main topic discussed in chapter 3?",
                lines=2
            )
            ask_btn = gr.Button("Get Answer", variant="primary")
            
            answer_output = gr.Textbox(
                label="Answer",
                lines=5,
                interactive=False
            )
            context_output = gr.Textbox(
                label="Retrieved Context (for reference)",
                lines=10,
                interactive=False
            )
            
            ask_btn.click(
                fn=answer_question,
                inputs=question_input,
                outputs=[answer_output, context_output]
            )
        
        with gr.Tab("📝 Summarize"):
            gr.Markdown("### Generate a summary of your uploaded PDF")
            summary_pdf_upload = gr.File(
                label="Upload PDF to Summarize",
                file_types=[".pdf"],
                type="filepath"
            )
            summarize_btn = gr.Button("Generate Summary", variant="primary")
            summary_output = gr.Textbox(
                label="Summary",
                lines=10,
                interactive=False
            )
            
            summarize_btn.click(
                fn=summarize_documents,
                inputs=summary_pdf_upload,
                outputs=summary_output
            )
        
        with gr.Tab("ℹ️ About"):
            gr.Markdown(
                """
                ## About This Application
                
                This is a **Retrieval-Augmented Generation (RAG)** system that answers questions
                based solely on uploaded PDF documents.
                
                ### Key Features:
                - **Context-Aware**: Answers are generated only from uploaded content
                - **No Hallucination**: System explicitly says "I don't know" when information is not available
                - **Semantic Search**: Uses embeddings for better understanding than keyword matching
                - **Summarization**: Generate quick overviews of your notes
                
                ### Technical Stack:
                - **PyTorch**: Deep learning framework
                - **HuggingFace Transformers**: Pre-trained models
                - **SentenceTransformers**: Semantic embeddings
                - **FAISS**: Vector similarity search
                - **Gradio**: Web interface
                
                ### How It Works:
                1. **PDF Processing**: Extracts text from PDFs
                2. **Chunking**: Splits text into manageable chunks (~500 tokens)
                3. **Embeddings**: Converts chunks to semantic vectors
                4. **Indexing**: Stores vectors in FAISS for fast search
                5. **Retrieval**: Finds most relevant chunks for a question
                6. **Generation**: Generates answer from retrieved context
                
                See README.md for detailed explanations of each component.
                """
            )
    
    return app


if __name__ == "__main__":
    # Initialize pipeline
    initialize_pipeline()
    
    # Create and launch interface
    app = create_interface()
    app.launch(share=False, server_name="0.0.0.0", server_port=7860)
