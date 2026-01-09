"""
Example Usage Script

This script demonstrates how to use the RAG pipeline programmatically
(without the Gradio UI). Useful for testing and integration.
"""

from rag_pipeline import RAGPipeline


def main():
    """Example usage of the RAG pipeline."""
    
    # Initialize pipeline
    print("Initializing RAG Pipeline...")
    pipeline = RAGPipeline()
    
    # Example: Process a PDF
    pdf_path = "example_notes.pdf"  # Replace with your PDF path
    
    try:
        print(f"\nProcessing PDF: {pdf_path}")
        chunks, full_text = pipeline.process_pdf(pdf_path)
        print(f"Created {len(chunks)} chunks from {len(full_text)} characters")
        
        # Index documents
        print("\nIndexing documents...")
        pipeline.index_documents(chunks)
        print("Documents indexed successfully!")
        
        # Ask questions
        questions = [
            "What is the main topic of this document?",
            "What are the key concepts discussed?",
            "Summarize the important points."
        ]
        
        print("\n" + "="*60)
        print("ANSWERING QUESTIONS")
        print("="*60)
        
        for question in questions:
            print(f"\nQuestion: {question}")
            answer, retrieved_chunks = pipeline.answer_question(question, k=3)
            print(f"Answer: {answer}")
            print(f"\nRetrieved {len(retrieved_chunks)} relevant chunks")
            for i, (chunk, score) in enumerate(retrieved_chunks, 1):
                print(f"  Chunk {i} (similarity: {score:.3f}): {chunk[:100]}...")
        
        # Generate summary
        print("\n" + "="*60)
        print("GENERATING SUMMARY")
        print("="*60)
        summary = pipeline.summarize_documents(full_text)
        print(f"\nSummary:\n{summary}")
        
        # Save index for later use
        print("\nSaving index...")
        pipeline.save_index("faiss_index.idx", "texts.pkl")
        print("Index saved successfully!")
        
    except FileNotFoundError:
        print(f"Error: PDF file '{pdf_path}' not found.")
        print("Please provide a valid PDF path or create an example PDF.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
