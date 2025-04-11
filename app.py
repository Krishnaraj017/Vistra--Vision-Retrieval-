from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import uuid
import base64
from typing import List, Dict, Any, Union, Optional
from IPython.display import Image, display
from unstructured.partition.pdf import partition_pdf
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
import torch

from multi_modal_main import MultiModalRAG
from fastapi.middleware.cors import CORSMiddleware  # Add this import at the top

app = FastAPI()
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all  origins - for development only!
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize the MultiModalRAG instance
rag = MultiModalRAG()

class QuestionRequest(BaseModel):
    question: str
    return_sources: bool = False
    return_images: bool = True

class ImageResponse(BaseModel):
    data: str
    content_type: str = "image/jpeg"

class RAGResponse(BaseModel):
    text_response: str
    images: Optional[List[ImageResponse]] = None
    sources: Optional[Dict[str, Any]] = None

def make_serializable(obj):
    """Convert unstructured document elements to serializable dictionaries."""
    from unstructured.documents.elements import Element, CompositeElement
    
    if isinstance(obj, CompositeElement) or isinstance(obj, Element):
        # Convert Element objects to dictionaries with their relevant attributes
        return {
            "element_type": obj.__class__.__name__,
            "text": str(obj),
            "metadata": obj.metadata.to_dict() if hasattr(obj, "metadata") else {}
        }
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    else:
        return obj

# Extend the MultiModalRAG class with a method to retrieve relevant images
def get_relevant_images(rag_instance, question, docs_retrieved):
    """
    Get images relevant to the question based on retrieved documents.
    
    Args:
        rag_instance: The MultiModalRAG instance
        question: The user's question
        docs_retrieved: The retrieved documents from the vectorstore
        
    Returns:
        List of base64-encoded images
    """
    relevant_images = []
    
    # Parse docs to separate images from text
    docs_by_type = rag_instance._parse_docs(docs_retrieved)
    
    # Add direct image matches from retrieval
    if docs_by_type["images"]:
        relevant_images.extend(docs_by_type["images"])
    
    # Check if the question explicitly asks for images
    image_keywords = ["image", "picture", "figure", "diagram", "chart", "photo", "illustration", "graph"]
    if any(keyword in question.lower() for keyword in image_keywords):
        # If asking for images but none retrieved, add some images from the document
        if not relevant_images and rag_instance.images:
            # Limit to first 3 images to avoid overwhelming response
            relevant_images.extend(rag_instance.images[:3])
    
    return relevant_images

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        file_path = f"/tmp/{file.filename}"
        with open(file_path, "wb") as buffer:
            buffer.write(file.file.read())
        
        # Process the PDF
        rag.process_pdf(file_path)
        rag.generate_summaries()
        rag.index_content()
        rag.setup_chain()
        
        # Clean up the temporary file
        os.remove(file_path)
        
        return JSONResponse(content={
            "message": "PDF processed successfully",
            "stats": {
                "text_chunks": len(rag.texts),
                "tables": len(rag.tables),
                "images": len(rag.images)
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask-question/", response_model=RAGResponse)
async def ask_question(request: QuestionRequest):
    try:
        # Get raw docs first to access the retrieved documents
        if request.return_sources or request.return_images:
            raw_result = rag.chain_with_sources.invoke(request.question)
            text_response = raw_result["response"]
            
            # Get the retrieved documents
            retrieved_docs = raw_result["context"]["texts"] if isinstance(raw_result["context"], dict) and "texts" in raw_result["context"] else []
            retrieved_raw = [doc for doc in raw_result["context"] if isinstance(doc, str)]
            
            # Get relevant images if requested
            images_data = None
            if request.return_images:
                relevant_images = get_relevant_images(rag, request.question, retrieved_raw)
                if relevant_images:
                    images_data = [
                        ImageResponse(data=img) for img in relevant_images
                    ]
            
            # Make context serializable before returning
            serializable_context = make_serializable(raw_result["context"]) if request.return_sources else None
            
            # Update memory with the exchange
            rag._save_to_memory(request.question, text_response)
            
            # Prepare the response
            response = RAGResponse(
                text_response=text_response,
                images=images_data,
                sources=serializable_context
            )
            
            return response
        else:
            # Just get the answer without sources or images
            text_response = rag.chain.invoke(request.question)
            rag._save_to_memory(request.question, text_response)
            
            return RAGResponse(text_response=text_response)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/clear-memory/")
async def clear_memory():
    try:
        rag.clear_memory()
        return JSONResponse(content={"message": "Conversation memory cleared successfully"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list-images/")
async def list_images():
    try:
        return JSONResponse(content={
            "image_count": len(rag.images),
            "image_descriptions": rag.image_summaries if hasattr(rag, "image_summaries") else []
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)