import os
import json
import re
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from jsonschema import ValidationError

from multi_modal_main import MultiModalRAG
from backend.schemas import QuestionRequest, RAGResponse, RAGTextResponse, ImageResponse

import PyPDF2  # You'll need this to count pages

router = APIRouter()
rag = MultiModalRAG()

@router.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        file_path = f"/tmp/{file.filename}"
        # check page count
        # Save the uploaded file to disk first
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Check page count
        with open(file_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            page_count = len(pdf_reader.pages)

        if page_count > 4:
            os.remove(file_path)
            return JSONResponse(
            status_code=400,
            content={
            "message": "Please upload a PDF with 2 pages or less",
            "stats": {
                "text_chunks": 0,
                "tables": 0,
                "images":0,
            }
            }
        )
        
       
        # Process the PDF
        rag.process_pdf(file_path)
        rag.generate_summaries()
        rag.index_content()
        rag.setup_chain()
        
        # Clean up the temporary file
        os.remove(file_path)
        
        return JSONResponse(
            status_code=200,
            content={
            "message": "PDF processed successfully",
            "stats": {
                "text_chunks": len(rag.texts),
                "tables": len(rag.tables),
                "images": len(rag.images)
            }
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
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
    
def extract_json(text: str) -> str:
    # Extracts the first valid JSON block from markdown or raw string
    match = re.search(r'{.*}', text, re.DOTALL)
    return match.group(0) if match else text

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


@router.post("/ask-question/", response_model=RAGResponse)
async def ask_question(request: QuestionRequest):
    try:
        # Get raw docs first to access the retrieved documents
        if request.return_sources or request.return_images:
            raw_result = rag.chain_with_sources.invoke(request.question)
            text_response = raw_result["response"]
            
            # Get the retrieved documents
            retrieved_docs = raw_result["context"]["texts"]
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
            try:
                raw_response = rag.chain.invoke(request.question)
                print(raw_response)
                cleaned_json = extract_json(raw_response)

                parsed_json = json.loads(cleaned_json)
                print(parsed_json)
                structured_response = RAGTextResponse(**parsed_json)
                print(structured_response)
            except (json.JSONDecodeError, ValidationError) as e:
                raise HTTPException(status_code=500, detail=str(e))

            return RAGResponse(text_response=structured_response)


            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/clear-memory/")
async def clear_memory():
    try:
        rag.clear_memory()
        return JSONResponse(content={"message": "Conversation memory cleared successfully"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list-images/")
async def list_images():
    try:
        return JSONResponse(content={
            "image_count": len(rag.images),
            "image_descriptions": rag.image_summaries if hasattr(rag, "image_summaries") else []
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @router.post("/upload-pdf/")
# async def upload_pdf(file: UploadFile = File(...)):
#     try:
#         # Save the uploaded file temporarily
#         file_path = f"/tmp/{file.filename}"
#         with open(file_path, "wb") as buffer:
#             buffer.write(file.file.read())
        
#         # Process the PDF
#         stats = rag_service.process_pdf(file_path)
        
#         # Clean up the temporary file
#         os.remove(file_path)
        
#         return JSONResponse(content={
#             "message": "PDF processed successfully",
#             "stats": stats
#         })
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @router.post("/ask-question/", response_model=RAGResponse)
# async def ask_question(request: QuestionRequest):
#     try:
#         result = rag_service.ask_question(
#             question=request.question,
#             return_sources=request.return_sources,
#             return_images=request.return_images
#         )
        
#         # Format the response according to the response model
#         if "images" in result and result["images"]:
#             images_data = [
#                 ImageResponse(data=img) for img in result["images"]
#             ]
#         else:
#             images_data = None
            
#         try:
#             if isinstance(result["text_response"], dict):
#                 structured_response = RAGTextResponse(**result["text_response"])
#             else:
#                 # It's already a string, we need to parse it as JSON
#                 structured_response = result["text_response"]
#         except (json.JSONDecodeError, ValidationError) as e:
#             raise HTTPException(status_code=500, detail=f"Error parsing response: {str(e)}")
        
#         return RAGResponse(
#             text_response=structured_response,
#             images=images_data,
#             sources=result.get("sources")
#         )
            
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @router.get("/clear-memory/")
# async def clear_memory():
#     try:
#         result = rag_service.clear_memory()
#         return JSONResponse(content=result)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @router.get("/list-images/")
# async def list_images():
#     try:
#         result = rag_service.list_images()
#         return JSONResponse(content=result)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

